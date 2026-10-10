"""S35-E: ElasticTier against a simulated card (vaayu/elastic_tier.py).

The real ElasticTier, the real ExpertVRAMCache and the real planner; only the card is
simulated. What the card models, each measured on gpt-oss-20b 2026-10-10
(docs/superpowers/specs/2026-10-10-S35-E-elastic-tier-design.md §2.4):
  * a filled tier occupies `1 + rounding` times the bytes it counts;
  * evicting PART of a tier leaves holes -- only `returns` of the evicted bytes reach
    the driver at the next empty_cache() (13 % measured). Emptying the tier returns
    everything;
  * PyTorch's pool (`pool`) hides free bytes from the driver until empty_cache();
  * what the running request holds (`request`) is gone from driver-free VRAM and
    shows in the allocator's own counter.
"""
import pytest

from vaayu.elastic_tier import (ALLOCATOR_ALLOWANCE, GRANT_CHUNKS, ElasticTier,
                                build_elastic_tier)
from vaayu.persist.expert_cache import ExpertVRAMCache
from vaayu.vram_plan import (VramPlan, VramPlanError, VramShape, plan_vram,
                             reserve_floor_bytes)

GB = 1024 ** 3
MB = 1024 ** 2
FLOOR = int(3.5 * GB)            # vaayu_base._VRAM_RESERVE_FLOOR_BYTES: a hardware headroom

# gpt-oss-20b, read off its server-default load (plans/probes/s35e_out/shape-gptoss.log):
# eager attention, per-chunk prefill, 12 windowed layers stored at window length.
GPTOSS = dict(
    shape=VramShape(
        kv_per_token=24576, N=1, chunk=512, hidden=2880, n_heads=64, vocab=201088,
        elem_bytes=2, score_elem_bytes=6.03125, alloc_factor=1.0, head_bytes=1158266976,
        score_bytes_per_key=16384.0, kv_constant_bytes=28286976),
    plan=VramPlan(max_seq_len=33213, reserve_bytes=9416564832, pinning_budget=0,
                  s_max=59901, reason="gpt-oss-20b load log"),
    expert=13236480, base_experts=191)

# Qwen3-235B-like under sdpa (synthetic: the checkpoint is retired): 94 layers,
# 128 experts of 11.28 MiB, chunk-major retention. Its plan is derived, not copied.
_Q_SHAPE = VramShape(
    kv_per_token=192512, N=1, chunk=512, hidden=4096, n_heads=64, vocab=151936,
    elem_bytes=2, score_elem_bytes=10.9, alloc_factor=1.0, head_bytes=1244659712,
    resident_expert_bytes=128 * 11827937, layer_major_prefill=True)
QWEN235 = dict(
    shape=_Q_SHAPE,
    plan=plan_vram(
        free_vram=12 * GB, shape=_Q_SHAPE, min_resident=11827937,
        requested_max_seq_len=None, position_limit=2 ** 31 - 1, layer_slack=0,
        reserve_floor=reserve_floor_bytes(shape=_Q_SHAPE, base_floor=FLOOR, free_vram=12 * GB,
                                          min_resident=11827937, layer_slack=0)),
    expert=11827937, base_experts=320)

GEOMETRIES = {"gptoss_like": GPTOSS, "qwen235_like_sdpa": QWEN235}
ALL = list(GEOMETRIES)


class _Blob:
    def __init__(self, nbytes):
        self.nbytes = int(nbytes)


class CardTier(ExpertVRAMCache):
    """The real tier, plus what the card needs to know about it: the holes a partial
    eviction leaves behind."""

    def __init__(self, budget, card):
        super().__init__(budget)
        self.card, self.holes, self.fresh_holes, self._n = card, 0, 0, 0
        card.tier = self

    def footprint(self):
        return int((self.current_usage + self.holes) * (1.0 + self.card.rounding))

    def footprint_when_full(self):
        return int(max(self.current_usage + self.holes, self.budget_bytes)
                   * (1.0 + self.card.rounding))

    def _strand(self, before):
        self.holes += before - self.current_usage
        self.fresh_holes += before - self.current_usage
        if self.current_usage == 0:
            # nothing left to pin a segment: every hole goes back with the pool
            self.card.pool += int(self.holes * (1.0 + self.card.rounding))
            self.holes = self.fresh_holes = 0

    def set_budget(self, budget_bytes):
        before = self.current_usage
        n = super().set_budget(budget_bytes)
        self._strand(before)
        return n

    def clear(self):
        # dropped references sit in PyTorch's pool until empty_cache() returns them
        self.card.pool += self.footprint()
        super().clear()
        self.holes = self.fresh_holes = 0

    def net_evicts(self, n_bytes):
        """What the prefill net does on device (ExpertVRAMCache.evict_until_free, which
        needs CUDA): drop LRU entries worth n_bytes, counted in prefill_evictions.
        They leave holes."""
        before, freed = self.current_usage, 0
        while freed < n_bytes and self.cache:
            _key, entry = self.cache.popitem(last=False)
            size = self._group_bytes(entry)
            self.current_usage -= size
            freed += size
            self.prefill_evictions += 1
        self._strand(before)
        return freed

    def fill(self, expert_bytes, reuse=1.0):
        """Decode until the tier is full: entries up to the budget, none evicted.
        `reuse` is the share of each new entry that lands in a hole an earlier
        eviction left; the rest is new driver memory."""
        while self.current_usage + expert_bytes <= self.budget_bytes:
            self._n += 1
            assert self.put(0, self._n, {"w": {"t": _Blob(expert_bytes)}})
            self.holes = max(0, self.holes - int(expert_bytes * reuse))
            self.fresh_holes = min(self.fresh_holes, self.holes)


class Card:
    """Bytes this process can get from the driver: `total`, less what `other` holders,
    the running `request`, PyTorch's `pool` and the tier's footprint take."""

    def __init__(self, total, rounding=0.012, returns=0.13):
        self.total, self.rounding, self.returns = int(total), rounding, returns
        self.other = self.request = self.pool = 0
        self.pinned = 4 * GB                 # weights resident since the load
        self.empties = 0
        self.tier = None

    def driver_free(self):
        return self.total - self.other - self.request - self.pool - self.tier.footprint()

    def allocated(self):
        """Live tensors in this process: pinned weights, the request, the tier."""
        return self.pinned + self.request + self.tier.current_usage

    def empty_cache(self):
        self.empties += 1
        self.pool = 0
        back = int(self.tier.fresh_holes * self.returns)
        self.tier.holes -= back
        self.tier.fresh_holes = 0


def rig(name, *, rounding=0.012, returns=0.13, chunk=0):
    """(card, tier, elastic, budgets) for one geometry. The tier starts EMPTY, with
    its static budget, on a card that has exactly the plan's reserve free beside that
    budget -- as a load leaves it."""
    geo = GEOMETRIES[name]
    base = geo["base_experts"] * geo["expert"]
    card = Card(base + geo["plan"].reserve_bytes, rounding=rounding, returns=returns)
    tier = CardTier(base, card)
    budgets = []
    elastic = ElasticTier(tier=tier, plan=geo["plan"], shape=geo["shape"], base_floor=FLOOR,
                          expert_bytes=geo["expert"], driver_free=card.driver_free,
                          empty_cache=card.empty_cache, allocated=card.allocated,
                          on_budget=budgets.append, prefill_chunk_size=chunk)
    return card, tier, elastic, budgets


def assert_covered(card, tier, elastic, live=0):
    """THE invariant: with the tier grown to its budget, this process can still get
    what the granted request may yet take. PyTorch's pool counts: it is the request's
    own reusable memory, and the allocator returns it when a new block needs it."""
    need = elastic.reserve_for(elastic.granted) - live
    worst = card.total - card.other - card.request - tier.footprint_when_full()
    assert worst >= need, (worst - need, tier.budget_bytes)


def finish(card, tier, elastic, leftover_pool=GB, reuse=1.0):
    """The request ends: decode filled the tier, and its transients sit in the pool."""
    tier.fill(elastic.expert_bytes, reuse=reuse)
    card.request, card.pool = 0, leftover_pool


SHORT = 2048


@pytest.mark.parametrize("name", ALL)
def test_a_short_request_grows_the_tier_and_flushes_nothing(name):
    card, tier, elastic, budgets = rig(name)
    peak = elastic.admit(SHORT, SHORT + 64)
    assert peak > 0 and elastic.covering and elastic.granted == SHORT + 64
    lend = elastic.plan.reserve_bytes - elastic.reserve_for(SHORT + 64)
    assert lend > 100 * elastic.expert_bytes, "the geometry must have something to lend"
    assert elastic.base < tier.budget_bytes <= elastic.base + lend
    # the allowance is held back from everything the tier could otherwise take
    assert tier.budget_bytes == int((elastic.base + lend) * (1 - ALLOCATOR_ALLOWANCE))
    assert elastic.flushes == 0 and budgets[-1] == tier.budget_bytes
    assert_covered(card, tier, elastic)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("returns", [0.0, 0.13, 0.78])
def test_a_longer_request_on_a_full_tier_empties_it_and_fits(name, returns):
    """Evicting the entries a longer request needs back would hand the driver a
    fraction of them (`returns`). The tier is emptied instead, which returns
    everything whatever the fraction; it refills by demand."""
    card, tier, elastic, _ = rig(name, returns=returns)
    elastic.admit(SHORT, SHORT + 64)
    finish(card, tier, elastic)
    full = tier.current_usage
    limit = elastic.plan.max_seq_len
    elastic.admit(limit - 64, limit)
    assert elastic.flushes == 1 and elastic.flushed_bytes == full
    assert tier.current_usage == 0 and tier.holes == 0
    assert tier.budget_bytes <= elastic.base + MB
    assert_covered(card, tier, elastic)


@pytest.mark.parametrize("name", ALL)
def test_a_tier_over_the_plans_cap_is_emptied_even_when_the_driver_has_room(name):
    """A card with more free VRAM than the plan counted: a longer request finds the
    driver already showing its need, but the tier holds more than the plan lets it
    hold for that request. It is emptied, not trimmed -- trimming is the partial
    eviction that leaves holes."""
    card, tier, elastic, _ = rig(name)
    lend = elastic.plan.reserve_bytes - elastic.reserve_for(SHORT + 64)
    card.total += lend + GB
    elastic.admit(SHORT, SHORT + 64)
    finish(card, tier, elastic)
    limit = elastic.plan.max_seq_len
    assert tier.current_usage > elastic.base + lend // 2
    assert card.driver_free() - card.pool >= 0 and (
        card.total - card.other - tier.footprint() >= elastic.reserve_for(limit)), \
        "the scenario needs the driver to show the long request's need with the tier full"
    elastic.admit(limit - 64, limit)
    assert elastic.flushes == 1 and elastic.unforced_flushes == 0
    assert tier.current_usage == 0 and tier.holes == 0
    assert tier.budget_bytes <= elastic.base
    assert_covered(card, tier, elastic)


@pytest.mark.parametrize("name", ALL)
def test_after_a_long_request_the_next_short_one_regrows_the_tier(name):
    card, tier, elastic, _ = rig(name)
    limit = elastic.plan.max_seq_len
    elastic.admit(limit - 64, limit)
    finish(card, tier, elastic)
    small = tier.budget_bytes
    elastic.admit(SHORT, SHORT + 64)
    assert tier.budget_bytes > small + 100 * elastic.expert_bytes
    assert elastic.flushes == 0, "growing must not evict"
    assert_covered(card, tier, elastic)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("rounding", [0.0, 0.012, 0.047])
def test_the_same_length_again_neither_flushes_nor_shrinks(name, rounding):
    """Measured rounding: 1.2 % behind a 2k prompt, 4.7 % behind a 16k one."""
    card, tier, elastic, _ = rig(name, rounding=rounding)
    elastic.admit(SHORT, SHORT + 64)
    previous = tier.budget_bytes
    for _ in range(6):
        finish(card, tier, elastic)
        usage = tier.current_usage
        elastic.admit(SHORT, SHORT + 64)
        assert elastic.flushes == 0 and tier.current_usage == usage
        # full to within one entry, and it stays there: the allowance is not eaten
        assert previous - elastic.expert_bytes < tier.budget_bytes <= previous
        assert_covered(card, tier, elastic)


@pytest.mark.parametrize("name", ALL)
def test_bytes_gone_for_good_cost_one_quiet_flush(name, capsys):
    """More than the allowance's slack disappears outside every budget (a process
    allocates some VRAM for good at its first requests). The next fit finds the
    driver short, empties the tier and sizes it around what is left. Once, quietly:
    no warning and no raised allowance for bytes that are simply gone."""
    card, tier, elastic, _ = rig(name)
    elastic.admit(SHORT, SHORT + 64)
    card.other += 700 * MB
    for _ in range(4):
        finish(card, tier, elastic)
        elastic.admit(SHORT, SHORT + 64)
        assert_covered(card, tier, elastic)
    assert elastic.unforced_flushes == 1 and elastic.flushes == 1
    assert elastic.allowance == ALLOCATOR_ALLOWANCE
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("name", ALL)
def test_bytes_gone_twice_teach_nothing_and_say_nothing(name, capsys):
    """A second unforced flush measures the tier. When what it measures is inside the
    allowance -- the shortfall was bytes taken from outside, not the tier's packing --
    the allowance stays and no warning claims otherwise."""
    card, tier, elastic, _ = rig(name)
    elastic.admit(SHORT, SHORT + 64)
    for _ in range(2):
        card.other += 700 * MB
        finish(card, tier, elastic)
        elastic.admit(SHORT, SHORT + 64)
        assert_covered(card, tier, elastic)
    assert elastic.unforced_flushes == 2
    assert elastic.allowance == ALLOCATOR_ALLOWANCE
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("rounding", [0.09, 0.25])
def test_a_tier_that_occupies_more_than_its_bytes_plus_the_allowance_is_learned(
        name, rounding, capsys):
    """Rounding over the allowance over-fills while decode fills the tier. The next
    fit -- before any prefill -- finds the driver short and empties it. The first
    time says nothing (bytes gone for good look the same); the second raises the
    allowance to what the flush measured and warns, and after that the tier fits
    when full and nothing is flushed again."""
    card, tier, elastic, _ = rig(name, rounding=rounding)
    elastic.admit(SHORT, SHORT + 64)
    for cycle in range(6):
        finish(card, tier, elastic)
        elastic.admit(SHORT, SHORT + 64)
        # at every fit, whatever the rounding: the driver shows the request its need
        assert card.driver_free() >= elastic.reserve_for(elastic.granted)
        if cycle >= 1:
            assert_covered(card, tier, elastic)
    assert elastic.unforced_flushes == elastic.flushes == 2
    # what was learned is what the flush measured: the share of a filled tier's
    # footprint that its own count does not see, plus the pad -- not "more than before"
    share = rounding / (1.0 + rounding)
    assert share < elastic.allowance <= share + 0.01
    assert capsys.readouterr().out.count("WARNING: elastic tier flushed") == 1


@pytest.mark.parametrize("name", ALL)
def test_a_flush_a_longer_request_forced_is_not_counted_as_unforced(name, capsys):
    """Two requests in a row that each need more than the last: each empties the tier,
    and neither says anything about the allocator."""
    card, tier, elastic, _ = rig(name)
    limit = elastic.plan.max_seq_len
    middle = (SHORT + limit) // 2
    assert (elastic.reserve_for(SHORT + 64) < elastic.reserve_for(middle)
            < elastic.reserve_for(limit)), "the geometry must be above its floor here"
    elastic.admit(SHORT, SHORT + 64)
    for length in (middle, limit):
        finish(card, tier, elastic)
        elastic.admit(length - 64, length)
        assert_covered(card, tier, elastic)
    assert elastic.flushes == 2 and elastic.unforced_flushes == 0
    assert elastic.allowance == ALLOCATOR_ALLOWANCE
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("reuse", [0.0, 0.5])
def test_a_flush_behind_the_nets_partial_eviction_teaches_nothing(name, reuse, capsys):
    """The prefill net evicts part of the tier and decode refills it without reusing
    all the holes: the tier now occupies much more than it counts, and the next
    request of the same length has to empty it. That is not the allocator's packing
    of a filled tier -- the flush removes the holes -- so it is neither counted as
    unforced nor learned, even with one unforced flush already on record."""
    card, tier, elastic, _ = rig(name)
    elastic.admit(SHORT, SHORT + 64)
    card.other += 700 * MB                           # the one quiet unforced flush
    finish(card, tier, elastic)
    elastic.admit(SHORT, SHORT + 64)
    assert elastic.unforced_flushes == 1
    finish(card, tier, elastic)
    tier.net_evicts(int(0.4 * tier.current_usage))   # during the next request's prefill
    assert tier.holes > 100 * elastic.expert_bytes
    finish(card, tier, elastic, reuse=reuse)
    assert tier.holes > 0
    elastic.admit(SHORT, SHORT + 64)
    assert elastic.flushes == 2, "the holes must cost this request a flush"
    assert elastic.unforced_flushes == 1 and elastic.allowance == ALLOCATOR_ALLOWANCE
    assert tier.holes == 0 and "WARNING" not in capsys.readouterr().out
    assert_covered(card, tier, elastic)
    # ...and with the holes gone the same length fits again without one
    finish(card, tier, elastic)
    elastic.admit(SHORT, SHORT + 64)
    assert elastic.flushes == 2
    # ...and the net's evictions are forgotten once a fit has seen them: a later
    # shortfall with no eviction behind it IS counted
    card.other += 700 * MB
    finish(card, tier, elastic)
    elastic.admit(SHORT, SHORT + 64)
    assert elastic.flushes == 3 and elastic.unforced_flushes == 2


@pytest.mark.parametrize("name", ALL)
def test_bytes_another_holder_took_come_out_of_the_tier(name):
    card, tier, elastic, _ = rig(name)
    card.other = GB
    elastic.admit(SHORT, SHORT + 64)
    lend = elastic.plan.reserve_bytes - elastic.reserve_for(SHORT + 64)
    assert tier.budget_bytes <= elastic.base + lend - int(GB * (1 - ALLOCATOR_ALLOWANCE)) + MB
    assert_covered(card, tier, elastic)


@pytest.mark.parametrize("name", ALL)
def test_the_tier_never_passes_the_plans_cap_on_a_roomier_card(name):
    """More free VRAM than the plan counted (something released since the load) is
    not the tier's to take: the budget stops at base + lend."""
    card, tier, elastic, _ = rig(name)
    card.total += 2 * GB
    elastic.admit(SHORT, SHORT + 64)
    lend = elastic.plan.reserve_bytes - elastic.reserve_for(SHORT + 64)
    assert tier.budget_bytes == elastic.base + lend
    assert_covered(card, tier, elastic)


@pytest.mark.parametrize("name", ALL)
def test_a_request_the_card_cannot_cover_is_refused_and_the_tier_kept(name):
    card, tier, elastic, _ = rig(name)
    elastic.admit(SHORT, SHORT + 64)
    finish(card, tier, elastic)
    usage, budget = tier.current_usage, tier.budget_bytes
    card.other = elastic.base + GB           # gone since the load: more than the tier holds back
    limit = elastic.plan.max_seq_len
    granted = elastic.granted
    with pytest.raises(VramPlanError) as err:
        elastic.admit(limit - 64, limit)
    assert f"Context of {limit} tokens does not fit" in str(err.value)
    assert "is all it can be given" in str(err.value)
    assert tier.current_usage == usage and tier.budget_bytes == budget
    assert elastic.flushes == 0 and elastic.granted == granted, "a refusal changes nothing"


@pytest.mark.parametrize("name", ALL)
def test_a_short_request_refused_for_the_floor_is_told_the_floor(name):
    """With less than the floor obtainable, a request whose own predicted peak would
    fit is still refused -- and the message must say what is short, not offer a
    longer context."""
    card, tier, elastic, _ = rig(name)
    card.other = elastic.base + elastic.plan.reserve_bytes - FLOOR + 100 * MB
    with pytest.raises(VramPlanError) as err:
        elastic.admit(SHORT, SHORT + 64)
    text = str(err.value)
    assert "3.50 GiB floor" in text and "largest context that fits" not in text
    assert not elastic.covering


@pytest.mark.parametrize("name", ALL)
def test_a_request_over_the_plan_is_refused_without_touching_the_card(name):
    card, tier, elastic, _ = rig(name)
    with pytest.raises(VramPlanError):
        elastic.admit(SHORT, 4 * elastic.plan.s_max)
    assert card.empties == 0 and tier.budget_bytes == elastic.base and not elastic.covering


@pytest.mark.parametrize("name", ALL)
def test_suspend_returns_to_the_static_plan(name):
    card, tier, elastic, budgets = rig(name)
    elastic.admit(SHORT, SHORT + 64)
    finish(card, tier, elastic)
    elastic.suspend()
    assert tier.budget_bytes == elastic.base and tier.current_usage == 0
    assert not elastic.covering and budgets[-1] == elastic.base
    assert card.driver_free() >= elastic.plan.reserve_bytes - card.pool
    card.pool, empties = GB, card.empties
    elastic.suspend()                                # nothing in the tier this time
    assert card.empties == empties + 1 and card.pool == 0


# ── stage 2: grants in steps, extended before the forward that needs them ──────────

STEP = GRANT_CHUNKS * 512


@pytest.mark.parametrize("name", ALL)
def test_a_request_without_max_tokens_is_granted_a_step_not_the_whole_window(name):
    card, tier, elastic, _ = rig(name, chunk=512)
    limit = elastic.plan.max_seq_len
    elastic.admit(SHORT - 100, limit)
    assert STEP == SHORT and elastic.granted == STEP
    assert tier.budget_bytes > elastic.base + 100 * elastic.expert_bytes
    assert_covered(card, tier, elastic)


@pytest.mark.parametrize("name", ALL)
def test_the_first_grant_covers_the_first_decode_forward(name):
    """A prompt that ends ON a grant step, or one token under it, must not make the
    first decode forward extend the grant: the driver is read there while the
    prefill's last blocks are still pinned (measured: 2.2 GiB short of the truth
    behind an 8k prompt)."""
    for prompt in (STEP, STEP - 1):
        card, tier, elastic, _ = rig(name, chunk=512)
        elastic.admit(prompt, elastic.plan.max_seq_len)
        assert elastic.granted >= prompt + 1
        elastic.cover(prompt)                        # the prefill forward
        elastic.cover(prompt + 1)                    # the first decode forward
        assert elastic.extensions == 0
    # ...and a reply capped right behind the prompt is granted whole
    card, tier, elastic, _ = rig(name, chunk=512)
    elastic.admit(STEP, STEP + 64)
    assert elastic.granted == STEP + 64


@pytest.mark.parametrize("name", ALL)
def test_the_grant_covers_every_forward_before_it_runs(name):
    card, tier, elastic, _ = rig(name, chunk=512)
    limit = elastic.plan.max_seq_len
    prompt = STEP - 10
    elastic.admit(prompt, limit)
    first = elastic.granted
    assert first == STEP
    kv_per_token = elastic.shape.kv_per_token
    crossed, context = 0, prompt
    end = min(limit, prompt + 3 * STEP + 5)
    elastic.cover(context)                              # the prefill forward
    assert elastic.extensions == 0
    while context < end:
        before = elastic.granted
        context += 1
        card.request = context * kv_per_token            # the request's KV, live
        card.pool = 256 * MB                             # transients between forwards
        elastic.cover(context)
        crossed += elastic.granted != before
        assert context <= elastic.granted <= limit
        assert elastic.granted % STEP == 0 or elastic.granted == limit
        assert_covered(card, tier, elastic, live=card.request)
        tier.fill(elastic.expert_bytes)
    # one extension, of one step, per grant step the context passed
    assert crossed == elastic.extensions == len(range(first, end, STEP)) == 3


@pytest.mark.parametrize("name", ALL)
def test_what_the_request_already_holds_is_not_reserved_a_second_time(name):
    """The request's reserve includes its KV and its transients. What it holds on the
    device when a grant is extended -- the prefill's logits are still referenced when
    the first decode forward starts (measured: the tier lost 140 experts to them
    before this was credited) -- is gone from driver-free VRAM, so the need the driver
    must still show is smaller by exactly that, and the tier's budget does not
    depend on it."""
    budgets = []
    for live in (0, 300 * MB, 2 * GB):
        card, tier, elastic, _ = rig(name, chunk=512)
        elastic.admit(SHORT - 1, elastic.plan.max_seq_len)
        assert elastic.granted == STEP
        card.request = live
        elastic.cover(STEP + 1)
        assert elastic.granted == 2 * STEP and elastic.flushes == 0
        assert_covered(card, tier, elastic, live=live)
        budgets.append(tier.budget_bytes)
    assert budgets[0] == budgets[1] == budgets[2]


@pytest.mark.parametrize("name", ALL)
def test_what_was_allocated_before_admission_is_not_the_requests(name):
    """Weights pinned at load, and anything else alive when the request is admitted,
    are not credited to it: only what it allocates afterwards is."""
    card, tier, elastic, _ = rig(name, chunk=512)
    card.pinned = 9 * GB
    elastic.admit(SHORT, elastic.plan.max_seq_len)
    assert elastic._live() == 0
    card.request = 123 * MB
    assert elastic._live() == 123 * MB
    tier.fill(elastic.expert_bytes)                  # the tier's entries are not the request's
    assert elastic._live() == 123 * MB


@pytest.mark.parametrize("name", ALL)
def test_a_warm_tier_is_not_mistaken_for_the_request(name):
    """The server's steady state: the tier is full from the last request when the
    next one is admitted. What that tier holds was allocated before admission and is
    not the request's; what the request allocates afterwards is, to the byte."""
    card, tier, elastic, _ = rig(name, chunk=512)
    limit = elastic.plan.max_seq_len
    elastic.admit(SHORT - 1, limit)
    finish(card, tier, elastic)
    assert tier.current_usage > elastic.base
    elastic.admit(SHORT - 1, limit)                  # admitted on the warm tier
    assert elastic._live() == 0
    card.request = 600 * MB
    assert elastic._live() == 600 * MB
    flushes = elastic.flushes
    elastic.cover(STEP + 1)
    assert elastic._live() == 600 * MB
    if elastic.reserve_for(2 * STEP) == elastic.reserve_for(STEP):
        # under the floor nothing more is needed: crediting the request keeps the tier
        assert elastic.flushes == flushes
    assert_covered(card, tier, elastic, live=card.request)


@pytest.mark.parametrize("name", ALL)
def test_a_request_past_its_own_reserve_gets_the_tier_nothing_more(name):
    """A request that already holds its whole predicted reserve has out-run the
    predictor. Whatever the driver still shows, the tier is not given any of it."""
    card, tier, elastic, _ = rig(name, chunk=512)
    card.total += 8 * GB
    elastic.admit(SHORT - 1, elastic.plan.max_seq_len)
    for _ in range(40):                              # a partly filled tier
        tier._n += 1
        tier.put(0, tier._n, {"w": {"t": _Blob(elastic.expert_bytes)}})
    usage = tier.current_usage
    card.request = elastic.reserve_for(2 * STEP) + GB
    elastic.cover(STEP + 1)
    assert elastic.flushes == 0 and tier.current_usage == usage
    assert tier.budget_bytes == usage


@pytest.mark.parametrize("name", ALL)
def test_past_its_limit_a_request_is_still_granted_in_steps(name):
    """A caller that runs forward beyond what it was admitted for re-fits once per
    step, not once per token."""
    card, tier, elastic, _ = rig(name, chunk=512)
    elastic.admit(SHORT - 1, SHORT + 10)
    elastic.cover(SHORT + 10)                        # up to its limit: the whole request
    assert elastic.granted == SHORT + 10 and elastic.extensions == 1
    empties = card.empties
    for context in range(SHORT + 11, SHORT + 21):    # ten forwards past the limit
        elastic.cover(context)
    assert elastic.extensions == 2 and elastic.granted == 2 * STEP
    assert card.empties - empties <= 2


@pytest.mark.parametrize("name", ALL)
def test_a_grant_never_passes_the_requests_own_limit(name):
    card, tier, elastic, _ = rig(name, chunk=512)
    elastic.admit(SHORT - 100, SHORT + 300)
    assert elastic.granted == SHORT
    elastic.cover(SHORT + 1)
    assert elastic.granted == SHORT + 300 and elastic.extensions == 1
    elastic.cover(SHORT + 300)
    assert elastic.extensions == 1


@pytest.mark.parametrize("name", ALL)
def test_an_unchunked_prefill_gets_the_whole_request_granted(name):
    card, tier, elastic, _ = rig(name, chunk=0)
    elastic.admit(SHORT, SHORT + 5000)
    assert elastic.granted == SHORT + 5000
    elastic.cover(SHORT + 4999)
    assert elastic.extensions == 0


@pytest.mark.parametrize("name", ALL)
def test_cover_does_nothing_for_a_request_it_did_not_admit(name):
    card, tier, elastic, budgets = rig(name, chunk=512)
    elastic.cover(20000)
    assert elastic.extensions == 0 and budgets == [] and tier.budget_bytes == elastic.base
    elastic.admit(SHORT, SHORT + 64)
    elastic.suspend()
    n = len(budgets)
    elastic.cover(20000)
    assert len(budgets) == n and card.empties > 0


@pytest.mark.parametrize("name", ALL)
def test_a_grant_the_driver_cannot_back_empties_the_tier_and_warns_once(name, capsys):
    card, tier, elastic, _ = rig(name, chunk=512)
    limit = elastic.plan.max_seq_len
    elastic.admit(SHORT - 1, limit)
    tier.fill(elastic.expert_bytes)
    card.other = elastic.plan.reserve_bytes          # another process, mid-request
    elastic.cover(STEP + 1)
    assert tier.budget_bytes == 0 and tier.current_usage == 0 and elastic.unbacked == 1
    elastic.cover(2 * STEP + 1)
    assert elastic.unbacked == 2
    assert capsys.readouterr().out.count("WARNING: the driver cannot back") == 1
    # the warning is per request: the next request that goes unbacked says so again
    card.other = 0
    elastic.admit(SHORT - 1, limit)
    card.other = elastic.plan.reserve_bytes
    elastic.cover(STEP + 1)
    assert elastic.unbacked == 3
    assert capsys.readouterr().out.count("WARNING: the driver cannot back") == 1


# ── the load-time decision ────────────────────────────────────────────────────────


def _build(name, **over):
    geo = GEOMETRIES[name]
    card = Card(16 * GB)
    args = dict(tier=CardTier(geo["base_experts"] * geo["expert"], card), plan=geo["plan"],
                shape=geo["shape"], base_floor=FLOOR, expert_bytes=geo["expert"],
                decode_graphs=False, driver_free=card.driver_free,
                empty_cache=card.empty_cache, allocated=card.allocated,
                prefill_chunk_size=512)
    args.update(over)
    return build_elastic_tier(**args)


@pytest.mark.parametrize("name", ALL)
def test_a_load_with_reserve_above_the_floor_gets_an_elastic_tier_and_one_line(name, capsys):
    elastic = _build(name)
    assert isinstance(elastic, ElasticTier) and elastic.grant_step == STEP
    out = capsys.readouterr().out
    assert out.count("elastic expert tier ON") == 1 and "3.50 GiB floor" in out


@pytest.mark.parametrize("name", ALL)
def test_a_load_whose_reserve_is_the_floor_gets_none_and_prints_nothing(name, capsys):
    geo = GEOMETRIES[name]
    at_floor = VramPlan(max_seq_len=8192, reserve_bytes=FLOOR, pinning_budget=0,
                        s_max=65536, reason="flash: the reserve is the floor")
    assert _build(name, plan=at_floor) is None
    # one byte under one expert of lendable reserve is still nothing to lend
    almost = VramPlan(max_seq_len=8192, reserve_bytes=FLOOR + geo["expert"] - 1,
                      pinning_budget=0, s_max=65536, reason="")
    assert _build(name, plan=almost) is None
    assert _build(name, tier=None) is None
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", ALL)
def test_decode_graphs_keep_the_tier_static_and_say_so(name, capsys):
    assert _build(name, decode_graphs=True) is None
    assert "elastic expert tier OFF: decode graphs" in capsys.readouterr().out


@pytest.mark.parametrize("name", ALL)
def test_a_speculator_keeps_the_tier_static_and_says_so(name, capsys):
    """speculative_generate sizes the tier's layer range for its verify width and
    restores it afterwards; a budget that moves with the request would overwrite it."""
    assert _build(name, speculator=True) is None
    out = capsys.readouterr().out
    assert out.count("elastic expert tier OFF: a speculator is loaded") == 1
    assert "elastic expert tier ON" not in out
