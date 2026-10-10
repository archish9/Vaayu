"""The elastic expert VRAM tier (S35-E).

The VRAM plan holds ONE reserve, sized for a request at max_seq_len, and the expert
VRAM tier gets what is left. A shorter request cannot use most of that reserve; this
module lends the unused part to the tier while that request runs and takes it back
before the request can need it.

No torch here: the three memory readings are injected, so every decision below is
tested on CPU against a simulated card (tests/test_elastic_tier.py).

What the design rests on, each measured on gpt-oss-20b 2026-10-10
(docs/superpowers/specs/2026-10-10-S35-E-elastic-tier-design.md §2.4):

* Evicting PART of a filled tier does not hand its bytes to the driver. Vaayu runs
  PyTorch's allocator with expandable segments, which gives memory back in 20 MiB
  units that are entirely free; evicted entries leave holes between live ones.
  Evicting 134 of 618 entries returned 13 % of their bytes; one allocation per entry
  instead of six changed nothing (111 entries, 22 %). Emptying the tier returns all
  of it. So a shrink here empties the tier, and the result is read off the driver.
* A filled tier occupies more driver memory than the bytes it counts -- the holes
  the request's own tensors leave between its entries. So the tier is never given
  all the headroom the driver shows (ALLOCATOR_ALLOWANCE), and what the driver shows
  is read again at every grant.
* A low driver-free reading is often the allocator's own cache, not demand. So the
  cache is returned before every reading.
"""
from __future__ import annotations

from .vram_plan import (VramPlan, VramPlanError, VramShape, _too_big_message,
                        predict_peak_vram, request_reserve_bytes)

# Share of what the tier and the driver's surplus add up to that the tier is NOT
# given, to start with: a filled tier occupies more driver memory than the bytes it
# counts. Measured on gpt-oss-20b, 2026-10-10: 1.1-1.3 % after a fill behind a 2k
# prompt, under 4 % behind an 8k one, 4.7 % behind a 16k one. A property of PyTorch's
# allocator and of tensor sizes, not of a model (CLAUDE.md §5), so it is a starting
# point and not a promise: a load that packs worse over-fills by the difference while
# the tier fills -- which is during decode, when the request's own demand is a small
# part of the PREFILL peak it was granted -- and the next fit(), which runs before
# any prefill or extension, finds the driver short and empties the tier; the second
# time that happens on a request that needed no more than the last one, it also
# raises the allowance to what it measured (ElasticTier.allowance).
ALLOCATOR_ALLOWANCE = 0.06

# A request is granted context in steps of this many prefill chunks, and the grant is
# extended before the forward that would pass it. A policy constant: larger steps mean
# fewer re-fits (each one above the floor empties the tier) and a slightly smaller tier.
GRANT_CHUNKS = 4

# What a flush teaches (see fit): the allowance becomes the share that flush measured
# plus this pad -- a budget sized to the last byte goes short on integer rounding --
# and never more than the cap. A reading that says a tier occupied twice the bytes it
# counted is not the allocator's packing: something else returned memory at the same
# moment, and holding back more than half the tier for it would be giving up.
_ALLOWANCE_PAD = 0.005
_ALLOWANCE_CAP = 0.5

_GIB = 1024.0 ** 3


def build_elastic_tier(*, tier, plan, shape, base_floor: int, expert_bytes: int,
                       decode_graphs: bool, driver_free, empty_cache, allocated,
                       on_budget=None, prefill_chunk_size: int = 0) -> "ElasticTier | None":
    """The load's ElasticTier, or None when there is nothing to lend.

    Silent when the plan's reserve is the base floor (every flash-kernel load): no
    line is printed and nothing else in this module runs. Decode graphs keep a
    private pool that empty_cache() has a known trap with (S12), and no checkpoint on
    disk has both, so that pair is refused -- with a line, because there the lend
    exists and is not used.
    """
    if tier is None or plan is None or shape is None:
        return None
    lendable = int(plan.reserve_bytes) - int(base_floor)
    if lendable < int(expert_bytes):
        return None
    if decode_graphs:
        print(f"[Vaayu-VRAM] elastic expert tier OFF: decode graphs are installed, and "
              f"their memory pool is not something this tier may return to the driver. "
              f"{lendable / _GIB:.2f} GiB of the reserve stays unlent.")
        return None
    elastic = ElasticTier(tier=tier, plan=plan, shape=shape, base_floor=base_floor,
                          expert_bytes=expert_bytes, driver_free=driver_free,
                          empty_cache=empty_cache, allocated=allocated,
                          on_budget=on_budget, prefill_chunk_size=prefill_chunk_size)
    print(f"[Vaayu-VRAM] elastic expert tier ON: up to {lendable / _GIB:.2f} GiB of the "
          f"{int(plan.reserve_bytes) / _GIB:.2f} GiB reserve (everything above the "
          f"{int(base_floor) / _GIB:.2f} GiB floor) is lent to the expert VRAM tier while "
          f"the running request is shorter than max_seq_len={int(plan.max_seq_len)}")
    return elastic


class ElasticTier:
    """Owns the expert VRAM tier's budget on a load that has reserve to lend.

    `base` is the tier's budget when this is built: the static entitlement, what the
    tier keeps while a request runs at max_seq_len. `granted` is the context length
    the running request has been promised VRAM for; the budget never exceeds
    base + (reserve - request_reserve(granted)).

    Three readings, injected: `driver_free()` -- bytes the driver reports free;
    `empty_cache()` -- return the allocator's cached blocks to the driver;
    `allocated()` -- bytes of live tensors in this process (the allocator's own
    counter). The last one says what the RUNNING request holds: everything allocated
    beyond the tier's entries and what was there when the request was admitted.
    """

    def __init__(self, *, tier, plan: VramPlan, shape: VramShape, base_floor: int,
                 expert_bytes: int, driver_free, empty_cache, allocated, on_budget=None,
                 prefill_chunk_size: int = 0):
        self.tier = tier
        self.plan = plan
        self.shape = shape
        self.base_floor = int(base_floor)
        self.expert_bytes = int(expert_bytes)
        self._driver_free = driver_free
        self._empty_cache = empty_cache
        self._allocated = allocated
        self._on_budget = on_budget
        self._idle_allocated = 0
        self.base = int(tier.budget_bytes)
        # 0 = whole-request grants: an unchunked prefill gives no step to derive.
        self.grant_step = GRANT_CHUNKS * max(0, int(prefill_chunk_size))
        self.granted = 0
        self.limit = 0
        self.covering = False
        self.grants = 0
        self.extensions = 0
        self.flushes = 0
        self.flushed_bytes = 0
        self.unforced_flushes = 0
        self.unbacked = 0
        self.allowance = ALLOCATOR_ALLOWANCE
        self._last_reserve = None
        self._tier_evictions_seen = self._tier_evictions()
        self._warned_unforced = False
        self._warned_unbacked = False

    # ── the rule ────────────────────────────────────────────────────────────────

    def reserve_for(self, context: int) -> int:
        return request_reserve_bytes(self.plan, self.shape, context, self.base_floor)

    def _grant_for(self, context: int) -> int:
        """The context length to grant a request that is about to hold `context`
        tokens: the next grant step at or above it, never past the request's limit;
        the whole request when there is no step. A context already past the limit
        (a caller that runs forward beyond what it was admitted for) is still granted
        in steps, so that it re-fits once per step and not once per token."""
        context = int(context)
        step = self.grant_step
        granted = context if step <= 0 else -(-context // step) * step
        if self.limit >= context:
            granted = self.limit if step <= 0 else min(granted, self.limit)
        return granted

    def _live(self) -> int:
        """Bytes the running request holds right now: its KV and whatever transients
        its caller still references (the prefill's logits are alive when the first
        decode forward starts). Its reserve already counts them, and they are gone
        from driver-free VRAM, so what the driver must still show is smaller by
        exactly this. Anything else the ALLOCATOR handed out for good since admission
        would be credited too; nothing does on the on-device models (the ~70 MiB a
        process gains after its first request is outside the allocator's count)."""
        return max(0, int(self._allocated()) - int(self.tier.current_usage)
                   - self._idle_allocated)

    # ── the budget ──────────────────────────────────────────────────────────────

    def _read(self) -> "tuple[int, int]":
        """(driver-free, tier usage) with the allocator's pool returned first: cached
        blocks are not lost VRAM, and a budget measured without returning them is
        short by exactly the pool (measured: 1.72 GiB after a 2k prefill)."""
        self._empty_cache()
        return int(self._driver_free()), int(self.tier.current_usage)

    def _tier_evictions(self) -> int:
        """How many entries something OTHER than a budget has evicted from the tier
        (the prefill net, an admission eviction): each leaves a hole."""
        return int(getattr(self.tier, "prefill_evictions", 0))

    def _set_budget(self, budget: int) -> None:
        self.tier.set_budget(int(budget))
        if self._on_budget is not None:
            self._on_budget(int(budget))

    def fit(self, granted: int, live: int = 0) -> bool:
        """Size the tier for a request granted `granted` tokens of context; `live` is
        what the request already holds on the device (_live()). Returns False when
        the driver cannot cover the request even with the tier empty (the budget is
        then 0).

        The only place the budget is written while this object exists.
        """
        reserve = self.reserve_for(granted)
        live = max(0, int(live))
        need = max(0, reserve - live)                 # what the driver must still show
        want = self.base + int(self.plan.reserve_bytes) - reserve   # the plan's cap
        before = int(self.tier.budget_bytes)
        free, usage = self._read()
        flushed = 0
        if (free < need or usage > want) and usage > 0:
            # The tier holds what this request needs. Evicting that much would not
            # hand it to the driver (see the module docstring); emptying the tier
            # does, and it refills by demand, most used experts first.
            #
            # "Unforced": the driver is short although this request needs no more
            # than the last one, the tier is within its cap, and nothing has evicted
            # PART of the tier since the last fit. That last condition matters: the
            # prefill net can, the holes it leaves are not the tier's own packing,
            # and they are gone after this flush -- learning from them would shrink
            # the tier for the life of the process.
            unforced = (free < need and usage <= want
                        and self._tier_evictions() == self._tier_evictions_seen
                        and self._last_reserve is not None and reserve <= self._last_reserve)
            short, shown = need - free, free
            flushed = usage
            self.tier.clear()
            self.flushes += 1
            self.flushed_bytes += flushed
            free, usage = self._read()
            if unforced:
                # Once can be bytes this process allocated for good at its first
                # requests (about 70 MiB on gpt-oss), which the flush re-measures
                # around. Twice means the filled tier occupies more than its bytes
                # plus the allowance. How much more was just measured -- the driver
                # got back `free - shown` for `flushed` bytes counted -- so hold that
                # share back from now on, and the refilled tier fits.
                self.unforced_flushes += 1
                occupied = max(flushed, free - shown)
                learned = min(_ALLOWANCE_CAP, 1.0 - flushed / occupied + _ALLOWANCE_PAD)
                if self.unforced_flushes > 1 and learned > self.allowance:
                    self.allowance = learned
                    if not self._warned_unforced:
                        self._warned_unforced = True
                        print(f"[Vaayu-VRAM] WARNING: elastic tier flushed "
                              f"{flushed / 1e9:.2f} GB although this request needs no "
                              f"more than the last one: the driver was "
                              f"{short / 1e9:.3f} GB short of the {need / 1e9:.2f} GB it "
                              f"must show. A filled tier occupies "
                              f"{occupied / flushed - 1.0:.1%} more than its bytes on "
                              f"this model's tensor sizes; the allowance is now "
                              f"{self.allowance:.1%}.")
        self._last_reserve = reserve
        self._tier_evictions_seen = self._tier_evictions()
        ok = free >= need
        # A request that already holds its whole predicted reserve has out-run the
        # predictor: whatever the driver shows, none of it is given to the tier.
        surplus = free - need if live < reserve else 0
        # The tier may hold (1 - allowance) of what it and the driver's surplus add up
        # to -- never less than it holds now: with the driver showing the need there
        # is nothing to take back, and the allowance is what keeps it that way.
        budget = (min(want, max(usage, int((usage + surplus) * (1.0 - self.allowance))))
                  if ok else 0)
        self._set_budget(budget)
        if flushed or abs(budget - before) >= self.expert_bytes:
            print(f"[Vaayu-VRAM] elastic tier: granted {int(granted)} of "
                  f"{int(self.limit) or int(granted)} tokens; tier budget "
                  f"{before / 1e9:.2f} -> {budget / 1e9:.2f} GB "
                  f"(~{budget // self.expert_bytes} experts)"
                  + (f"; flushed {flushed / 1e9:.2f} GB" if flushed else ""))
        return ok

    # ── a request ───────────────────────────────────────────────────────────────

    def admit(self, prompt_tokens: int, limit: int) -> int:
        """Admit ONE request whose context may reach `limit` (prompt + new tokens) and
        size the tier for its first grant. Returns the predicted peak at `limit`, as
        _assert_context_fits does; raises VramPlanError when it cannot be covered.
        """
        limit = int(limit)
        peak = predict_peak_vram(self.shape, limit)
        reserve = int(self.plan.reserve_bytes)
        if peak > reserve:
            raise VramPlanError(_too_big_message(
                requested=limit, s_max=self.plan.s_max, free_vram=int(self._driver_free()),
                shape=self.shape, min_resident=self.expert_bytes, headroom=reserve))
        # The whole request, measured: an emptied tier returns at least the bytes it
        # counts, so free + usage is what this request can be given at most.
        whole = self.reserve_for(limit)
        free, usage = self._read()
        if free + usage < whole:
            raise VramPlanError(self._cannot_give(limit, whole, free, usage))
        self.limit = limit
        # The grant covers the first decode forward too (prompt + 1): a re-fit there
        # would read the driver while the prefill's last blocks are still pinned
        # (measured: 2.2 GiB not returnable at that moment behind an 8k prompt).
        self.granted = self._grant_for(max(1, min(limit, int(prompt_tokens) + 1)))
        self.grants += 1
        self._warned_unbacked = False
        # Nothing of this request is on the device yet: what is allocated now, less
        # the tier's entries, is not the request's.
        self._idle_allocated = int(self._allocated()) - usage
        if not self.fit(self.granted, 0):
            # Unreachable while the whole-request check above holds; kept so that a
            # request is never run on a grant the driver did not back.
            raise VramPlanError(self._cannot_give(
                limit, whole, int(self._driver_free()), int(self.tier.current_usage)))
        self.covering = True
        return peak

    def _cannot_give(self, limit: int, whole: int, free: int, usage: int) -> str:
        """The refusal for a request the plan allows and the card cannot cover."""
        return (f"Context of {int(limit)} tokens does not fit: a request that long must be "
                f"able to take {whole / _GIB:.2f} GiB of VRAM (its predicted peak, never less "
                f"than the {self.base_floor / _GIB:.2f} GiB floor), and "
                f"{(free + usage) / _GIB:.2f} GiB is all it can be given: "
                f"{free / _GIB:.2f} GiB is free, and the expert VRAM tier would hand back the "
                f"{usage / _GIB:.2f} GiB it holds. Something else holds memory this load "
                f"reserved (the plan reserves {int(self.plan.reserve_bytes) / _GIB:.2f} GiB "
                f"for a request at max_seq_len={int(self.plan.max_seq_len)}).\n"
                f"Real levers: free VRAM on the device, or shorten the request.\n"
                f"There is no silent fallback: shrinking your context to survive would be "
                f"the degradation CLAUDE.md §5 forbids.")

    def cover(self, context_after: int) -> None:
        """Called before a forward that leaves the context at `context_after` tokens:
        extend the grant when that passes it, and re-fit the tier BEFORE the forward
        allocates. One comparison when the grant already covers it, and nothing at
        all for a request this object did not admit."""
        context_after = int(context_after)
        if not self.covering or context_after <= self.granted:
            return
        granted = self._grant_for(context_after)
        self.granted = granted
        self.extensions += 1
        live = self._live()
        if self.fit(granted, live):
            return
        self.unbacked += 1
        if not self._warned_unbacked:
            self._warned_unbacked = True
            need = max(0, self.reserve_for(granted) - live)
            print(f"[Vaayu-VRAM] WARNING: the driver cannot back a context of {granted} "
                  f"tokens: this request may still take {need / _GIB:.2f} GiB and "
                  f"{int(self._driver_free()) / _GIB:.2f} GiB is free with the expert VRAM "
                  f"tier empty. The tier gets no budget until a later grant finds the "
                  f"memory; generation continues and may run out of VRAM (a real limit, "
                  f"not a bug: something else holds the memory this load reserved).")

    def suspend(self) -> None:
        """Back to the static plan, for a caller that runs the static rules
        (batched_generate): no lend, and nothing resident that a partial eviction of
        its own could strand."""
        if int(self.tier.current_usage) > 0:
            self.flushes += 1
            self.flushed_bytes += int(self.tier.current_usage)
            self.tier.clear()
        self._empty_cache()
        self._set_budget(self.base)
        self.granted = 0
        self.limit = 0
        self.covering = False
        self._last_reserve = None
