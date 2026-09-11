"""S1d: decide speculation on a MoE model from its OWN routing and acceptance.

The Fix #9 gate sizes a speculative verify pass at
min(num_experts, positions * top_k) -- a zero-overlap upper bound. Two things
measured on 2026-09-05 say a static gate cannot do this job at all:

  * the bound is 24-51% (Qwen) / 14-35% (MiniMax) too pessimistic and refuses
    MiniMax's draft_tokens=3, which measured 1.29x; but
  * replacing it with the measured union ADMITS Qwen's draft_tokens=1, which
    fits the budget (13.53 GB against 15.8) and measured 0.836x -- 16% slower.

So bytes-fit is necessary and not sufficient. What separates the two is
ACCEPTANCE, a property of what the model is emitting and unknowable at load
time. Hence a runtime gate. Nothing here imports torch or touches a model:
the state machine is a pure object the generation loop feeds.
"""
from __future__ import annotations

from statistics import fmean, median


def observed_union_per_layer(snapshots: list[dict[int, list[int]]], k: int) -> float:
    """Mean unique-expert union over every window of `k` consecutive snapshots.

    `snapshots[i][layer_idx]` is the expert list that layer routed at decode
    step i -- exactly the shape `vaayu_ref._expert_history` already holds, and
    which all three MoE forwards write (moe_patch.py:1281, :1419, :2028).

    Windows SLIDE rather than tile: a short calibration window is the whole
    evidence base, and tiling would discard half of it. Returns 0.0 when there
    are not enough snapshots for one window -- a caller that treats 0.0 as
    "does not fit" refuses, which is the safe direction.
    """
    if k < 1 or len(snapshots) < k:
        return 0.0
    layers = sorted({layer for snap in snapshots for layer in snap})
    if not layers:
        return 0.0
    window_means = []
    for start in range(len(snapshots) - k + 1):
        window = snapshots[start:start + k]
        sizes = []
        for layer in layers:
            seen: set[int] = set()
            for snap in window:
                seen.update(snap.get(layer, ()))
            sizes.append(len(seen))
        window_means.append(sum(sizes) / len(sizes))
    return sum(window_means) / len(window_means)


def union_bytes(union_per_layer: float, geometry: dict) -> int:
    """Expert bytes one verify pass reads, given a per-layer union.

    Every term comes from the geometry (CLAUDE.md 5) -- this function holds no
    model number of its own.
    """
    return int(union_per_layer
               * geometry["num_moe_layers"]
               * geometry["expert_bytes"])


def spec_breakeven_acceptance(union_bytes_1: int, union_bytes_k: int,
                              draft_tokens: int) -> float:
    """Accepted-per-drafted-token above which a verify pass beats plain decode.

    A round reads `union_bytes_k` and yields `1 + accepted` tokens; plain
    decode reads `union_bytes_1` per token. Equal cost per token when
    `union_k / (1 + a*d) == union_1`, hence the expression below.

    Returns inf when the answer is unsatisfiable (no drafting, or an unusable
    single-position figure) so the caller refuses rather than divides by zero.
    """
    if draft_tokens < 1 or union_bytes_1 <= 0:
        return float("inf")
    return max(0.0, (union_bytes_k / union_bytes_1 - 1.0) / draft_tokens)


def wall_breakeven_acceptance(alpha: float, draft_tokens: int) -> float:
    """Accepted-per-drafted-token above which a verify pass beats plain decode
    ON THE CLOCK, given a measured per-position cost factor `alpha`.

    Row 13, measured 2026-09-10: a (d+1)-position verify costs about
    (d+1)*alpha plain-decode rounds -- near-LINEAR in POSITIONS, and nothing
    like the expert-byte ratio spec_breakeven_acceptance models. A round then
    costs (1+d)*alpha base-units and yields 1 + a*d tokens, so it breaks even
    at ((1+d)*alpha - 1)/d.

    Same contract as spec_breakeven_acceptance: inf when unsatisfiable, and
    never negative -- a verify pass cheaper than one decode round needs no
    acceptance at all, it does not earn a discount.
    """
    if draft_tokens < 1:
        return float("inf")
    return max(0.0, ((1.0 + draft_tokens) * alpha - 1.0) / draft_tokens)


# Phase-length policy. These are properties of the STATISTICS, not of any
# model (CLAUDE.md 5 -- no model numbers in code), and are module constants
# rather than config knobs because nobody asked for the knobs (CLAUDE.md 2).
#
# CALIBRATE_STEPS: a FLOOR, not the calibration length itself. It exists so
# there are enough windows to average over -- 8 gives 5 windows at
# draft_tokens=3 and 7 at draft_tokens=1 -- and costs nothing, since these
# steps decode normally. The actual length a gate waits for is
# max(CALIBRATE_STEPS, draft_tokens + 1) (computed per-instance in __init__,
# see _calibrate_steps): a window of d+1 snapshots is the thing
# observed_union_per_layer measures at all, so at draft_tokens >= 8 the floor
# alone is not enough snapshots for even ONE window, and _decide_union would
# see union_k == 0 and latch off_union on every load regardless of budget --
# a silent, permanent, budget-independent refusal (CLAUDE.md 5: never
# silently degrade).
CALIBRATE_STEPS = 8
# TRIAL_DRAFTED_TOKENS: at 48 drafts the binomial standard error is ~0.07, so
# Qwen's measured 0.286 sits ~3 SE below its 0.520 bar -- decisive without
# over-sampling.
TRIAL_DRAFTED_TOKENS = 48
# TRIAL_LEG_ROUNDS / TRIAL_PAIRS (row 13, `even-4`): TRIAL alternates legs of
# TRIAL_LEG_ROUNDS drafting rounds and TRIAL_LEG_ROUNDS suppressed ones, and
# the verdict pools at least TRIAL_PAIRS such pairs. One narrow window, however
# well measured, sampled the cheapest stretch of a call and admitted an arm
# that lost -19.7% on its own A/B (2026-09-10). Replayed on the six archived
# MiniMax-M2.7 traces with CALIBRATE charged: 4 pairs of 4+4 decide 6/6 at
# every leg geometry tried; 3 pairs drop to 5/6, 1 pair never beats 4/6.
# 4 x 4 drafting rounds x 3 drafted = 48 = TRIAL_DRAFTED_TOKENS at the
# measured draft_tokens=3, so the acceptance estimate is not loosened; at other
# widths the trial ends at whichever of the two floors is reached LAST.
TRIAL_LEG_ROUNDS = 4
TRIAL_PAIRS = 4
# TRIAL_ROUND_CAP: bounds the case where the proposer rarely matches at all,
# so a trial cannot run forever waiting for drafts that never come. Doubled
# from 64 when TRIAL began alternating, so a proposer still gets the same 64
# drafting-eligible rounds; a multiple of 2 * TRIAL_LEG_ROUNDS, so the cap
# always lands on a pair boundary.
TRIAL_ROUND_CAP = 128


class AdaptiveSpecGate:
    """Decides, at runtime, whether speculation pays on THIS MoE model.

    Three phases. CALIBRATE runs the loop without drafting and watches real
    routing; if the observed union of a verify pass does not fit the expert
    cache budget, speculation is refused before a single draft is paid for.
    TRIAL alternates drafting and suppressed legs -- an A/B inside the call --
    and admits only if acceptance clears a break-even derived from those same
    observed unions AND the drafting legs were cheaper per token than their
    suppressed partners, on average over the pairs. RUN speculates.

    The decision latches: once off, it stays off for the process. Pure -- no
    torch, no model -- so the whole machine is testable without a load.
    """

    def __init__(self, geometry: dict, draft_tokens: int, budget_bytes: int):
        self.geometry = geometry
        self.draft_tokens = int(draft_tokens)
        self.budget_bytes = int(budget_bytes)
        # The calibration length THIS gate needs: never less than the floor,
        # and never less than one full union window (draft_tokens + 1) --
        # see CALIBRATE_STEPS's comment above.
        self._calibrate_steps = max(CALIBRATE_STEPS, self.draft_tokens + 1)
        self.phase = "calibrate"
        self.decision: str | None = None
        self._snapshots: list[dict[int, list[int]]] = []
        self._union_1 = 0
        self._union_k = 0
        self._breakeven = float("inf")
        self._drafted = 0
        self._accepted = 0
        self._drafting_rounds = 0
        self._rounds = 0
        # Row 13: the wall-clock arm. CALIBRATE's rounds do not draft and
        # TRIAL's do, so the gate already runs both arms of the A/B -- these
        # two lists are the only thing it was missing, and they cost the loop
        # two timestamps per round. The gate never reads a clock itself: it
        # stays pure and deterministic, which is what makes the whole state
        # machine testable without a model.
        self._base_times: list[float] = []
        self._null_times: list[float] = []
        self._wide_times: list[float] = []
        self._alpha: float | None = None
        self._wall_breakeven: float | None = None
        self._bar_used: str | None = None
        self._base_from: str | None = None
        # Row 13, `even-4`: per-pair [draft_s, draft_tokens, supp_s,
        # supp_tokens], timed rounds only.
        self._pairs: list[list[float]] = []
        self._pooled_ratio: float | None = None
        self._pairs_used = 0

    # -- phase questions the loop asks ------------------------------------
    def should_draft(self) -> bool:
        if self.phase == "trial":
            return self._in_drafting_leg()
        return self.phase == "run"

    def _in_drafting_leg(self) -> bool:
        """Whether the NEXT trial round falls in a drafting leg. Each pair
        drafts first, so CALIBRATE's suppressed rounds are never followed by
        a further suppressed leg before the first draft."""
        return self._rounds % (2 * TRIAL_LEG_ROUNDS) < TRIAL_LEG_ROUNDS

    def decided_off(self) -> bool:
        return self.phase == "off"

    # -- evidence the loop feeds ------------------------------------------
    def observe(self, history: dict[int, list[int]],
                elapsed_s: float | None = None) -> None:
        """One decode step's per-layer routing (a copy of _expert_history).

        `elapsed_s` is that round's wall clock. A calibrate round does not
        draft, so it is the 1-wide reference the timed bar divides by. It is
        optional: omit it and the gate decides on bytes alone, exactly as it
        did before row 13.
        """
        if self.phase != "calibrate":
            return
        if elapsed_s is not None:
            self._base_times.append(float(elapsed_s))
        self._snapshots.append({int(k): list(v) for k, v in history.items()})
        if len(self._snapshots) >= self._calibrate_steps:
            self._decide_union()

    def record_round(self, drafted: int, accepted: int,
                     elapsed_s: float | None = None,
                     draft_overhead_s: float | None = None) -> None:
        """One speculative round's outcome. A round that drafted nothing is a
        1-wide decode producing one token -- economically identical to plain
        decode -- so it counts toward the round cap but NOT toward acceptance.

        `elapsed_s` is that round's wall clock. A drafting round is the
        (d+1)-wide pass the bar is about; a k=0 round is an ordinary 1-wide
        decode, so it is kept SEPARATELY as the contemporaneous baseline --
        never in the wide median, which would understate alpha and bias toward
        ADMIT.

        `draft_overhead_s` is the proposer's OWN time in that round. It is
        subtracted from a k=0 round's baseline and from nothing else: the
        counterfactual a timed bar compares against is speculation OFF, where
        the proposer never runs, so the baseline is that round's DECODE, not
        its decode plus a search that found nothing. On a drafting round the
        search is a real cost of speculating and stays in the numerator.

        Row 13, `even-4`: the round is also booked to its leg of the current
        pair -- seconds and tokens, a drafting round yielding `accepted + 1`
        and a suppressed one 1. No draft-overhead correction there: a leg's
        s/token is what the call actually paid in that arm.
        """
        if self.phase != "trial":
            return
        cycle = 2 * TRIAL_LEG_ROUNDS
        drafting_leg = self._in_drafting_leg()
        pair = self._rounds // cycle
        self._rounds += 1
        if elapsed_s is not None:
            while len(self._pairs) <= pair:
                self._pairs.append([0.0, 0, 0.0, 0])
            leg = 0 if drafting_leg else 2
            self._pairs[pair][leg] += float(elapsed_s)
            self._pairs[pair][leg + 1] += (accepted + 1) if drafted > 0 else 1
        if drafted > 0:
            self._drafting_rounds += 1
            self._drafted += drafted
            self._accepted += accepted
            if elapsed_s is not None:
                self._wide_times.append(float(elapsed_s))
        elif elapsed_s is not None:
            _decode_s = float(elapsed_s) - float(draft_overhead_s or 0.0)
            if _decode_s > 0:
                self._null_times.append(_decode_s)
        # Decide only on a pair boundary, so every leg keeps its partner.
        if (self._rounds >= TRIAL_ROUND_CAP
                or (self._rounds % cycle == 0
                    and self._rounds // cycle >= TRIAL_PAIRS
                    and self._drafted >= TRIAL_DRAFTED_TOKENS)):
            self._decide_acceptance()

    # -- the two decisions -------------------------------------------------
    def _decide_union(self) -> None:
        self._union_1 = union_bytes(
            observed_union_per_layer(self._snapshots, 1), self.geometry)
        self._union_k = union_bytes(
            observed_union_per_layer(self._snapshots, self.draft_tokens + 1),
            self.geometry)
        self._breakeven = spec_breakeven_acceptance(
            self._union_1, self._union_k, self.draft_tokens)
        if self._union_k <= 0:
            # calibrate_steps is always >= draft_tokens + 1 (see __init__), so
            # there are always enough snapshots for a full k-window; the only
            # way observed_union_per_layer still returns 0.0 here is that
            # every snapshot's routing dict was empty -- a forward that has
            # stopped writing _expert_history (design doc S8's named risk),
            # not a real over-budget union. Report it as its own decision so
            # it is never read as "measured a real union of zero and it
            # exceeded budget_bytes" -- that reading is impossible, since a
            # genuine over-budget refusal requires union_k > budget_bytes > 0.
            self.phase, self.decision = "off", "off_no_routing"
        elif self._union_k > self.budget_bytes:
            self.phase, self.decision = "off", "off_union"
        else:
            self.phase = "trial"

    def _decide_acceptance(self) -> None:
        """`run` iff acceptance clears the byte bar AND the mean per-pair
        s/token ratio is below 1 (row 13, `even-4`).

        The ratio decides DIRECTLY, not through alpha and the wall bar:
        replayed on the same six traces, pooled-alpha -> wall bar scored 4/6,
        worse than the shipped 5/6, while bytes AND the pooled ratio scored
        6/6 at every leg geometry. alpha is a per-position cost from 4-sample
        medians with acceptance re-applied afterwards; the s/token ratio
        already contains realised acceptance, which is the quantity at issue.
        MEAN, not median: a median over 4 pairs flips when one pair moves.

        With fewer than TRIAL_PAIRS timed pairs there is nothing to pool, and
        the shipped single-window rule decides; `bar_used` says which.
        alpha and the wall bar are still computed either way, as telemetry.
        """
        bar = self._acceptance_bar()
        measured = self.measured_acceptance
        ratios = self._pair_ratios()
        self._pairs_used = len(ratios)
        if len(ratios) >= TRIAL_PAIRS:
            self._pooled_ratio = fmean(ratios)
            self._bar_used = "pooled"
            admit = (measured is not None and measured > self._breakeven
                     and self._pooled_ratio < 1.0)
        else:
            admit = measured is not None and measured > bar
        if admit:
            self.phase, self.decision = "run", "run"
        else:
            self.phase, self.decision = "off", "off_acceptance"

    def _pair_ratios(self) -> list[float]:
        """(drafting-leg s/token) / (suppressed-leg s/token) for every pair
        with both legs timed. A pair missing either leg is not an A/B."""
        return [(ds / dt) / (ss / st) for ds, dt, ss, st in self._pairs
                if dt > 0 and st > 0 and ss > 0]

    def _acceptance_bar(self) -> float:
        """The bar acceptance must clear: bytes AND clock, whichever binds.

        The byte break-even stays a NECESSARY condition -- it is free, it is
        sound (a verify pass that does not fit thrashes), and it is what
        refuses Qwen3-235B at zero cost. The timed bar is the sufficient one:
        measured 2026-09-07, acceptance 0.6875 cleared a byte bar of 0.6259
        and still cost -16% of decode, because the byte model prices a verify
        pass in expert bytes and time is near-linear in POSITIONS.

        Falls back to bytes alone when the timing sample is too small or
        unusable, and says which rule decided in status() -- a gate that
        silently changes that is worse than either rule.
        """
        self._alpha = self._measured_alpha()
        if self._alpha is None:
            self._wall_breakeven = None
            self._bar_used = "bytes"
            return self._breakeven

        self._wall_breakeven = wall_breakeven_acceptance(
            self._alpha, self.draft_tokens)
        self._bar_used = ("wall" if self._wall_breakeven > self._breakeven
                          else "bytes")
        return max(self._breakeven, self._wall_breakeven)

    def _measured_alpha(self) -> float | None:
        """Per-position cost of a verify pass, in plain-decode round units.

        MEDIANS, never means: the spike saw single-round outliers at 1.35x the
        median, and one NVMe stall must not move a latched verdict. Every list
        needs at least `draft_tokens + 1` samples -- the same window width
        observed_union_per_layer is built around, and the floor
        _calibrate_steps already guarantees -- or this is not a measurement.

        WHICH BASELINE, and why it is not CALIBRATE's by default: CALIBRATE's
        rounds are the FIRST rounds of a call, immediately after the prefill,
        so they run against a cold expert cache while TRIAL's run warm.
        Measured on device 2026-09-10, MiniMax-M2.7's losing arm: that
        mismatch alone moved alpha 0.7801 -> 0.6976 and put the wall bar
        (0.5969) UNDER the byte bar (0.6305), so the timed test never bound
        and the configuration was admitted exactly as before. TRIAL's own k=0
        rounds are 1-wide decodes sampled in TRIAL's regime -- the
        contemporaneous baseline the ratio wants. They carry the n-gram
        search's own small cost, which biases toward ADMIT, but measurably
        less than the cold-start mismatch does. CALIBRATE remains the fallback
        for a workload that matches every round and so suppresses none.
        """
        floor = self.draft_tokens + 1
        if len(self._wide_times) < floor:
            self._base_from = None
            return None
        if len(self._null_times) >= floor:
            samples, self._base_from = self._null_times, "trial_suppressed"
        elif len(self._base_times) >= floor:
            samples, self._base_from = self._base_times, "calibrate"
        else:
            self._base_from = None
            return None
        base = median(samples)
        if base <= 0:
            self._base_from = None
            return None
        return median(self._wide_times) / (base * (1 + self.draft_tokens))

    @property
    def measured_acceptance(self) -> float | None:
        return (self._accepted / self._drafted) if self._drafted else None

    # -- what env.json and /v1/models show --------------------------------
    def status(self) -> dict:
        return {
            "observed_union_gb": round(self._union_k / 1e9, 2),
            "budget_gb": round(self.budget_bytes / 1e9, 2),
            "breakeven_acceptance": (None if self._breakeven == float("inf")
                                     else round(self._breakeven, 4)),
            "measured_acceptance": (None if self.measured_acceptance is None
                                    else round(self.measured_acceptance, 4)),
            "drafting_rounds": self._drafting_rounds,
            "alpha": (None if self._alpha is None else round(self._alpha, 4)),
            "wall_breakeven": (None if self._wall_breakeven is None
                               else round(self._wall_breakeven, 4)),
            "bar_used": self._bar_used,
            "base_from": self._base_from,
            "pooled_ratio": (None if self._pooled_ratio is None
                             else round(self._pooled_ratio, 4)),
            "pairs_used": self._pairs_used,
            "decision": self.decision,
        }
