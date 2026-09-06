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
# TRIAL_ROUND_CAP: bounds the case where the proposer rarely matches at all,
# so a trial cannot run forever waiting for drafts that never come.
TRIAL_ROUND_CAP = 64


class AdaptiveSpecGate:
    """Decides, at runtime, whether speculation pays on THIS MoE model.

    Three phases. CALIBRATE runs the loop without drafting and watches real
    routing; if the observed union of a verify pass does not fit the expert
    cache budget, speculation is refused before a single draft is paid for.
    TRIAL speculates and measures acceptance against a break-even derived from
    those same observed unions. RUN speculates.

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

    # -- phase questions the loop asks ------------------------------------
    def should_draft(self) -> bool:
        return self.phase in ("trial", "run")

    def decided_off(self) -> bool:
        return self.phase == "off"

    # -- evidence the loop feeds ------------------------------------------
    def observe(self, history: dict[int, list[int]]) -> None:
        """One decode step's per-layer routing (a copy of _expert_history)."""
        if self.phase != "calibrate":
            return
        self._snapshots.append({int(k): list(v) for k, v in history.items()})
        if len(self._snapshots) >= self._calibrate_steps:
            self._decide_union()

    def record_round(self, drafted: int, accepted: int) -> None:
        """One speculative round's outcome. A round that drafted nothing is a
        1-wide decode producing one token -- economically identical to plain
        decode -- so it counts toward the round cap but NOT toward acceptance.
        """
        if self.phase != "trial":
            return
        self._rounds += 1
        if drafted > 0:
            self._drafting_rounds += 1
            self._drafted += drafted
            self._accepted += accepted
        if self._drafted >= TRIAL_DRAFTED_TOKENS or self._rounds >= TRIAL_ROUND_CAP:
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
        measured = self.measured_acceptance
        if measured is None or measured <= self._breakeven:
            self.phase, self.decision = "off", "off_acceptance"
        else:
            self.phase, self.decision = "run", "run"

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
            "decision": self.decision,
        }
