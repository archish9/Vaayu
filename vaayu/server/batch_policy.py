"""Pure batch-formation policy for the server's scheduler.

No torch, no model, no I/O -- the decision is the part worth testing
exhaustively, and it must be testable over both MoE geometries without a
checkpoint. Same split vaayu/batch.py already uses against batched_generate.

WHY THIS EXISTS. Batching and the cross-turn prefix cache are in direct
conflict, and nothing on the P10 row had priced it: the +1.606x was measured
with the prefix cache OFF while the server defaults it ON.
`batched_generate` performs ONE lookup against the batch's COMMON prefix and
never looks up each stream's own session entry, so batching two warm agent
sessions discards each one's warm prefix -- the largest measured lever in this
engine (136-377x TTFT) -- to buy the batch win. Per-row prefixes of differing
lengths are an engine change (broadcast_cache_rows gives every row the same
`m`) and are out of v1, so the scheduler decides instead.

HOW IT DECIDES. Prefill at agentic context is expert-weight streaming per
chunk -- a batched chunked prefill re-reads the expert set once per chunk
(vaayu_base.py:5455) -- so both plans are priced in CHUNK PASSES, derived from
the prompts and prefill_chunk_size. No invented constants, no model numbers
(CLAUDE.md section 5).

WHAT IT DOES NOT PRICE. Decode. Batching's decode gain is P6/P8-sized
(+3-18 %) and is not in this comparison, which biases the rule towards
serialising -- the conservative direction, and what the acceptance A/B checks.
"""
from dataclasses import dataclass
from math import ceil

from ..batch import batch_common_prefix


@dataclass
class Candidate:
    """One waiting request, as formation sees it.

    `own_match` is what this prompt would get from the prefix pool on its own
    (PrefixCacheStore.peek_match) -- the reuse batching would spend.
    """
    prompt_ids: list
    own_match: int = 0


def chunk_passes(tokens: int, chunk: int) -> int:
    """Prefill chunk passes for `tokens` new tokens.

    `chunk <= 0` means chunking is disabled (PREFILL_CHUNK_SIZE = 0), which is
    ONE unchunked pass -- not a ZeroDivisionError on the admission path.
    """
    if tokens <= 0:
        return 0
    if chunk <= 0:
        return 1
    return ceil(tokens / chunk)


def serial_passes(cands, chunk: int) -> int:
    """What these run as separate requests: each pays its own tail, and each
    keeps its own warm prefix."""
    return sum(chunk_passes(len(c.prompt_ids) - c.own_match, chunk)
               for c in cands)


def batched_passes(cands, chunk: int) -> int:
    """What these run as ONE batch: the rectangular prefill covers the longest
    prompt from the batch's COMMON prefix, and every per-session prefix is
    forgone."""
    if not cands:
        return 0
    common = len(batch_common_prefix(cands))
    longest = max(len(c.prompt_ids) for c in cands)
    return chunk_passes(longest - common, chunk)


def select_width(cands, *, max_batch_size: int, chunk: int) -> int:
    """How many of `cands`, IN ARRIVAL ORDER, to run as one batch. Always >= 1.

    Greedy and prefix-ordered: a candidate joins only while the batch still
    beats running the group serially, and the first candidate that fails ENDS
    formation rather than being skipped -- it keeps its queue position, so
    arrival order is preserved and nothing starves. The head is always a
    member of whatever is returned, so head-of-line always progresses.

    `max_batch_size` is a CEILING, never a target: max width is model- and
    context-dependent and is not 3 (MiniMax admits 2 here; Qwen admits 3 forced
    but is faster at 2).
    """
    limit = min(len(cands), max(1, max_batch_size))
    width = 1
    for k in range(2, limit + 1):
        if cands[k - 1].own_match > 0:
            break
        if batched_passes(cands[:k], chunk) < serial_passes(cands[:k], chunk):
            width = k
        else:
            break
    return width
