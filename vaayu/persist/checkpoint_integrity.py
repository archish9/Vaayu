"""Refuse a checkpoint that is not all there, before anything reads it.

Row 30's RC5: the splitter dropped tensors whenever a layer spanned
non-adjacent shards and stamped its `.done` marker anyway, so the damage was
invisible until inference produced wrong output. An incomplete download is the
same failure with an earlier cause -- and it happens: the DeepSeek-V4-Flash
checkpoint on this box was 46-of-48 shards with no index for part of
2026-09-20.

Model-agnostic by construction: it reads the shard NAMES and the index's own
`weight_map`, never a config field or an architecture name.
"""
import json
import re
from pathlib import Path

_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.safetensors$")


def assert_checkpoint_complete(checkpoint_dir: str) -> None:
    """Raise RuntimeError if `checkpoint_dir`'s shard set is incomplete.

    Silent on directories that are not sharded checkpoints -- deciding what
    counts as a checkpoint is not this function's job.
    """
    path = Path(checkpoint_dir)
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        return

    totals = {int(m.group(2)) for m in
              (_SHARD_RE.search(s.name) for s in shards) if m}
    if totals:
        total = max(totals)
        present = {int(m.group(1)) for m in
                   (_SHARD_RE.search(s.name) for s in shards) if m}
        missing = sorted(set(range(1, total + 1)) - present)
        if missing:
            names = [f"{i:05d}-of-{total:05d}" for i in missing[:4]]
            raise RuntimeError(
                f"{path}: incomplete checkpoint -- {len(present)} of {total} shards "
                f"present, missing {names}{' ...' if len(missing) > 4 else ''}. "
                f"Loading it would stream only part of the model. If it is still "
                f"downloading, wait for it to finish."
            )

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        referenced = set(weight_map.values())
        absent = sorted(f for f in referenced if not (path / f).exists())
        if absent:
            raise RuntimeError(
                f"{path}: the safetensors index references {len(absent)} file(s) that "
                f"are not present: {absent[:4]}. The checkpoint is inconsistent."
            )


def assert_expert_layout_indexable(checkpoint_dir: str, tensor_keys) -> None:
    """Raise when the config implies routed experts but no key matches the
    per-expert layout the offset index understands.

    The equivalent check in `ExpertOffsetIndex.build()` runs only after the
    split has been written and -- under the server's DELETE_ORIGINAL default --
    after the original shards are gone, so a checkpoint Vaayu cannot index would
    cost the user their only copy before anything said so. Same question, asked
    while they still have the checkpoint.

    Layout-agnostic in the same way the later guard is: it asks "config says E,
    no key carries experts in a layout the index knows", never anything about a
    named layout.

    It accepts BOTH layouts the index reads: per-expert (`.experts.<N>.`) and
    fused (a tensor under an `experts.` prefix carrying every expert on a
    leading axis). This seam sees KEYS only, never shapes, so the fused answer
    here is the weaker one -- it cannot confirm the leading axis really is the
    expert axis. That is the right trade at this seam: its job is to refuse a
    layout plainly outside both before the split deletes the original, and the
    exact check still runs in `ExpertOffsetIndex.build()`, which has the shapes.
    """
    from .expert_offset_index import ExpertOffsetIndex, _EXPERT_RE, _FUSED_EXPERTS_RE

    expected = ExpertOffsetIndex(checkpoint_dir).expected_expert_count()
    if expected is None:
        return

    keys = list(tensor_keys)
    if any(_EXPERT_RE.search(k) or _FUSED_EXPERTS_RE.search(k) for k in keys):
        return

    seen = [k for k in keys if "expert" in k.lower()]
    raise RuntimeError(
        f"{checkpoint_dir}: config implies {expected} routed experts, but no key "
        f"in the checkpoint index matches a layout the expert offset index reads "
        f"(per-expert '.experts.<N>.', or one fused tensor under an 'experts.' "
        f"prefix). Expert-ish keys actually present: "
        f"{seen[:4] if seen else '(none)'}. The checkpoint was NOT split or deleted."
    )
