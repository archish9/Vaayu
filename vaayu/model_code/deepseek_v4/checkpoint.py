"""What A5.7b's load path reads off this checkpoint's files, before any weight
is loaded: the key names, and the fp4 group size.

Both answer questions the config cannot. The keys are DeepSeek's own
(`layers.3.ffn.experts.7.w1.weight`), not HF's; the group size is in no config
field and is knowable only from a weight/scale shape pair.
"""

_KEPT_ROOTS = ("layers", "norm")


def remap_checkpoint_key(key: str) -> "str | None":
    """This checkpoint's key -> the module path A5.6's tree exposes, or None to drop.

    Applied by `split_and_save_layers(key_remap=...)` BEFORE the split, so the
    split output is named like every other model's and no consumer downstream
    learns a second convention.

    `mtp.*` returns None: the MTP draft module is excluded (user, 2026-09-23).
    It is not only 10.117 GiB of unused weights -- `detect_moe_blocks` takes the
    layer index from the first `.N.` in a path, so `mtp.0.ffn` would be detected
    as layer 0 and collide with the real one.

    Anything else RAISES. A key silently dropped is a weight silently missing,
    and this architecture's own `from_config` would then build a skeleton whose
    meta tensors nothing ever fills.
    """
    if key.startswith("mtp."):
        return None
    if key == "embed.weight":
        return "model.embed_tokens.weight"
    if key == "head.weight":
        # `lm_head` sits outside `model.` by HF convention, and `vaayu/utils.py`
        # builds its layer list with a bare `lm_head.`.
        return "lm_head.weight"
    if key.startswith("hc_head_"):
        # NOT "model." + key. `vaayu/utils.py` assigns a checkpoint key to a
        # shard by SUBSTRING match against ['model.embed_tokens.', ...,
        # 'model.norm.', 'lm_head.'] -- and 'model.hc_head_base' contains none
        # of those as a substring, so it would land in no shard and be
        # silently dropped (Task 4b; caught by utils.py's G2 guard, which
        # raises on exactly this because key_remap is not None here). Routing
        # it into the `model.norm.` shard is also correct on the merits: that
        # shard loads exactly when `run_norm` -- the only caller of `hc_head`
        # -- runs. `VaayuDeepseekV4.init_model` undoes the rename back to the
        # real module path (`model.hc_head_*`) via `_path_remappings` after
        # the shard is read.
        return "model.norm." + key
    root = key.split(".", 1)[0]
    if root in _KEPT_ROOTS:
        return "model." + key
    raise ValueError(
        f"DeepSeek-V4 checkpoint key {key!r} has an unrecognised root {root!r}. "
        f"Known roots: embed, layers, norm, hc_head_*, head, mtp (dropped). "
        f"Refusing to guess -- a key dropped here is a weight that stays "
        f"meta-initialized with nothing to say so.")


def fp4_group_size_from_split(split_dir, layer_prefix: str = "model.layers") -> int:
    """`in_features // scale.shape[-1]`, undoing fp4's 2-per-byte packing.

    Read off the SPLIT shards, not the originals: `delete_original_shards` keeps
    `model.safetensors.index.json` and deletes the shards it names, so a deriver
    that reads the original checkpoint works exactly once.

    Headers only -- `get_slice(...).get_shape()` reads no tensor data.

    The group size is in no config field. A5.4's skeleton refuses to invent one
    and raises a named ValueError, so this has to run before `from_config`.
    """
    from pathlib import Path

    from safetensors import safe_open

    shard = Path(split_dir) / f"{layer_prefix}.0.safetensors"
    if not shard.exists():
        raise FileNotFoundError(
            f"cannot derive fp4_group_size: no {layer_prefix}.0 shard under "
            f"{split_dir}. The split has not been written yet.")

    with safe_open(str(shard), framework="pt") as handle:
        keys = set(handle.keys())
        weight_key = next(
            (k for k in sorted(keys)
             if k.endswith("w1.weight") and ".experts." in k), None)
        scale_key = weight_key.replace(".weight", ".scale") if weight_key else None
        if weight_key is None or scale_key not in keys:
            raise ValueError(
                f"cannot derive fp4_group_size from {shard}: no routed-expert "
                f"weight/scale pair (looked for '*.experts.*.w1.weight' and its "
                f"'.scale'). The group size is in no config field and must not "
                f"be invented.")
        in_features = handle.get_slice(weight_key).get_shape()[-1] * 2
        groups = handle.get_slice(scale_key).get_shape()[-1]

    return in_features // groups
