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
        return "model." + key
    root = key.split(".", 1)[0]
    if root in _KEPT_ROOTS:
        return "model." + key
    raise ValueError(
        f"DeepSeek-V4 checkpoint key {key!r} has an unrecognised root {root!r}. "
        f"Known roots: embed, layers, norm, hc_head_*, head, mtp (dropped). "
        f"Refusing to guess -- a key dropped here is a weight that stays "
        f"meta-initialized with nothing to say so.")
