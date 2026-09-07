"""A tied-weight checkpoint has no `lm_head` tensor, and the loader must notice.

Bug D1. When `config.tie_word_embeddings` is true the checkpoint stores only
`model.embed_tokens.weight` (measured on Llama-3.2-1B-Instruct: 146 keys, no
`lm_head.weight`). Vaayu's splitter still emits `lm_head.safetensors` — a VALID
but EMPTY 16-byte safetensors file containing `{}` — plus its `.done` marker, so
the split verifies as complete and `load_layer()` SUCCEEDS returning zero
tensors.

`load_layer_to_cpu` had a tied-weights fallback, but it lived in an `except`
handler whose comment read "lm_head.safetensors might not be generated": it
anticipated a MISSING file and so never fired for an empty one. `lm_head` was
left meta-initialized and the model emitted uniform-random logits — measured
perplexity 128365 against a vocab-uniform baseline of 128256, i.e. zero signal.

The decision is a pure function of (which layer, did the load raise, what came
back), so it is tested here without a model load — see CLAUDE.md's method note
about load-time decisions verified by hand-run harnesses that copy the logic
instead of executing it.
"""
import pytest

from vaayu.vaayu_base import should_use_tied_lm_head, rekey_embed_as_lm_head


LM_HEAD = "lm_head"


def test_empty_lm_head_shard_triggers_fallback():
    """The D1 case: file loads fine, contains nothing."""
    assert should_use_tied_lm_head(LM_HEAD, LM_HEAD, {}, load_failed=False) is True


def test_missing_lm_head_shard_triggers_fallback():
    """The case the original except-handler covered. Must keep working."""
    assert should_use_tied_lm_head(LM_HEAD, LM_HEAD, None, load_failed=True) is True


def test_populated_lm_head_is_left_alone():
    """An untied checkpoint (e.g. Llama-3.3-70B) ships real lm_head weights."""
    sd = {"lm_head.weight": object()}
    assert should_use_tied_lm_head(LM_HEAD, LM_HEAD, sd, load_failed=False) is False


def test_other_layers_never_fall_back():
    """Only lm_head has an embedding to tie to. An empty decoder layer is a
    real error and must propagate, not get silently filled with embeddings."""
    assert should_use_tied_lm_head("model.layers.7", LM_HEAD, {},
                                   load_failed=False) is False
    assert should_use_tied_lm_head("model.layers.7", LM_HEAD, None,
                                   load_failed=True) is False


def test_trailing_dot_forms_match():
    """Layer names reach this code both with and without a trailing dot
    depending on the call site (the original handler compared against both)."""
    assert should_use_tied_lm_head("lm_head.", LM_HEAD, {}, load_failed=False) is True
    assert should_use_tied_lm_head(LM_HEAD, "lm_head.", {}, load_failed=False) is True


def test_rekey_maps_embed_tensors_onto_the_head():
    w = object()
    out = rekey_embed_as_lm_head({"model.embed_tokens.weight": w},
                                 "model.embed_tokens", LM_HEAD)
    assert out == {"lm_head.weight": w}


def test_rekey_of_nothing_is_nothing():
    """Feeds the caller's 'still empty -> fail honestly' check (CLAUDE.md §5)."""
    assert rekey_embed_as_lm_head({}, "model.embed_tokens", LM_HEAD) == {}


def test_rekey_uses_a_non_llama_naming_scheme():
    """§5: never assume Llama key names. ChatGLM maps lm_head ->
    transformer.output_layer (see vaayu_chatglm.py)."""
    w = object()
    out = rekey_embed_as_lm_head({"transformer.embedding.word_embeddings.weight": w},
                                 "transformer.embedding.word_embeddings",
                                 "transformer.output_layer")
    assert out == {"transformer.output_layer.weight": w}


def test_no_lm_head_in_module_tree_never_falls_back():
    """detect_layer_names leaves 'lm_head' None when no nn.Linear head is found
    (vaayu_base.py:982). Model-agnostic per CLAUDE.md §5: must not crash."""
    assert should_use_tied_lm_head("model.layers.0", None, {},
                                   load_failed=False) is False
    assert should_use_tied_lm_head(None, None, {}, load_failed=True) is False


# --- The THIRD path, found 2026-09-07 (row 16) --------------------------------
#
# D1 was fixed in two places: `_pin_shared_layers` and `load_layer_to_cpu`.
# `_load_from_strategy` -- the PREFETCH route, which is what the forward uses
# for any non-pinned layer while prefetching is on -- calls
# `loader_strategy.load_from_disk` directly and had no fallback at all.
#
# It only fires when the head is NOT pinned, which is why it hid for so long:
# the head is normally pinned first (it is in `_pin_shared_layers`' phase-1
# list). Measured trigger on Llama-3.2-1B: `attn_implementation="sdpa"` charges
# a score-buffer term flash does not (SCORE_ELEM_BYTES), which inflates
# `reserve_bytes` until `pinning_budget` reaches EXACTLY 0 -- so nothing is
# pinned, the head streams through here, and comes back `{}`.
#
# Observed end to end: lm_head.weight left on the meta device, logits
# std 0.02 against a correct 2.21, output ' القرنuthor matching overall...'
# against 'Red, blue, and yellow.'. Same checkpoint under flash_attention_2
# (pinning_budget 10.97 GB) is correct, and so is stock transformers.
class _FakeStrategy:
    """load_from_disk that mimics the splitter: `{}` for the tied head."""

    def __init__(self, embed_name):
        self._embed = embed_name
        self.cache_hits = {}
        self.calls = []

    def load_from_disk(self, layer_name, checkpoint_path):
        self.calls.append(layer_name)
        if layer_name == self._embed:
            return {f"{self._embed}.weight": "EMBED_TENSOR"}
        return {}


class _FakeModel:
    """Only the surface _load_from_strategy touches."""

    def __init__(self, names):
        self.layer_names_dict = names
        self.loader_strategy = _FakeStrategy(names["embed"])
        self.checkpoint_path = "/nonexistent"
        self._layer_from_cache = {}


def _names(embed="model.embed_tokens", head="lm_head"):
    return {"embed": embed, "lm_head": head}


def test_prefetch_path_falls_back_to_the_embedding_for_a_tied_head():
    """The bug. An empty head shard through the prefetch route must come back
    carrying the embedding's tensors, re-keyed onto the head -- exactly what
    the pinning path already does."""
    from vaayu.vaayu_base import VaayuBaseModel
    m = _FakeModel(_names())
    out = VaayuBaseModel._load_from_strategy(m, "lm_head")
    assert out == {"lm_head.weight": "EMBED_TENSOR"}, (
        "a tied head loaded through the prefetch path came back empty, which "
        "leaves lm_head.weight on the meta device and emits garbage logits")


def test_prefetch_path_leaves_a_populated_head_alone():
    """An UNTIED checkpoint ships real head weights; they must not be replaced
    by the embedding, which would silently swap a trained head for a tied one."""
    from vaayu.vaayu_base import VaayuBaseModel
    m = _FakeModel(_names())
    m.loader_strategy.load_from_disk = (
        lambda layer_name, checkpoint_path: {"lm_head.weight": "REAL_HEAD"})
    assert VaayuBaseModel._load_from_strategy(m, "lm_head") == {"lm_head.weight": "REAL_HEAD"}


def test_prefetch_path_does_not_paper_over_an_empty_decoder_layer():
    """Only the head may be tied. An empty decoder shard is a real failure and
    must stay empty rather than be handed embedding weights."""
    from vaayu.vaayu_base import VaayuBaseModel
    m = _FakeModel(_names())
    assert VaayuBaseModel._load_from_strategy(m, "model.layers.3") == {}


def test_prefetch_fallback_uses_the_models_own_names():
    """Names come from layer_names_dict, never assumed to be Llama's (§5)."""
    from vaayu.vaayu_base import VaayuBaseModel
    m = _FakeModel(_names(embed="transformer.wte", head="output_head"))
    out = VaayuBaseModel._load_from_strategy(m, "output_head")
    assert out == {"output_head.weight": "EMBED_TENSOR"}
