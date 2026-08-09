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
