"""D4: a tied-weight checkpoint has no lm_head weight, and the splitter
expected one anyway.

`layers` is built unconditionally ending in the lm_head entry, but a checkpoint
with `tie_word_embeddings: true` stores no `lm_head.weight` -- so the index has
no key for it. That produced TWO different failures depending on the source
layout:

  * single-file checkpoints (the shard-loading block is skipped entirely):
    layer_state_dict came out {}, and a valid EMPTY 16-byte shard plus its
    `.done` marker got written -- so `all(found_layers.values())` reported the
    split COMPLETE. An incomplete split that verifies.
  * sharded checkpoints: the write loop hit its "No weights for this layer in
    index, skip it" continue, so nothing was written and found_layers[lm_head]
    stayed False forever -- every subsequent load re-split the whole model.

Both come from expecting a layer the checkpoint does not have. Deciding that
from the INDEX rather than from a config flag is detection over assumption
(CLAUDE.md §5) and covers every reason the weight can be absent.
"""
from vaayu.utils import drop_absent_lm_head


def test_lm_head_is_kept_when_the_index_has_it():
    layers = ["model.embed_tokens.", "model.layers.0.", "model.norm.", "lm_head."]
    index = {"model.embed_tokens.weight": "s.safetensors",
             "model.layers.0.q_proj.weight": "s.safetensors",
             "model.norm.weight": "s.safetensors",
             "lm_head.weight": "s.safetensors"}
    assert drop_absent_lm_head(layers, index, "lm_head.") == layers


def test_lm_head_is_dropped_when_the_checkpoint_ties_it():
    layers = ["model.embed_tokens.", "model.layers.0.", "model.norm.", "lm_head."]
    index = {"model.embed_tokens.weight": "s.safetensors",
             "model.layers.0.q_proj.weight": "s.safetensors",
             "model.norm.weight": "s.safetensors"}
    assert drop_absent_lm_head(layers, index, "lm_head.") == layers[:-1]


def test_only_the_lm_head_entry_is_ever_dropped():
    """A decoder layer missing from the index is a real, loud problem -- this
    helper must not quietly paper over it the way it does for a tied head."""
    layers = ["model.embed_tokens.", "model.layers.0.", "model.norm.", "lm_head."]
    index = {"lm_head.weight": "s.safetensors"}
    assert drop_absent_lm_head(layers, index, "lm_head.") == layers


def test_a_custom_lm_head_name_is_honoured():
    """layer_names lets an architecture rename its head; the check must follow
    that name, not the default string."""
    layers = ["tok_emb.", "blocks.0.", "final_norm.", "output_head."]
    index = {"tok_emb.weight": "s.safetensors", "blocks.0.w.weight": "s.safetensors",
             "final_norm.weight": "s.safetensors"}
    assert drop_absent_lm_head(layers, index, "output_head.") == layers[:-1]
