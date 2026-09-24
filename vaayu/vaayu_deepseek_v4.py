"""DeepSeek-V4-Flash on Vaayu's streamed, layer-major path (A5.7a).

`VaayuBaseModel.forward` never calls a model's own `forward`: it calls the
embedding, each decoder layer, `run_norm` and `run_lm_head` itself, one layer
at a time, because this checkpoint is 157 G and nothing holds it resident.
So everything `DeepseekV4Model.forward` does has to be said again here, in
the hooks that loop already offers.

Design: `docs/superpowers/specs/2026-09-24-A5.7a-streamed-path-design.md`.
The durable copy is row 28b's A5.7a bullet in `todo.md`.
"""

import torch

from .vaayu_base import VaayuBaseModel


class VaayuDeepseekV4(VaayuBaseModel):
    """`set_layer_names_dict` is deliberately NOT overridden: the base's
    defaults (`model.embed_tokens`, `model.layers`, `model.norm`, `lm_head`)
    are already the names A5.6 gave this model."""

    def __init__(self, *args, **kwargs):
        """Refuse the two cache features that do not understand this cache.

        `DeepseekV4CacheLayer` holds a preallocated sliding window plus a
        compressed half and index state. `batch.cache_holds_recurrent_state`
        (`batch.py:500`), `prefix_cache.py:112`/`:364`, `kv_persist`, and
        `Cache.max_cache_len` / `max_batch_size` / `batch_repeat_interleave` /
        `batch_select_indices` all reason about a `DynamicCache`-shaped one.
        Serving a restored or sliced prefix out of this cache would return
        plausible tokens computed against the wrong history.

        Read out of `kwargs` BEFORE `super().__init__`: both options are
        passed by name everywhere a model is built, and refusing afterwards
        would mean refusing at the end of a multi-minute load.
        """
        _unsupported = {
            "enable_prefix_cache": "cross-turn prefix caching",
            "enable_kv_persist": "KV disk persistence",
        }
        for flag, what in _unsupported.items():
            if kwargs.get(flag):
                raise NotImplementedError(
                    f"{flag}=True is not supported for DeepSeek-V4: {what} "
                    f"reasons about a DynamicCache-shaped cache, and "
                    f"DeepseekV4CacheLayer is a preallocated sliding window "
                    f"plus a compressed half and index state. Restoring or "
                    f"slicing a prefix out of it would return plausible tokens "
                    f"computed against the wrong history. Run with {flag}=False "
                    f"until that cache-management code understands this layout.")

        # This architecture has NO HF attention backend -- `sparse_attn` is its
        # only kernel, which is why DeepseekV4PreTrainedModel reports "eager"
        # unconditionally. But that report is not what the engine records:
        # `init_model` stores the REQUESTED candidate (`vaayu_base.py:3373`),
        # and because the model accepts any request without raising, the first
        # candidate wins -- "flash_attention_2" on a box that has it installed.
        # `_active_attn_impl` would then open the chunk-major prefill gate
        # (`vaayu_base.py:4666`), which passes `input_ids` as a LIST of chunks
        # and trips this class's own TypeError on any prompt over
        # `prefill_chunk_size`. Requesting eager collapses the candidate list to
        # one (`vaayu_base.py:3321`) and keeps that gate shut.
        # `setdefault`, so an explicit caller choice still wins.
        kwargs.setdefault("attn_implementation", "eager")

        super().__init__(*args, **kwargs)

    def get_checkpoint_key_remap(self):
        """This checkpoint's keys do not name the module tree A5.6 built.

        `layers.3.ffn.experts.7.w1.weight` on disk, `model.layers.3.…` in the
        tree, and `mtp.*` dropped. Applied before the split, which is what makes
        `ExpertOffsetIndex` find 43 x 256 rather than the 11,776 its regex sees
        in the raw checkpoint.
        """
        from .model_code.deepseek_v4.checkpoint import remap_checkpoint_key

        return remap_checkpoint_key

    def _create_kv_cache(self):
        """A `DeepseekV4Cache`, which is the only cache this model's forward
        accepts (it raises `TypeError` on any other).

        `max_seq_len` comes from `self.max_seq_len`, which the VRAM plan
        already derived (`vaayu_base.py:2446`) -- this needs no new knob. The
        cache is PREALLOCATED (`sliding_window + max_seq_len //
        compress_ratio` slots per layer), which is exactly why
        `DeepseekV4Model.forward` refuses to default it from
        `max_position_embeddings`.
        """
        from .model_code.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Cache

        if self.kv_cache_mode not in (None, "auto"):
            print(f"[Vaayu] kv_cache_mode={self.kv_cache_mode!r} ignored: "
                  f"DeepSeek-V4 has one cache layout -- a preallocated sliding "
                  f"window plus a compressed half and its index state -- and no "
                  f"fp8 / bf16 / TurboQuant variant of it.")

        return DeepseekV4Cache(self.config, max_seq_len=self.max_seq_len,
                               max_batch_size=1, device=self.running_device,
                               dtype=self.running_dtype)

    def run_embed(self, layer, seq):
        """`[b, s, d]` -> `[b, s, hc_mult, d]`.

        Every block of this architecture reads and writes `hc_mult` copies of
        the residual (`model.py:916`). `.repeat`, not `.expand`: A5.6 measured
        the two numerically indistinguishable here and kept the reference's
        storage as a fidelity choice.
        """
        return layer(seq).unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)

    def run_norm(self, layer, seq):
        """`hc_head`, then the norm -- in that order.

        `hc_head` is the final `hc_mult` -> 1 reduction (`model.py:709`,
        called at `model.py:921`), so it collapses `[b, s, hc_mult, d]` back
        to `[b, s, d]`. A norm running first would normalise the wrong axis.
        It lives on `DeepseekV4Model`, where its parameters are -- hence the
        double hop through `self.model`.
        """
        return layer(self.model.model.hc_head(seq))

    def get_pos_emb_args(self, len_p, len_s):
        """`start_pos` IS the past length.

        This model builds no rotary table for the engine to slice: attention
        takes a scalar `start_pos` and `precompute_freqs_cis` does the rest
        inside the layer.
        """
        return {"start_pos": len_p}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s,
                                layer=None):
        """Nothing. `sparse_attn` masks by reading `-1` out of `topk_idxs`
        (`kernel.py:322`); there is no HF mask for this architecture, and
        passing one would be an unexpected kwarg on the decoder layer."""
        return {}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        """Nothing. Position is carried in `start_pos`."""
        return {}

    def forward(self, input_ids=None, **kwargs):
        """`VaayuBaseModel.forward`, wrapped in this model's id scope.

        The first `num_hash_layers` MoE layers route by TOKEN ID. Vaayu's
        `sparse_forward(hidden_states, **kwargs)` never receives ids, so
        `DeepseekV4Model.forward` stashes them in a `ContextVar` for the
        duration of its layer loop. The streamed path never reaches that
        forward, so the stash has to happen here instead.

        Chunked prefill re-enters through `self.forward` per chunk
        (`vaayu_base.py:4707`), so each chunk opens its own scope with its own
        ids -- which is correct, because those are the tokens its layers
        route. `ContextVar` makes the nesting and the release exception-safe.
        """
        from .model_code.deepseek_v4.modeling_deepseek_v4 import input_ids_scope

        if input_ids is None:
            raise ValueError(
                "DeepSeek-V4 needs input_ids: its first `num_hash_layers` MoE "
                "layers route by TOKEN ID, not by score, so inputs_embeds alone "
                "has nothing to route on. Refusing rather than routing every "
                "token to the same experts.")

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"DeepSeek-V4 needs input_ids as a tensor, got "
                f"{type(input_ids).__name__}. The chunk-major prefill branch "
                f"(`vaayu_base.py:4674`) passes a LIST of 1-D chunks, and the "
                f"hash-routed layers would be handed that list. That branch is "
                f"gated on flash/sdpa and this model forces eager, so it should "
                f"not be reachable -- if you are seeing this, that gate changed "
                f"and this model needs per-chunk scoping before it can use it.")

        if input_ids.shape[0] > 1:
            raise NotImplementedError(
                f"DeepSeek-V4 runs one sequence at a time, got "
                f"{input_ids.shape[0]}. The ids the hash-routed layers read are "
                f"held in a ContextVar -- one per REQUEST, not per batch element "
                f"-- and the layer-major loop runs every batch element inside a "
                f"single scope, so sequence 1 onward would route on sequence 0's "
                f"ids and silently return plausible logits from the wrong "
                f"experts. The server is N=1 today; batching this model needs "
                f"per-element scoping first.")

        with input_ids_scope(input_ids):
            return super().forward(input_ids, **kwargs)
