"""DeepSeek-V4-Flash on Vaayu's streamed, layer-major path (A5.7a).

`VaayuBaseModel.forward` never calls a model's own `forward`: it calls the
embedding, each decoder layer, `run_norm` and `run_lm_head` itself, one layer
at a time, because this checkpoint is 157 G and nothing holds it resident.
So everything `DeepseekV4Model.forward` does has to be said again here, in
the hooks that loop already offers.

Design: `docs/superpowers/specs/2026-09-24-A5.7a-streamed-path-design.md`.
The durable copy is row 28b's A5.7a bullet in `todo.md`.
"""

from .vaayu_base import VaayuBaseModel


class VaayuDeepseekV4(VaayuBaseModel):
    """`set_layer_names_dict` is deliberately NOT overridden: the base's
    defaults (`model.embed_tokens`, `model.layers`, `model.norm`, `lm_head`)
    are already the names A5.6 gave this model."""

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
