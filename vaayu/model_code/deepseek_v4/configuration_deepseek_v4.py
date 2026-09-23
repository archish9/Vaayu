"""`DeepseekV4Config` -- the config transformers 5.5.4 does not ship.

transformers has `deepseek_v2/v3/vl` and no `deepseek_v4`, and the checkpoint
ships no `auto_map`, so `AutoConfig.from_pretrained` raises and Vaayu dies at
`vaayu_base.py:2131` before it reaches a weight. MiniMax-M2.7 is the precedent:
unknown to transformers, runs here because its repo carries its own
`configuration_*.py` plus an `auto_map` entry.

This file is copied verbatim into the checkpoint directory, where that
`auto_map` points at it. The repo holds the source of truth; the checkpoint
holds the copy, and `tests/test_deepseek_v4_config.py` asserts the two are
byte-identical.

Every field defaults to `None`. The checkpoint sets all of them, and a
plausible-but-wrong default loads silently -- `deepseek_v3`'s 61 layers where
this model has 43 -- which is worse than a missing one. `ModelArgs` (the
reference's dataclass in `inference/model.py`) is deliberately NOT ported: it
is a second source of truth with its own defaults, and A5.2 builds one FROM
this config, one direction only.
"""
from transformers.configuration_utils import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):

    model_type = "deepseek_v4"

    # shape
    vocab_size: int | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    head_dim: int | None = None
    hidden_act: str | None = None
    initializer_range: float | None = None
    rms_norm_eps: float | None = None
    tie_word_embeddings: bool | None = None
    use_cache: bool | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    attention_bias: bool | None = None
    attention_dropout: float | None = None

    # MLA -- one 512-dim latent shared by 64 heads, grouped low-rank O
    q_lora_rank: int | None = None
    o_lora_rank: int | None = None
    o_groups: int | None = None
    qk_rope_head_dim: int | None = None
    num_key_value_heads: int | None = None

    # sparse index attention. `compress_ratios` carries 46 entries for 43
    # layers: the extra three belong to the MTP modules, which A5.7 excludes.
    index_topk: int | None = None
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    sliding_window: int | None = None
    compress_ratios: list | None = None
    compress_rope_theta: float | None = None

    # MoE -- 94.3 % of the checkpoint's bytes
    n_routed_experts: int | None = None
    num_experts_per_tok: int | None = None
    n_shared_experts: int | None = None
    moe_intermediate_size: int | None = None
    num_hash_layers: int | None = None
    scoring_func: str | None = None
    topk_method: str | None = None
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None
    swiglu_limit: float | None = None
    expert_dtype: str | None = None

    # hyper-connections (A5.6)
    hc_mult: int | None = None
    hc_sinkhorn_iters: int | None = None
    hc_eps: float | None = None

    # rope. `rope_scaling` is the shipped key; PretrainedConfig standardizes it
    # onto `rope_parameters` and folds `rope_theta` and
    # `original_max_position_embeddings` into that dict. Declaring
    # `rope_parameters` keeps the base class off its "hasn't set it as
    # attribute" BC path, and `max_position_embeddings` must be a field because
    # rope standardization reads it BEFORE the leftover kwargs are assigned.
    rope_theta: float | None = None
    max_position_embeddings: int | None = None
    rope_parameters: dict | None = None

    # MTP / DSpark -- carried so the config round-trips, read by nothing.
    # `num_nextn_predict_layers` says 1; the checkpoint holds `mtp.0/1/2` and
    # the reference's own `inference/config.json` says `n_mtp_layers: 3`. The
    # field is WRONG -- A5.7's index must not derive the MTP skip from it.
    num_nextn_predict_layers: int | None = None
    dspark_block_size: int | None = None
    dspark_noise_token_id: int | None = None
    dspark_target_layer_ids: list | None = None
    dspark_markov_rank: int | None = None

    # The checkpoint's `quantization_config` under a name nothing dispatches on.
    # It declares `quant_method: "fp8"`, and `vaayu_base.py:3394` hands any
    # non-exl3 method to `AutoHfQuantizer` -- a FineGrainedFP8HfQuantizer would
    # then replace every Linear, which is wrong twice over: 94 % of these
    # weights are fp4-in-I8 that the block names nothing about, and A5.3
    # dequantises through DeepSeek's own TileLang kernels. Renamed rather than
    # dropped because `fmt`/`scale_fmt` are the only source for the reference
    # ModelArgs' `dtype`/`scale_fmt` (A5.2).
    checkpoint_quantization_config: dict | None = None

    def __post_init__(self, **kwargs):
        quantization_config = kwargs.pop("quantization_config", None)
        if quantization_config is not None:
            self.checkpoint_quantization_config = quantization_config
        super().__post_init__(**kwargs)
