"""DeepSeek-V4-Flash on Vaayu's streamed, layer-major path (A5.7a).

`VaayuBaseModel.forward` never calls a model's own `forward`: it calls the
embedding, each decoder layer, `run_norm` and `run_lm_head` itself, one layer
at a time, because this checkpoint is 157 G and nothing holds it resident.
So everything `DeepseekV4Model.forward` does has to be said again here, in
the hooks that loop already offers.

Design: `docs/superpowers/specs/2026-09-24-A5.7a-streamed-path-design.md`.
The durable copy is row 28b's A5.7a bullet in `todo.md`.
"""

import sys
from pathlib import Path

import torch

from .vaayu_base import VaayuBaseModel


class VaayuDeepseekV4(VaayuBaseModel):
    """`set_layer_names_dict` is deliberately NOT overridden: the base's
    defaults (`model.embed_tokens`, `model.layers`, `model.norm`, `lm_head`)
    are already the names A5.6 gave this model."""

    def __init__(self, *args, **kwargs):
        """Refuse speculative decoding by name; normalise attn_implementation.

        `drafter_model_path` is refused: `cache_holds_recurrent_state` finds no
        recurrent state on `DeepseekV4CacheLayer`, so `spec_recurrent_refusal`
        returns `None` and drafting would start -- then the first REJECTED
        draft reaches `_trim_kv_cache` (`vaayu_base.py:5428`) ->
        `DeepseekV4Cache.crop`, which cannot rewind a preallocated sliding
        window plus a compressed half. Nothing else refuses it, so it is
        refused here.

        Prefix caching and KV persistence are NOT refused here -- this class's
        own `_refuse_unsupported_prefix_cache` override (below) handles them
        by detection against `DeepseekV4Cache` / `DeepseekV4CacheLayer`,
        without constructing a probe cache (Task 6a: the base's own probe,
        `vaayu_base.py:3770`, constructs one via `_create_kv_cache` and
        crashes there -- `DeepseekV4Cache` is preallocated from `max_seq_len`,
        which is still `None` at the point this runs). It still finds no
        prefix-store surface on `DeepseekV4Cache`, warns once and clears the
        flag; `kv_persist` then prints `ignored: enable_prefix_cache is False`
        (`:3097`). That matters because a hard refusal here used to run FIRST
        and make the prefix-cache check dead code on this path, which left the
        server's DEFAULT config (`ENABLE_PREFIX_CACHE = True`, passed
        unconditionally at `app.py:406`) unable to load this model at all.

        `attn_implementation` is normalised from absent-or-`None` to `"eager"`
        (an explicit non-`None` request still wins), which keeps the
        chunk-major prefill gate shut -- see the comment below.

        Read out of `kwargs` BEFORE `super().__init__`: every option here is
        passed by name everywhere a model is built, and refusing afterwards
        would mean refusing at the end of a multi-minute load.
        """
        # Prefix caching and KV persistence are NOT refused here. This class's
        # own `_refuse_unsupported_prefix_cache` override (below) does it, by
        # detection against the CLASSES rather than by probing an instance --
        # see that method's docstring for why (Task 6a: the base's instance
        # probe, vaayu_base.py:3770, crashes on this model). It finds no
        # prefix-store surface on `DeepseekV4Cache`, prints one WARNING and
        # clears the flag; `kv_persist` then prints `ignored:
        # enable_prefix_cache is False` (`:3097`). A hard refusal here ran
        # FIRST and made that check dead code on this path, which meant the
        # server's DEFAULT config (`ENABLE_PREFIX_CACHE = True`, passed
        # unconditionally at `app.py:406`) could not load this model at all.
        #
        # Speculative decoding IS refused, because nothing else refuses it:
        # `cache_holds_recurrent_state` finds no conv_states here, so
        # `spec_recurrent_refusal` returns None and drafting starts -- and then
        # the first REJECTED draft reaches `_trim_kv_cache`
        # (`vaayu_base.py:5428`) -> `DeepseekV4Cache.crop`, which cannot rewind
        # a sliding window plus a compressed half. Read out of kwargs BEFORE
        # `super().__init__`, so a bad flag is not learned at the end of a
        # multi-minute load.
        if kwargs.get("drafter_model_path"):
            raise NotImplementedError(
                "drafter_model_path is not supported for DeepSeek-V4: rejecting "
                "a draft rewinds the KV cache through _trim_kv_cache, and this "
                "cache is a preallocated sliding window indexed by "
                "pos % window_size plus a Compressor that flushes on boundaries "
                "counted from start_pos -- crop() cannot put either back. Run "
                "without a drafter until that is built.")

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
        #
        # Absent AND explicitly-None both mean "no choice made" -- `setdefault`
        # only covers absent, and the server never omits the key
        # (`app.py:418` passes `config.attn_implementation`, default None at
        # `server/config.py:209`). With None left in place, `_forced_attn_impl`
        # stays None, `init_model`'s candidate list becomes
        # ["flash_attention_2", "sdpa", "eager"], and the chunk-major prefill
        # gate (`vaayu_base.py:4666`) opens -- which hands `forward` a LIST of
        # chunks that this class refuses. A real explicit choice still wins.
        if kwargs.get("attn_implementation") is None:
            kwargs["attn_implementation"] = "eager"

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

    def init_model(self):
        """Put `fp4_group_size` on the config before the skeleton is built.

        A5.4's `DeepseekV4Linear` refuses to invent a group size and raises a
        named `ValueError`, and `init_model` is the first thing that calls
        `AutoModelForCausalLM.from_config`. Derived from the split shards, so
        it keeps working after `delete_original` has reclaimed the originals.
        """
        from .model_code.deepseek_v4.checkpoint import fp4_group_size_from_split

        if getattr(self.config, "fp4_group_size", None) is None:
            self.config.fp4_group_size = fp4_group_size_from_split(
                self.checkpoint_path, self.layer_names_dict["layer_prefix"])
            print(f"[Vaayu] DeepSeek-V4 fp4_group_size={self.config.fp4_group_size} "
                  f"(derived from the shard headers; the config carries none)")

        super().init_model()

        # `remap_checkpoint_key` names the hc_head_* tensors `model.norm.hc_head_*`
        # so they land in the `model.norm.` shard (the substring split has no
        # other prefix they match -- see checkpoint.py and utils.py's G2 guard).
        # Their real module path is `model.hc_head_*`: `hc_head` is a method
        # and its parameters live directly on `DeepseekV4Model`, not under its
        # `norm` submodule. Added to the dict `super().init_model()` just
        # created, never replacing it -- the base fills it for its own reasons
        # (`_resolve_and_map_path`). `getattr` defends the same way
        # `vaayu_base.py` reads this dict elsewhere: a stubbed-out
        # `super().init_model()` (as in this file's own tests) need not create
        # one for this to still be safe.
        self._path_remappings = getattr(self, "_path_remappings", {})
        self._path_remappings["model.norm.hc_head_"] = "model.hc_head_"

        # DeepSeek-V4's TileLang kernels ship inside the checkpoint, under
        # `inference/`, beside the ORIGINAL shards -- `model_local_path`, not
        # `checkpoint_path` (the `splitted_model/` subdirectory Vaayu writes
        # for its own streamed reads, which holds no `inference/`). `kernels()`
        # (`model_code/deepseek_v4/modeling_deepseek_v4.py`) reads a module
        # global, `KERNEL_DIR`, that A5.7's loader was always meant to set and
        # never did.
        #
        # The class `super().init_model()` just built is loaded through
        # `trust_remote_code`, so it lives in the HuggingFace modules cache
        # (`~/.cache/huggingface/modules/transformers_modules/...`) -- a
        # DIFFERENT module object than this repo's own copy of
        # `modeling_deepseek_v4.py`, with its own separate `KERNEL_DIR` global.
        # Setting the repo module's global would be silently useless: resolve
        # the module the loaded class actually lives in, at runtime, rather
        # than importing the repo module or hardcoding the cache's mangled
        # path.
        kernel_dir = Path(self.model_local_path) / "inference"
        kernel_source = kernel_dir / "kernel.py"
        if not kernel_source.is_file():
            raise FileNotFoundError(
                f"DeepSeek-V4's TileLang kernels are not at {kernel_source}. "
                f"They ship inside the checkpoint, under `inference/`, beside "
                f"the original shards at {self.model_local_path}.")

        target_module = sys.modules[type(self.model).__module__]
        target_module.KERNEL_DIR = kernel_dir
        print(f"[Vaayu] DeepSeek-V4 kernels found at {kernel_dir}")

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

    def _refuse_unsupported_prefix_cache(self):
        """Answer the base's question (`vaayu_base.py:3770`) WITHOUT
        constructing a probe cache -- constructing one is exactly what crashes
        this model's `__init__`.

        Ordering, by line number: `VaayuBaseModel.__init__` assigns the
        CALLER's `max_seq_len` (`None` by default) at `:2255`, calls
        `_refuse_unsupported_prefix_cache` at `:2274`, and only assigns the
        VRAM-plan-derived value (`self._vram_plan.max_seq_len`) at `:2461` --
        after this call. The base answers by building a real cache
        (`self._create_kv_cache()`) and inspecting it, which is fine for
        every other model: `DynamicCache` / `FP8DynamicCache` /
        `TurboQuantKVCache` all size lazily and construct fine with
        `max_seq_len=None`. `DeepseekV4Cache` is PREALLOCATED from
        `max_seq_len` (`modeling_deepseek_v4.py:728`: `window_size +
        max_seq_len // compress_ratio`) and raises `TypeError` on `None //
        int` before `max_seq_len` exists -- and this is unconditional:
        `_create_kv_cache` above always returns a `DeepseekV4Cache`
        regardless of `kv_cache_mode`, so the base's probe would crash on
        every load, not just some configurations of this model.

        So this override answers by DETECTION against the CLASSES
        (`DeepseekV4Cache`, `DeepseekV4CacheLayer`) instead of an instance:

        - `cache_supports_prefix_store` (`batch.py:479`) only does
          `callable(getattr(cache, m, None))` for the three prefix-store
          methods. Methods are class attributes, so this answers identically
          for the class and for any instance -- called unchanged, on
          `DeepseekV4Cache` itself.
        - `cache_holds_recurrent_state` (`batch.py:500`) walks `cache.layers`
          and checks each layer for `update_conv_state`/`conv_states` --
          again a `hasattr` capability test, but `layers` itself only exists
          on a constructed instance, and no instance is available here (that
          is the whole problem). A `SimpleNamespace(layers=[DeepseekV4CacheLayer])`
          stands in: the class is the closest thing to a layer instance short
          of building one, and the two attributes the predicate checks for
          are class-level (methods) or absent either way, so the class
          answers exactly as a real instance would.

        If `DeepseekV4Cache` ever grows `to_cpu`/`to_device`/`resident_bytes`,
        `cache_supports_prefix_store` starts returning `True` off the class
        alone and this override starts ALLOWING prefix caching automatically
        -- exactly as the base's instance probe would. Do not delete this as
        redundant with the base: the base's own probe is unreachable here (it
        would raise before it could answer), which is the whole reason this
        override exists.
        """
        from types import SimpleNamespace

        from .batch import cache_holds_recurrent_state, cache_supports_prefix_store
        from .model_code.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4Cache, DeepseekV4CacheLayer)

        # Recorded whether or not the cache is refused -- see the base's own
        # comment (vaayu_base.py:3788): the KV-only side channels (disk
        # persistence, TurboQuant pooling) read this and need False, not
        # absent, when the prefix cache is off entirely.
        self._cache_holds_recurrent_state = cache_holds_recurrent_state(
            SimpleNamespace(layers=[DeepseekV4CacheLayer]))
        if not getattr(self, "enable_prefix_cache", False):
            return
        if not cache_supports_prefix_store(DeepseekV4Cache):
            print(f"[Vaayu-Prefix] WARNING: cross-turn prefix cache DISABLED — "
                  f"{DeepseekV4Cache.__name__} does not implement "
                  f"to_cpu/to_device/resident_bytes, and DeepSeek-V4 has no "
                  f"other cache layout to fall back to (kv_cache_mode is "
                  f"ignored for this model -- see _create_kv_cache above).")
            self.enable_prefix_cache = False

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

    def get_past_key_value_args(self, past_key_values):
        """Pass the cache through, and remember how far it has advanced.

        `DeepseekV4Model._start_pos` validates that `cache_position` is
        `arange(start_pos, start_pos + seqlen)` and raises otherwise. The
        streamed path never calls that forward, so the check has to live in the
        hooks -- recorded here and asserted in `get_pos_emb_args`, which the
        layer-major loop evaluates immediately after this one
        (`vaayu_base.py:5222`).
        """
        self._cache_seq_len = (
            past_key_values.get_seq_length() if past_key_values is not None else None)
        return {"past_key_values": past_key_values}

    def get_pos_emb_args(self, len_p, len_s):
        """`start_pos` IS the past length.

        This model builds no rotary table for the engine to slice: attention
        takes a scalar `start_pos` and `precompute_freqs_cis` does the rest
        inside the layer.

        Checked against the cache, because nothing else checks it: the window
        indexes by `pos % window_size` and the Compressor flushes on block
        boundaries counted from `start_pos`, so a cache that was rewound under a
        `start_pos` that was not writes into the wrong slots and returns
        plausible tokens computed against the wrong history.
        """
        _cached = getattr(self, "_cache_seq_len", None)
        if _cached is not None and _cached != len_p:
            raise RuntimeError(
                f"DeepSeek-V4 start_pos {len_p} disagrees with the cache's "
                f"{_cached} tokens. This cache is a preallocated sliding window "
                f"indexed by pos % window_size; a discontiguous start_pos writes "
                f"into the wrong slots and the Compressor flushes on the wrong "
                f"boundary. Something rewound or replayed the cache without "
                f"telling the engine.")
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
