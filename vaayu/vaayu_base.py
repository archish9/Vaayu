import atexit
import gc
import os
import subprocess
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, LlamaForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights

from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer, HfQuantizer

from .profiler import LayeredProfiler
from .persist.loader_strategy import build_loader_strategy, LoaderStrategy
from .telemetry import VaayuTelemetry, LayerTiming
from .persist.safetensor_model_persister import get_and_clear_pinned_timings as _get_layer_disk_timings
from .persist.safetensor_model_persister import get_expert_pool_size
from .prefix_cache import PrefixCacheStore
from . import turboquant

# BetterTransformer is deprecated in newer optimum versions
try:
    from optimum.bettertransformer import BetterTransformer
    better_transformer_available = True
except (ImportError, ModuleNotFoundError):
    better_transformer_available = False
    print('>>>> BetterTransformer not available, will use sdpa attention implementation')

from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False



try:
    from transformers.cache_utils import Cache, DynamicCache

    cache_utils_installed = True
    print('>>>> cache_utils installed')
except ImportError:
    cache_utils_installed = False


def _auto_session_id(token_ids: list) -> str:
    """Stable key for a session the caller didn't name — hash of the first
    tokens. Lets auto-prefix-match sequences be stored and re-found."""
    import hashlib
    head = bytes(str(token_ids[:64]), "utf-8")
    return "auto:" + hashlib.sha1(head).hexdigest()[:16]


def compute_moe_geometry(expert_offset_index, num_layers, moe_top_k, moe_num_experts):
    """Model-agnostic MoE geometry, derived — never hardcoded (REPORT §1.5).

    Returns dict: expert_bytes, num_moe_layers, top_k, num_experts,
    working_set_bytes (= top_k * num_moe_layers * expert_bytes, one decode step).
    """
    # Sample one full layer's experts to derive expert_bytes. Uniform for
    # trellis-quant checkpoints (constant size); if a future MoE has
    # non-uniform experts, take the MAX (conservative — never underestimates
    # the working set) and WARN (CLAUDE.md §5: don't silently degrade).
    sizes = []
    for layer_tensors in expert_offset_index.index.values():
        for expert_tensors in layer_tensors.values():
            sizes.append(sum(t.byte_length for t in expert_tensors.values()))
        if sizes:
            break  # one layer's experts are a sufficient uniformity sample
    expert_bytes = max(sizes) if sizes else 0
    if sizes and min(sizes) != max(sizes):
        print(f"[Vaayu-MoE] WARNING: non-uniform expert sizes "
              f"(min={min(sizes)} max={max(sizes)}); using max for cache sizing")
    num_moe_layers = len(expert_offset_index.index) or max(1, num_layers - 3)
    working_set_bytes = moe_top_k * num_moe_layers * expert_bytes
    return {
        "expert_bytes": expert_bytes,
        "num_moe_layers": num_moe_layers,
        "top_k": moe_top_k,
        "num_experts": moe_num_experts,
        "working_set_bytes": working_set_bytes,
    }


def compute_spec_working_set(geometry: dict, verify_positions: int) -> int:
    """Expert working set of ONE speculative verify pass, in bytes (Fix #9).

    A verify pass routes `verify_positions` tokens through every MoE layer in a
    single forward; per layer the unique-expert union is bounded by
    min(num_experts, verify_positions * top_k). Conservative upper bound — cache
    sizing must not underestimate thrash (REPORT §2.4). verify_positions=1
    reduces to the single-token working set.
    """
    per_layer = min(
        geometry["num_experts"],
        max(1, verify_positions) * geometry["top_k"],
    )
    return per_layer * geometry["num_moe_layers"] * geometry["expert_bytes"]


def spec_decode_fits(geometry: dict, draft_tokens: int, budget_bytes: int) -> bool:
    """True if a speculative verify pass's expert working set fits the expert
    cache budget (Fix #9). draft_tokens drafts + 1 correction = verify positions."""
    return compute_spec_working_set(geometry, draft_tokens + 1) <= budget_bytes


# Hardware constant for THIS 32 GB box: max safe expert-cache size before OOM.
# Model-independent. Raised from the old 7 GB only after shrinking other RAM
# consumers (below). Do NOT exceed 9 GB — 10 GB caused a confirmed OOM-kill.
# HARDWARE CONSTANT, and now a FLOOR rather than a cap. Set 2026-07-10, when the
# pinned host allocator charged the next power of two for every request -- a
# 9.13 GB arena cost 16 GiB resident. The chunked-arena fix (2026-08-07) made
# chunks charge exactly, and this figure was never revisited. Measured
# 2026-08-16 on Qwen3-235B (2 sessions x 3 rounds, real context, MemoryMax=27G):
#
#   ceiling  8.5 GiB -> expert  9.12 GB, prefix 11.73 GB, warm wall 36.6 s
#   ceiling 13   GiB -> expert 13.95 GB, prefix  7.45 GB, warm wall 32.0 s
#   ceiling 18   GiB -> expert 19.33 GB, prefix  2.00 GB, warm wall 27.3 s
#
# -25 % end to end, with prefix reuse unchanged at 83 % in every arm. It is kept
# as the floor because it is the figure known to run on the smallest box this
# repo targets; derive_expert_ceiling raises it from available RAM.
_EXPERT_CACHE_OOM_CEILING = int(8.5 * 1024 ** 3)

# HARDWARE CONSTANT: RAM left for the OS, the CUDA host context and post-sizing
# host allocations when deriving the expert ceiling. Property of the 32 GB box.
_EXPERT_CEILING_RESERVE = int(3.5 * 1024 ** 3)


def derive_expert_ceiling(available_bytes: int, prefix_sessions: int,
                          kv_bytes_per_token: int, max_seq_len: int,
                          prefix_enabled: bool = True) -> int:
    """Ceiling for the expert RAM cache, derived rather than asserted.

    The expert arena is pinned BEFORE the prefix pool is sized
    (compute_prefix_cache_budget runs later and measures what is left), so an
    expert cache that takes everything silently starves cross-turn reuse. Reuse
    does not degrade gracefully: one session past capacity it collapses to ZERO,
    because each recomputed session's re-store cascade-evicts the ones before it
    (Plan I). So room for `prefix_sessions` full-length sessions is subtracted
    FIRST, and the expert cache gets what remains.

    Every term is derived from geometry (CLAUDE.md §5) -- `kv_bytes_per_token`
    and `max_seq_len` come from the model, `prefix_sessions` is a policy choice.
    Only the reserve and the floor are hardware literals.
    """
    reserve = _EXPERT_CEILING_RESERVE
    if prefix_enabled and prefix_sessions > 0:
        reserve += int(prefix_sessions) * int(max_seq_len) * int(kv_bytes_per_token)
    derived = int(available_bytes) - reserve
    # Floor, not cap: 8.5 GiB is the figure known to run everywhere.
    return max(_EXPERT_CACHE_OOM_CEILING, min(derived, int(available_bytes)))


def compute_expert_budget(
    available_bytes: int,
    working_set_bytes: int,
    num_moe_layers: int,
    expert_bytes: int,
    moe_top_k: int,
    moe_num_experts: int,
    ceiling_bytes: int | None = None,
):
    """Size the expert RAM cache to exceed the DERIVED decode working set, up to
    the hardware OOM ceiling. Model-agnostic (Fix #1, REPORT §1.5).

    working_set_bytes comes from compute_moe_geometry — never hardcoded. When RAM
    allows, budget > working_set → no thrash (Qwen: raises to ~8.5 GB; M2.5: its
    5.5 GB working set already fits). When the model's working set exceeds the
    achievable budget, the caller warns and proceeds (real hardware limit).

    Returns (expert_budget_bytes, ne_budget_bytes, max_per_layer).
    """
    GB = 1024 ** 3
    MB = 1024 ** 2
    # Non-expert cache shrunk 2GB -> 0.5GB: all layers are non-expert-pinned, so
    # this LRU is idle during decode (measured cache=0hit/0miss).
    ne_budget = min(512 * MB, int(available_bytes / 8))
    overhead = (
        int(1.5 * GB)        # OS + CUDA context margin (was 2 GB)
        + ne_budget          # non-expert tensor cache
        + 512 * MB           # io_uring expert buffer pool
    )
    # ceiling_bytes: None keeps the historical literal, so every existing caller
    # and test is unchanged; vaayu_base passes a figure from
    # derive_expert_ceiling, which reserves for the prefix pool first.
    _ceiling = _EXPERT_CACHE_OOM_CEILING if ceiling_bytes is None else int(ceiling_bytes)
    expert_budget = min(
        _ceiling,
        max(1 * GB, available_bytes - overhead),
    )
    # max_per_layer derived from expert_bytes (any quant/model). Clamped so the
    # collective per-layer capacity never exceeds the global budget (Fix #4).
    per_layer_bytes = expert_budget // max(1, num_moe_layers)
    max_per_layer = max(
        moe_top_k,
        min(moe_num_experts, int(per_layer_bytes // max(1, expert_bytes))),
    )
    return expert_budget, ne_budget, max_per_layer


def decode_stats_reset(*, is_moe: bool, has_expert_cache: bool,
                       is_prefill_pass: bool,
                       already_reset: bool) -> tuple[bool, bool]:
    """Should this forward pass reset the expert-cache counters, and what does
    the latch become? Returns (reset_now, latched).

    Decode hit-rate telemetry has to exclude prefill, which routes ~every expert
    once and so misses ~100% by construction. The reset therefore fires on the
    FIRST DECODE PASS of a turn -- and the latch that stops it firing again must
    drop when the next prefill runs, or it only ever fires once per process
    (F13). Turn 2's prefill pollutes the counters exactly as turn 1's did.

    Pure so the state machine is testable without a model load, per todo.md's
    METHOD NOTES.
    """
    if is_prefill_pass:
        return False, False          # the polluter itself: never reset, re-arm
    if is_moe and has_expert_cache and not already_reset:
        return True, True
    return False, already_reset


def unclamped_slab_count(budget_bytes: int, slab_bytes: int, pool_size: int) -> int:
    """Slabs the expert cache budget asks for, before allocator-round-up clamping:
    the cache's own slabs plus `pool_size` extra so in-flight loader threads never
    starve while the cache is full."""
    return max(1, int(budget_bytes) // int(slab_bytes)) + int(pool_size)


def clamp_slab_count(budget_bytes: int, slab_bytes: int, pool_size: int,
                      working_set_bytes: int = 0) -> int:
    """Slab count whose pinned backing store is a power of two.

    The backing store is ONE pinned allocation, and the caching host allocator
    charges the next power of two >= its size — so a backing store a hair over a
    power of two costs nearly double, and the surplus is never addressable. Clamp
    the whole backing store (cache slabs AND the in-flight staging addend, which
    must stay INSIDE the clamped total or it pushes the size back over the
    boundary) down to the power of two already being paid for.

    No longer on the arena's construction path as of Plan L (chunked backing is
    charged exactly per chunk, so there is nothing to clamp) — retained because it
    still describes real allocator behaviour for any future single pinned allocation.

    No-op when the unclamped backing size is already an exact power of two — which
    is exactly the case where the allocator was not over-charging. Never returns
    fewer than `pool_size` slabs: the loader's landing zone survives any clamp.

    `working_set_bytes` (default 0, a true no-op) floors the clamp at the model's
    working set: measured on-device, the clamp cut Qwen3-235B's cache budget from
    9.13 GB to 8.49 GB and crossed its 8.90 GB working-set boundary, costing 48% of
    decode throughput (MiniMax's 4.40 GB working set was nowhere near either
    boundary, so it clamped for free). Clamp only when the clamped cache still
    covers the working set; when the clamped cache would NOT cover it but the
    unclamped one does, skip the clamp — a fitting model must not be made to
    thrash. When the working set fits neither way it thrashes regardless, so take
    the RAM and clamp anyway.
    """
    from .persist.safetensor_model_persister import largest_pow2_at_most
    slab_bytes = int(slab_bytes)
    pool_size = int(pool_size)
    working_set_bytes = int(working_set_bytes)
    n = unclamped_slab_count(budget_bytes, slab_bytes, pool_size)
    max_backing = largest_pow2_at_most(n * slab_bytes)
    n_clamped = max(pool_size, max_backing // slab_bytes)
    # Compare CACHE bytes only (exclude the pool_size staging addend), matching
    # what the cache actually gets (Task 1 Step 7).
    unclamped_cache_bytes = max(0, n - pool_size) * slab_bytes
    clamped_cache_bytes = max(0, n_clamped - pool_size) * slab_bytes
    if clamped_cache_bytes < working_set_bytes <= unclamped_cache_bytes:
        return n
    return n_clamped


def arena_slab_count(budget_bytes: int, slab_bytes: int, pool_size: int,
                     working_set_bytes: int = 0, *, pin: bool) -> int:
    """Slab count for a single-pinned-allocation arena, clamped if pinning.

    No longer called from the arena's construction path as of Plan L (chunked
    backing is charged exactly per chunk, so there is nothing to clamp) —
    retained because it still describes real allocator behaviour for any future
    single pinned allocation. Do not wire this back into the loader.

    Power-of-two charging is a property of the caching HOST allocator, i.e.
    pinned memory only: a pageable arena pays no round-up, so clamping it would
    cost cache for zero RAM freed. Clamp only when we are actually about to pin;
    `working_set_bytes` is inert on the pageable path for the same reason (it
    exists solely to floor the clamp).
    """
    if not pin:
        return unclamped_slab_count(budget_bytes, slab_bytes, pool_size)
    return clamp_slab_count(budget_bytes, slab_bytes, pool_size, working_set_bytes)


def cache_budget_for_arena(n_slabs: int, slab_bytes: int, pool_size: int) -> int:
    """Bytes the expert RAM cache may charge against an arena of `n_slabs`.

    Excludes the `pool_size` in-flight staging addend: those slabs are the
    loader's landing zone and must stay permanently free, not be handed to the
    cache as usable budget (see `unclamped_slab_count`). Returns 0 for an arena
    with nothing beyond that reserve — the caller must WARN there rather than
    disable the cache silently (CLAUDE.md §5).
    """
    return max(0, int(n_slabs) - int(pool_size)) * int(slab_bytes)


def arena_backing_is_sufficient(n_slabs: int, requested_n_slabs: int,
                                slab_bytes: int, pool_size: int,
                                working_set_bytes: int) -> bool:
    """Whether a possibly-degraded chunked arena is worth keeping.

    A chunked arena keeps whatever chunks it secured when a later one fails to
    allocate (Plan L / E5): those chunks are still PINNED, and falling back to a
    pageable arena costs 19.8 -> ~7.1 GB/s across the whole expert path, far more
    than a slightly smaller cache. But a cache below the model's working set
    thrashes — measured on Qwen3-235B, 0.85 -> 0.43 tok/s with the expert RAM
    cache falling 47% -> 0% hit — so a shortfall that crosses that boundary is
    not worth keeping and the caller should fall back instead.

    Compares the CACHE share (excluding the staging reserve), matching
    clamp_slab_count. `working_set_bytes=0` is a true no-op.
    """
    if n_slabs >= requested_n_slabs:
        return True
    return cache_budget_for_arena(n_slabs, slab_bytes, pool_size) >= int(working_set_bytes)


def degraded_arena_decision(n_slabs: int, requested_n_slabs: int, slab_bytes: int,
                            pool_size: int, working_set_bytes: int,
                            pinned: bool) -> tuple:
    """Whether to keep a partially-allocated arena, and the operator-facing reason.

    Returns (keep, why). Split out from the loader so the branch selection and its
    message are testable without a model load -- the two defects this replaced
    (a hardcoded "(pinned)" on a pageable arena, and a pageable->pageable retry
    that could only lose slabs) were both invisible to the suite.

    A pageable arena has nothing to trade: a pageable retry would use the same
    slab count and the same pin=False, so it can only do as well or worse. Keep
    it unconditionally -- the working-set-exceeds-cache-budget WARNING further
    down the loader already reports a shortfall, if there is one. Only a PINNED
    degraded arena has a real trade to weigh (this arena's slightly smaller
    pinned cache vs. a pageable arena's full-but-slower one), so only that case
    calls arena_backing_is_sufficient.
    """
    _pin_word = "pinned" if pinned else "pageable"
    if not pinned:
        return True, (f"keeping it ({_pin_word}) — already pageable, so a "
                       f"pageable retry could not do better")
    _ws_gb = working_set_bytes / 1e9
    if arena_backing_is_sufficient(n_slabs, requested_n_slabs, slab_bytes,
                                    pool_size, working_set_bytes):
        return True, (f"keeping it ({_pin_word}) since its cache share still "
                       f"covers the {_ws_gb:.2f}GB working set")
    return False, (f"below the {_ws_gb:.2f}GB working set, falling back "
                    f"to a pageable arena")


def slab_arena_exhausted_warning(err: Exception) -> str:
    """Operator-facing message for when the pageable retry after a rejected
    degraded arena ALSO fails to allocate: there is no slab arena at all, not
    just a smaller one.

    Split into its own function (rather than an inline print, CLAUDE.md §5)
    so this exact path -- the arena falling through to `None` with nothing
    printed -- is covered by a test that does not need a model load, and a
    future edit that drops the print cannot do so without also breaking that
    assertion. This is the third silent-degradation defect on this branch.
    """
    return (f"[Vaayu-MoE] WARNING: pageable fallback also failed ({err}) — "
            f"there is NO slab arena at all. Every expert streams from disk "
            f"via the clone path; expect very slow decode (a real hardware "
            f"limit on this box, not a bug).")


# Hardware constant: VRAM headroom to leave UNused by the resident expert cache.
# Property of the 16 GB box, NOT the model (§5). Sizes the cache = free_vram - this.
#
# Raised 2 GB -> 3.5 GB after the 2026-07-23 prefill-OOM diagnostic
# (docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic-RESULTS.md). The 2 GB
# value covered PREFILL (proven: prefill held 2.01 GB free across all 62 layers and
# never OOMed) but NOT DECODE: at the prefill->decode boundary the cache re-warms
# toward its full budget while weights (~3.5 GB) + KV (~1.1 GB) are resident, leaving
# no room for the first decode step's logits/sampling transient (dies in
# TopKLogitsWarper) plus allocator fragmentation — OOM at 13.73 GB alloc / 17 MB free
# on the 3rd generation. 3.5 GB shrinks the cache by ~1.5 GB so decode peak drops
# clear of the 15.47 GB usable ceiling. The on-device gate
# (docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix-RESULTS.md) attributed the
# fix to the reserve bump ALONE (4/4, zero OOM with no expandable_segments set);
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is optional fragmentation insurance
# on top, not required for correctness.
#
# RETIRED as a sizing constant 2026-08-09 -- superseded by
# vaayu.vram_plan.VramPlan.reserve_bytes, which is context-aware. The 3.5 GB
# figure documented its own ceiling ("this reserve is context-length-insensitive
# -- the ~1.1 GB KV figure above was measured at MAX_LENGTH=2048"), and bug F4
# was the other half of the same defect.
#
# Kept ONLY as the floor below which no derived reserve may fall: it is the
# measured decode-boundary requirement at short context (2026-07-24 RESULTS,
# 4/4 with no expandable_segments), so a reserve smaller than this is known bad
# regardless of how short the context is.
_VRAM_RESERVE_FLOOR_BYTES = int(3.5 * 1024 ** 3)


# Hardware constant: RAM to leave for the OS + io_uring pools + post-sizing
# host allocations when sizing the prefix-cache pool. Property of the 32 GB
# box, NOT the model. Prefix caching borrows only what's comfortably free.
#
# This used to also reserve for "the expert RAM cache", double-counting it:
# by the time compute_prefix_cache_budget runs, the expert RAM cache AND its
# mirroring pinned slab arena are already allocated (see
# `test_the_two_budgets_leave_room_for_the_prefix_cache`), so they are already
# subtracted out of the `available`/free-RAM figure passed in here. Reserving
# for them again left the prefix cache at 0 on this box (RAM-accounting
# diagnostic RESULTS Q5). What legitimately remains, all allocated AFTER the
# prefix cache measures:
#   ~1.5 GB  OS working set / CUDA context margin (same figure
#            compute_expert_budget uses for its own overhead)
#   ~0.5 GB  io_uring expert buffer pool(s) (get_expert_pool_size() returns
#            1-2 buffers, each ~expert_bytes)
#   ~1.5 GB  safety margin for post-prefix-cache host allocations, notably
#            KV-cache growth and PyTorch host staging during generation
#
# NOTE: the "already allocated" premise holds for the EXL3/arena path (the
# on-device targets), where the slab arena is pinned up-front before this
# runs. On non-EXL3 clone-path layouts the expert cache fills lazily during
# decode, so at sizing time it is NOT yet in `available` and this reserve is
# not headroom-validated for that path — safe today because prefix caching
# defaults OFF and clone paths are gated-in-but-untuned (CLAUDE.md).
_PREFIX_CACHE_RAM_RESERVE = int(3.5 * 1024 ** 3)


def ram_snapshot() -> dict:
    """Categorized process RAM accounting, for the memory-arbitration ledger.

    Reads /proc/self/smaps_rollup — the kernel's pre-aggregated rollup, one small
    file, far cheaper than parsing per-mapping /proc/self/smaps. Fields:
      rss            resident set size
      pss            proportional set size (shared pages divided by sharer count)
      shared_clean   file-backed pages shared and unmodified (e.g. mmapped weights)
      shared_dirty   shared and modified
      private_clean  private and unmodified
      private_dirty  private and modified — anonymous allocations live here, which
                     is where a pinned arena and PyTorch host buffers show up
      swap           pages pushed to swap (pinned memory can never appear here)
    plus `available`/`total` from psutil for the system view.

    Degrades to zeros for smaps-derived fields on a kernel without smaps_rollup;
    never raises, because this is diagnostics and must not break a load.
    """
    import psutil as _ps
    _FIELDS = {
        "Rss": "rss", "Pss": "pss",
        "Shared_Clean": "shared_clean", "Shared_Dirty": "shared_dirty",
        "Private_Clean": "private_clean", "Private_Dirty": "private_dirty",
        "Swap": "swap",
    }
    out = {v: 0 for v in _FIELDS.values()}
    out["available"] = 0
    out["total"] = 0
    try:
        with open("/proc/self/smaps_rollup") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in _FIELDS:
                    out[_FIELDS[key]] = int(rest.strip().split()[0]) * 1024
    except Exception:
        pass   # diagnostics only — a missing/!readable smaps_rollup must not break load
    try:
        _vm = _ps.virtual_memory()
        out["available"] = int(_vm.available)
        out["total"] = int(_vm.total)
    except Exception:
        pass   # diagnostics only — psutil must never break a load either
    return out


def compute_prefix_cache_budget(free_ram_bytes: int) -> int:
    """RAM bytes the cross-turn prefix-cache pool may hold (A1). Derived from
    free system memory minus a hardware headroom reserve. Model-agnostic."""
    return max(0, int(free_ram_bytes) - _PREFIX_CACHE_RAM_RESERVE)


def compute_vram_layer_range(vram_budget_bytes: int, geometry: dict,
                             moe_layer_ids: list) -> "set | None":
    """Fix #7 sizing gap: a single VRAM LRU walked in layer order self-evicts
    and hits 0% whenever budget < one token's working set (see
    2026-07-10-fix7-vram-cache-zero-hit-report.md §4). Restrict caching to the
    FIRST N MoE layers whose per-token insert load (top_k experts each) fits
    the budget — those entries then survive across tokens and hit at the
    cross-token routing-overlap rate. Returns None when the whole working set
    fits (no restriction needed — e.g. MiniMax-M2.5), else the cacheable set.
    Model-agnostic: all inputs derived from _moe_geometry / detected blocks.
    """
    if vram_budget_bytes >= geometry["working_set_bytes"]:
        return None
    per_layer_bytes = geometry["top_k"] * geometry["expert_bytes"]
    n_fit = int(vram_budget_bytes // max(1, per_layer_bytes))
    return set(sorted(moe_layer_ids)[:n_fit])


def moe_supports_slab_arena(expert_styles, is_exl3: bool) -> bool:
    """True iff this model's MoE forward reads expert weights through the
    pinned slab arena on its decode miss path.

    Both the fused-EXL3 forward and the module-list forward reach
    `acquire_slab` / `load_expert_into_slab` / `put_slab` via the shared
    `moe_patch._dispatch_expert_misses` helper, which takes that path whenever
    `expert_cache.arena is not None` (falling back to a clone + `put()` when the
    arena is exhausted). Without an arena both silently clone into PAGEABLE RAM,
    and host-to-device transfer runs at bounce-buffer speed — measured at
    ~7.1 GB/s for MiniMax-M2.7, ~35% of a decode token.

    History worth keeping: the module-list layout was excluded here until
    2026-07-22 because its forward genuinely did clone — it carried its own
    inline serial loader. The Task 4/5 extraction (commits 2f50fea, dee5cdf)
    moved it onto the shared helper and made it slab-capable; this predicate
    lagged that change, so the arena was skipped and the clone path taken.

    Model-agnostic (CLAUDE.md §5): derived from the DETECTED expert layout and
    the loader strategy, never from an architecture name. An unrecognised style
    returns False — a layout whose forward has not been checked must not be
    assumed to consume the arena, since allocating one nothing reads is exactly
    the double-spend this gate exists to prevent. Mixed layouts return False: a
    partially-usable arena is not worth the pinned, unswappable RAM.
    """
    styles = list(expert_styles)
    return (bool(is_exl3) and bool(styles)
            and len(set(styles)) == 1
            and styles[0] in ("fused", "module_list"))


def moe_supports_prefill_stager(expert_styles, is_exl3: bool) -> bool:
    """True iff this model's MoE forward CONSUMES the prefill stager's staged
    layers (`vaayu_ref._prefill_staged`).

    Only `moe_patch._make_fused_sparse_forward_exl3` does: it alone peeks/pops
    `_prefill_staged`, tracks `_staged_consume_count`, and releases
    `_prefill_stage_sem`. The module-list forward passes `staged_layer=None` to
    `_dispatch_expert_misses` and never reads the staged dict, so starting the
    stager for it would read every expert of every layer into memory no consumer
    ever touches — pure wasted I/O and RAM, and it would hold stage slots against
    the TTFT this path just improved.

    Deliberately narrower than `moe_supports_slab_arena`: arena capability and
    staged-layer consumption are independent, and conflating them is what caused
    the module-list arena to be skipped after that layout became slab-capable.
    """
    styles = list(expert_styles)
    return bool(is_exl3) and bool(styles) and all(s == "fused" for s in styles)


def should_use_tied_lm_head(layer_name, lm_head_name, state_dict, load_failed: bool) -> bool:
    """True iff `layer_name`'s weights must come from the embedding instead.

    A checkpoint with `tie_word_embeddings` stores no `lm_head` tensor. Two
    shapes of that reach the loader, and only the first was ever handled:

    - the shard is ABSENT      -> `load_layer` raises      -> `load_failed=True`
    - the shard is EMPTY       -> `load_layer` SUCCEEDS with `{}` (bug D1)

    Vaayu's splitter produces the second: it writes a valid 16-byte `{}`
    safetensors file and its `.done` marker, so the split verifies as complete
    and the load never raises. `lm_head` then keeps its meta-initialized
    weights and the model emits uniform-random logits.

    Only the head can be tied, so every other layer keeps propagating its
    error — an empty decoder layer is a real failure, not something to paper
    over with embeddings. `lm_head_name` may be None when the module tree
    exposes no nn.Linear head (see `detect_layer_names`), which is not a
    fallback case either.
    """
    if lm_head_name is None or layer_name is None:
        return False
    if layer_name.rstrip(".") != lm_head_name.rstrip("."):
        return False
    return bool(load_failed) or not state_dict


def rekey_embed_as_lm_head(state_dict, embed_name: str, lm_head_name: str) -> dict:
    """Re-key an embedding shard's tensors so they land on the tied head.

    Names come from `layer_names_dict`, never assumed to be Llama's (§5).
    Returns {} for an empty input so callers can fail honestly rather than
    inject an unweighted head.
    """
    return {k.replace(embed_name, lm_head_name): v for k, v in state_dict.items()}


def compute_rope_inv_freq(config, device=None) -> "torch.Tensor":
    """RoPE inverse frequencies for `config`, honouring PARTIAL rotary.

    Single source of truth for the three places Vaayu computes RoPE itself:
    the `ROPE_INIT_FUNCTIONS["default"]` shim, `_compute_full_position_embeddings`,
    and the `_init_weights` `compute_default_rope_parameters` shim. All three
    previously duplicated the same full-`head_dim` formula, so a partial-rotary
    model was corrupted identically by whichever one happened to run.

    A model may rotate only a PREFIX of each head: MiniMax-M2 sets
    `rotary_dim: 64` against `head_dim: 128` (its config derives
    `partial_rotary_factor = rotary_dim / head_dim = 0.5`). `apply_rotary_pos_emb`
    infers the rotated width from `cos.shape[-1]`, so emitting a full-width
    cos/sin makes it rotate all 128 dims — including the half meant to pass
    through untouched — at the wrong frequency spacing. Measured effect: layer-0
    attention output ~500x the known-good Qwen baseline, degenerate text, and no
    NaN to make it obvious.

    Matches the transformers convention `dim = int(head_dim * partial_rotary_factor)`.
    Models without the attribute default to 1.0 (full rotary), which is
    byte-identical to the previous behaviour — Qwen3-MoE and Mixtral included.
    """
    head_dim = getattr(config, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = config.hidden_size // config.num_attention_heads
    partial = getattr(config, "partial_rotary_factor", 1.0) or 1.0
    dim = int(head_dim * partial)
    base = float(getattr(config, "rope_theta", 10000.0))
    return 1.0 / (base ** (
        torch.arange(0, dim, 2, dtype=torch.int64).to(
            device=device, dtype=torch.float) / dim
    ))


def act_stats(t) -> str:
    """Compact activation summary for VAAYU_TRACE_ACT=1 diagnostics.

    Forces a device sync (`.item()`), so it is DIAGNOSTIC ONLY and must never be
    called on the default path — every call site is gated on the env flag being
    captured at init. Reports norm/absmax plus explicit NaN/Inf flags, because a
    collapsed (norm -> 0) and an exploded (NaN) hidden state are different bugs
    and the norm alone cannot distinguish them.

    Deliberately print-based: ptrace-based samplers (py-spy) corrupt this
    process's io_uring state — see the io_uring_cqe_overflow_bug finding — so
    env-gated prints are the established instrumentation mechanism here.
    """
    f = t.detach().float()
    return (f"norm={f.norm().item():.4e} absmax={f.abs().max().item():.4e} "
            f"nan={int(torch.isnan(f).any().item())} "
            f"inf={int(torch.isinf(f).any().item())}")


def compute_prefill_pass(seq_len: int, has_speculator: bool, draft_tokens: int) -> bool:
    """True iff a forward pass of seq_len positions is prompt prefill.

    A speculative verify pass is (draft_tokens + 1) positions wide and must be
    treated as decode (its expert loads should be cached). Model-agnostic.
    """
    verify_cap = (draft_tokens + 1) if has_speculator else 1
    return seq_len > verify_cap


def speculative_ignored_sampling(temperature, kwargs) -> list:
    """Names the sampling controls a `speculative_generate` call asked for and
    will NOT get, so the caller can be told instead of guessing from the text.

    That loop is GREEDY at every decision point -- the prefill seed, the
    drafter, and the verifier all argmax -- and it takes `temperature` only to
    swallow it unread, so temperature belongs in this list like the rest.
    `top_p`/`top_k` are named only when sampling was actually requested, since
    HF's generate() ignores them under do_sample=False too and naming them
    there would fire on every greedy request carrying the defaults along.
    `stop_strings` is dropped regardless (this hand-written loop has no such
    machinery); `stopping_criteria` IS honoured per round, so it is never named.
    """
    ignored = []
    if bool(kwargs.get("do_sample")) or (temperature or 0.0) > 0.0:
        if (temperature or 0.0) > 0.0:
            ignored.append("temperature")
        top_p = kwargs.get("top_p")
        if top_p is not None and top_p != 1.0:
            ignored.append("top_p")
        top_k = kwargs.get("top_k")
        if top_k not in (None, 0):
            ignored.append("top_k")
        if not ignored:
            ignored.append("do_sample")
    if kwargs.get("stop_strings"):
        ignored.append("stop_strings")
    return ignored


def compute_stage_gate(seq_len: int, geometry: dict) -> bool:
    """Enable prefill expert staging only when the prompt is long enough that
    routing touches ~all experts per layer (expected coverage >= ~86% at
    seq_len * top_k == 2 * num_experts). Below that, staging every expert
    would read more than routing needs. Model-agnostic."""
    return seq_len * geometry["top_k"] >= 2 * geometry["num_experts"]


def compute_stage_depth(expert_budget_bytes: int, geometry: dict) -> int:
    """How many full layers of experts may be staged ahead during prefill.
    Bounded by the expert-cache budget (which prefill leaves empty — Task 5
    skips prefill puts, so staging borrows that RAM) and hard-capped at 2:
    depth beyond 2 adds RAM risk without TTFT benefit once disk is saturated."""
    layer_bytes = geometry["num_experts"] * geometry["expert_bytes"]
    by_budget = int(expert_budget_bytes // max(1, layer_bytes)) - 1
    return max(1, min(2, by_budget))


def compute_chunk_past_lens(chunk_lens: list, past0: int) -> list:
    """Layer-major chunked prefill: batch element j's KV-past length is the
    initial past plus the lengths of all earlier chunks (they were appended to
    the same per-layer cache before element j runs). Model-agnostic."""
    out, acc = [], int(past0)
    for L in chunk_lens:
        out.append(acc)
        acc += int(L)
    return out


def compute_pos_range(max_seq_len: int, past_len: int, seq_total: int) -> int:
    """Size of the precomputed RoPE cos/sin cache: must cover every position
    that will be sliced during this forward pass (existing past + ALL new
    tokens across every chunk combined, not just max_seq_len or a single
    chunk's length). seq_total is the TOTAL token count this call will
    process -- for layer-major chunked prefill (input_ids as a list of chunk
    tensors) that's the sum of every chunk, not any one chunk's length."""
    return max(int(max_seq_len), int(past_len) + int(seq_total))


def check_batch_kv_invariant(pad_mask, q_len: int, past_key_values) -> None:
    """Raise if the batched padding mask disagrees with the cache it will be
    used against.

    build_batch_attention_mask derives its KV axis from pad_mask.shape[1]
    alone; it never inspects the cache. Attention then expands that mask to
    the real key length, so any disagreement dies deep inside a decoder layer
    as a bare torch expand error naming head counts and no cause -- the same
    diagnostic-at-the-wrong-frame shape as bug D3. One 2026-08-09 batched run
    failed exactly this way (mask 1127 wide against a 1690-long KV) and was
    never explained, because nothing recorded which side was wrong.

    Both halves matter:
      - total length: pad_mask width must equal past + this call's queries.
      - per-layer agreement: get_seq_length() reads layer 0 ONLY, so a cache
        whose layers hold different lengths satisfies every length check in
        forward() and still mismatches at whichever layer differs. Checked
        only for caches that expose _seq_len (Vaayu's own classes); a plain
        transformers DynamicCache has no per-layer view to check.
    """
    past = past_key_values.get_seq_length() if past_key_values is not None else 0
    lens = getattr(past_key_values, "_seq_len", None)
    if lens and min(lens) != max(lens):
        raise RuntimeError(
            f"batched KV cache has per-layer lengths that disagree: "
            f"min={min(lens)} max={max(lens)} over {len(lens)} layers. The "
            f"attention mask can only be built for one length (layer 0's "
            f"{past}), so every layer that differs would mismatch inside "
            f"attention. A partially-completed forward (an exception in the "
            f"layer loop) is the way a cache reaches this state.")
    width = int(pad_mask.shape[1])
    if width != past + int(q_len):
        raise RuntimeError(
            f"batched attention mask is {width} wide but the KV it will be "
            f"used against will be {past + int(q_len)} long "
            f"(past={past} + q_len={int(q_len)}). The mask is built from the "
            f"caller's padding mask, so the caller's idea of how many cached "
            f"tokens precede this call is wrong by {past + int(q_len) - width}.")


def compute_warmup_ranges(offset_index, checkpoint_path, warm_bytes: int):
    """Byte ranges to pre-read to prime the NVMe controller's DRAM cache before
    generation (Task 7). Walks the offset index in order, emitting one
    (shard_path, file_offset, length) per expert tensor, until the accumulated
    length reaches warm_bytes. Model-agnostic: derived entirely from the index;
    warm_bytes is a hardware constant chosen by the caller."""
    ranges = []
    total = 0
    for layer_name, experts in offset_index.index.items():
        shard = f"{checkpoint_path}/{layer_name}.safetensors"
        for _eidx, metas in experts.items():
            for _key, meta in metas.items():
                ranges.append((shard, meta.file_offset, meta.byte_length))
                total += meta.byte_length
                if total >= warm_bytes:
                    return ranges
    return ranges


def check_lookahead_flag_safe() -> None:
    """Hard-block guard for VAAYU_LOOKAHEAD=1. **The reason changed on
    2026-08-30: the corruption is FIXED. The block now stands on performance.**

    History, because the previous version of this message sent readers to the
    wrong subsystem for six weeks. It claimed the cause was "worker-vs-worker
    races inside the multi-worker _stager_executor" and asked for a stager-pool
    concurrency audit. That was wrong. The bug was stager-vs-MAIN-thread, in
    ExpertRAMCache.put_slab's DECLINED-INSERT path: on a duplicate key or when
    no room could be freed, it called arena.release(slab_idx) and returned
    normally, while the caller was still holding views into that slab and about
    to DMA from them (moe_patch._load_one returns exactly those views to the
    forward). The next acquire() handed the live buffer out and io_uring
    overwrote it mid-copy -> torn expert weights -> NaN logits -> multinomial's
    "probability tensor contains inf, nan" assert. record_read()'s event guard
    could not help: the slab was freed BEFORE any read event existed. Both
    earlier fix rounds hardened the evict->reuse window; this was a different
    one, which is why neither found it. Fixed by making put_slab report
    ownership and never release a declined slab -- see put_slab() and
    ExpertRAMCache.orphan_slab() in vaayu/persist/expert_cache.py, and
    plans/2026-08-30-R2-structural-probe-RESULTS.md for the measurement trail.

    **Status 2026-08-30 (second revision, same day): the flag is now a WIN and
    this block is stale.** When the corruption was first fixed, the flag ran
    clean but read 60% more bytes and cost 15% of decode, because the stager and
    the decode miss path both read the same expert and one of the two threw its
    copy away. An in-flight load reservation (ExpertRAMCache.claim_load and
    friends) removed 6796 duplicate expert reads per 64 tokens, and the flag
    inverted from a regression to a gain:

        lookahead OFF                1.277 tok/s   (3 arms, 1.267-1.284)
        lookahead ON, no reservation 1.070 tok/s   (-15%)
        lookahead ON, reservation    1.422 tok/s   (+11.4%, 3 arms, 1.420-1.425)

    Interleaved same-box same-day arms; the two distributions do not overlap
    (worst ON beats best OFF by 10.6%). Coverage 62.6% -> 82.8%, NVMe bandwidth
    3.786 -> 5.009 GB/s, bulk bytes still +17.8% over baseline (the residual is
    eviction churn, not duplicate reads). Soaked at 1024 tokens with the stager
    at 4x width: no crashes, and greedy token ids byte-identical to a
    lookahead-off run across 256 tokens.

    **Status 2026-09-03 (P2): both coverage gaps are now CLOSED BY MEASUREMENT,
    and the flag is refuted as a default. The guard stays on a new reason.**
    The two grounds it used to stand on were "the predictor is wired ONLY into
    _make_fused_sparse_forward_exl3, so the flag is a silent no-op on the
    module-list forward (MiniMax)" and "everything is single-stream". Neither is
    true any more:

      - The module-list forward IS wired (moe_patch._lookahead_step, one
        implementation called by both forwards) and the predictor demonstrably
        fires on MiniMax: 76.6% accuracy, 13 000 prefetch inserts per 64-token
        arm, on-path misses 28 484 -> 24 098 (-15.4%), greedy ids byte-identical.
      - Fan-out was measured on both models (P1 for Qwen, P2 for MiniMax).

    **What the measurement says, same box, same hour, 64 tokens greedy,
    BALLAST_GB=25, ring pool 3 (post-P8):**

        MiniMax N=1  OFF 1.798/1.792/1.808   ON 1.804/1.815/1.809   +0.6% (OVERLAPS)
        MiniMax N=3  OFF 2.784/2.775         ON 2.639/2.607         -5.7% (no overlap)
        Qwen    N=1  OFF 1.083/1.122         ON 1.219/1.220        +10.6% (no overlap)

    MiniMax predicts worse (0.766 vs Qwen's 0.902) and wastes more of what it
    stages (28% vs 12%), so it pays +18.0% bytes/token at N=1 and +26.4% at N=3
    to remove 15% of its misses -- and the trade nets out negative under fan-out.
    Defaulting the flag ON would ship +10.6% to one actively-tested model and
    -5.7% to the other at the fan-out this engine exists for: the asymmetry
    CLAUDE.md section 5 forbids, now measured rather than assumed.

    **To lift it:** fix the byte residual first (row P4, prefetch admission that
    cannot evict the on-path working set) and re-run P2's arms. The flag is a
    default only if MiniMax stops LOSING at N>1; it is not gated on wiring any
    more. Full trail: plans/2026-09-03-P2-modulelist-lookahead-RESULTS.md.

    Called from __init__ before any MoE cache/executor/prefetch object is
    created, so a set flag fails fast at engine construction. No-op (dead code)
    when the flag is unset -- the default path is unaffected.
    """
    if os.environ.get("VAAYU_LOOKAHEAD") == "1":
        raise RuntimeError(
            "VAAYU_LOOKAHEAD=1 is disabled pending model coverage, NOT because "
            "it is unsafe or slow -- both of those were true earlier on 2026-08-30 "
            "and neither is now. The memory corruption was root-caused and fixed "
            "(ExpertRAMCache.put_slab released a declined slab the caller was "
            "still reading), and an in-flight load reservation then turned the "
            "flag's 15% regression into a measured +11.4% on Qwen3-235B (1.277 "
            "-> 1.422 tok/s, 3 interleaved arms each, non-overlapping; 1024-token "
            "soak clean, greedy ids byte-identical). It stays off because the win "
            "is Qwen's, not the flag's: the module-list forward IS wired now "
            "(P2, 2026-09-03) and MiniMax measures +0.6% at N=1 (overlapping arms) "
            "and -5.7% at N=3, against Qwen's +10.6% at N=1 on the same box the "
            "same hour -- it costs MiniMax +18-26% bytes/token to remove 15% of "
            "its misses. Fix the byte residual (row P4) before revisiting. See "
            "check_lookahead_flag_safe.__doc__. Unset VAAYU_LOOKAHEAD (or set it "
            "to anything other than \"1\") to proceed."
        )


class FP8DynamicCache(DynamicCache):
    """
    DynamicCache variant that stores K/V tensors in torch.float8_e4m3fn.

    Per-head per-token scales (shape [B, H, S, 1]) are stored alongside
    the FP8 data so decode steps are O(1): only new tokens are quantized.
    Full cache is dequantized to BF16/FP16 before returning to attention.

    Note: this class manages its own key_cache / value_cache lists directly
    rather than delegating to the parent's layer-based storage, so that FP8
    tensors (and their companion scale tensors) are stored contiguously and
    can be concatenated cheaply across decode steps.

    With quantize=False the fp8 store, the scales and the dequantized mirror are
    all skipped: K/V are held directly in their source dtype (bf16/fp16) and
    update() returns slices of them. Measured 2.0 vs 3.016 bytes/element -- the
    fp8 copy buys nothing once a full-precision copy has to exist for attention
    anyway, and bf16 storage is lossless where fp8 is not. What this class
    actually provides is the ring buffer (amortized O(1) append), which is
    mode-independent.
    """

    def __init__(self, quantize: bool = True):
        super().__init__()
        self.quantize = bool(quantize)
        # Capacity buffers (allocated lazily on first update per layer). Each holds
        # >= _seq_len[layer] valid tokens on the dim=-2 (sequence) axis; the tail is
        # unused capacity. Fix #8: replaces per-token torch.cat (O(seq^2)) with
        # in-place writes + geometric doubling (amortized O(1) append).
        self.key_cache: list = []       # fp8 K, shape [B, H, capacity, D]
        self.value_cache: list = []     # fp8 V
        # The four buffer lists below stay empty when quantize=False.
        self._key_scales: list = []     # [B, H, capacity, 1]
        self._value_scales: list = []
        self._k_out_cache: list = []    # dequantized K, [B, H, capacity, D]
        self._v_out_cache: list = []
        self._seq_len: list = []        # per-layer used length

    _GROWTH = 2  # capacity doubling factor (amortized O(1) append)

    def _ensure_capacity(self, layer_idx, needed, template_kv, template_scale):
        """Grow layer_idx's buffers so capacity >= needed. Copies existing valid
        tokens once on growth. template_* provide dtype/device/leading dims."""
        cap = self.key_cache[layer_idx].shape[-2] if layer_idx < len(self.key_cache) else 0
        if layer_idx < len(self.key_cache) and cap >= needed:
            return
        new_cap = max(needed, max(1, cap) * self._GROWTH)
        B, H, _, D = template_kv.shape
        dev, kv_dtype = template_kv.device, template_kv.dtype
        # unquantized: template_kv IS the source K, so its dtype is the storage
        # dtype and there is no separate scale dtype.
        out_dtype = kv_dtype if template_scale is None else template_scale.dtype

        def _grow(old_list, last_dim, dtype):
            buf = torch.zeros((B, H, new_cap, last_dim), dtype=dtype, device=dev)
            if layer_idx < len(old_list):
                used = self._seq_len[layer_idx]
                buf[..., :used, :] = old_list[layer_idx][..., :used, :]
                old_list[layer_idx] = buf
            else:
                old_list.append(buf)

        _grow(self.key_cache,     D, kv_dtype)
        _grow(self.value_cache,   D, kv_dtype)
        if self.quantize:
            _grow(self._key_scales,   1, template_scale.dtype)
            _grow(self._value_scales, 1, template_scale.dtype)
            _grow(self._k_out_cache,  D, out_dtype)
            _grow(self._v_out_cache,  D, out_dtype)
        if layer_idx >= len(self._seq_len):
            self._seq_len.append(0)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if not self.quantize:
            n_new = key_states.shape[-2]
            prev = self._seq_len[layer_idx] if layer_idx < len(self._seq_len) else 0
            self._ensure_capacity(layer_idx, prev + n_new, key_states, None)
            s = slice(prev, prev + n_new)
            self.key_cache[layer_idx][..., s, :] = key_states
            self.value_cache[layer_idx][..., s, :] = value_states
            self._seq_len[layer_idx] = prev + n_new
            used = self._seq_len[layer_idx]
            return (self.key_cache[layer_idx][..., :used, :],
                    self.value_cache[layer_idx][..., :used, :])

        k_scale = key_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        v_scale = value_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
        k_fp8 = (key_states / k_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        v_fp8 = (value_states / v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        k_new_out = k_fp8.to(key_states.dtype) * k_scale
        v_new_out = v_fp8.to(value_states.dtype) * v_scale

        n_new = key_states.shape[-2]
        prev = self._seq_len[layer_idx] if layer_idx < len(self._seq_len) else 0
        self._ensure_capacity(layer_idx, prev + n_new, k_fp8, k_scale)

        s = slice(prev, prev + n_new)
        self.key_cache[layer_idx][..., s, :] = k_fp8
        self.value_cache[layer_idx][..., s, :] = v_fp8
        self._key_scales[layer_idx][..., s, :] = k_scale
        self._value_scales[layer_idx][..., s, :] = v_scale
        self._k_out_cache[layer_idx][..., s, :] = k_new_out
        self._v_out_cache[layer_idx][..., s, :] = v_new_out
        self._seq_len[layer_idx] = prev + n_new

        used = self._seq_len[layer_idx]
        return (self._k_out_cache[layer_idx][..., :used, :],
                self._v_out_cache[layer_idx][..., :used, :])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._seq_len):
            return 0
        return self._seq_len[layer_idx]

    def crop(self, max_length: int) -> None:
        """Truncate every layer to max_length tokens (Fix #8: length-only, no copy).
        Ring buffer retains capacity; the leading [0:max_length) tokens are already
        the valid data, so trimming the tail is just shrinking the length."""
        for i in range(len(self._seq_len)):
            self._seq_len[i] = min(self._seq_len[i], max_length)

    def crop_front(self, keep_len: int) -> None:
        """Drop the oldest tokens, keeping only the newest keep_len (Fix #8:
        OOM-pressure path only, so a single copy-shift per layer is acceptable).
        Ring buffer's valid data starts at index 0, so removing the front
        requires shifting the retained [used-keep_len:used) window down to
        [0:keep_len) before shrinking the length."""
        for i in range(len(self._seq_len)):
            used = self._seq_len[i]
            if keep_len >= used:
                continue
            start = used - keep_len
            for name in self._BUFFER_LISTS:
                buf = getattr(self, name)
                buf[i][..., :keep_len, :] = buf[i][..., start:used, :].clone()
            self._seq_len[i] = keep_len

    @property
    def _BUFFER_LISTS(self):
        """Only the buffer lists this instance actually allocates: unquantized
        mode has no scales and no dequantized mirror."""
        if self.quantize:
            return ("key_cache", "value_cache", "_key_scales",
                    "_value_scales", "_k_out_cache", "_v_out_cache")
        return ("key_cache", "value_cache")

    def to_cpu(self) -> None:
        """Compact each layer to its used length and move all buffers to CPU
        (pinned when CUDA is available), for RAM-resident prefix-cache storage
        between turns. Drops the geometric-growth capacity tail."""
        pin = torch.cuda.is_available()
        for name in self._BUFFER_LISTS:
            lst = getattr(self, name)
            for i in range(len(lst)):
                used = self._seq_len[i]
                compact = lst[i][..., :used, :].contiguous().to("cpu")
                if pin:
                    compact = compact.pin_memory()
                lst[i] = compact

    def to_device(self, device) -> None:
        """Move all buffers to `device` (GPU) for an in-flight request."""
        for name in self._BUFFER_LISTS:
            lst = getattr(self, name)
            for i in range(len(lst)):
                lst[i] = lst[i].to(device, non_blocking=True)

    def resident_bytes(self) -> int:
        """Total byte footprint of all resident buffers (for budget sizing)."""
        return sum(t.nbytes for name in self._BUFFER_LISTS
                   for t in getattr(self, name))


class TurboQuantKVCache(FP8DynamicCache):
    """KV cache storing TurboQuant codes instead of values (arXiv 2504.19874).

    Per (token, head) it holds bit-packed codebook indices, bit-packed QJL sign
    bits and two fp16 norms -- 0.531 bytes/element at bits_total=4 and
    head_dim=128, against bf16's 2.000. Reconstruction happens on READ, inside
    update(); nothing full-size is ever retained, because a persistent
    reconstructed mirror is exactly what Plan E deleted from the fp8 mode.

    key_cache / value_cache keep their names so inherited code paths work, but
    they hold INDICES, NOT VALUES. Any consumer reading them directly must
    special-case this class -- see kv_extract._extract_kv_as_bf16 and
    kv_persist.DiskWriter.enqueue_append.

    Lossy in a way bf16 is not: measured relative L2 error ~0.23 on random K/V
    at bits_total=4.
    """

    def __init__(self, bits_total: int = 4, rot_seed: int = 0x5EED,
                 qjl_seed: int = 0x9317):
        # quantize=False: there are no fp8 scales. The flag alone does NOT make
        # the persistence path safe (both of its values mis-describe this
        # layout); explicit type checks do.
        super().__init__(quantize=False)
        if bits_total < 2:
            raise ValueError(f"bits_total must be >= 2 (1 bit goes to QJL), got {bits_total}")
        self.bits_total = int(bits_total)
        self._b_mse = self.bits_total - 1
        self.rot_seed = rot_seed
        self.qjl_seed = qjl_seed
        self._k_signs: list = []    # packed QJL signs, [B, H, capacity, D/8]
        self._v_signs: list = []
        self._k_norms: list = []    # (||x||, gamma) fp16, [B, H, capacity, 2]
        self._v_norms: list = []
        self._params: dict = {}     # (dim, device) -> (rot, gauss, codebook)
        self._head_dims: dict = {}  # layer_idx -> (k head_dim, v head_dim)

    @property
    def _BUFFER_LISTS(self):
        return ("key_cache", "value_cache", "_k_signs", "_v_signs",
                "_k_norms", "_v_norms")

    def _get_params(self, dim: int, device):
        """Rotation, QJL sketch and codebook for this head_dim. Built once from
        fixed seeds (data-oblivious) and shared across every layer and head."""
        key = (dim, str(device))
        params = self._params.get(key)
        if params is None:
            params = (turboquant.make_rotation(dim, self.rot_seed, device),
                      turboquant.make_gaussian(dim, self.qjl_seed, device),
                      turboquant.make_codebook(self._b_mse, dim, device))
            self._params[key] = params
        return params

    def _quantize(self, x):
        rot, gauss, cb = self._get_params(x.shape[-1], x.device)
        return turboquant.quantize(x, rot, gauss, cb, self._b_mse)

    def _reconstruct(self, packed, signs, norms, dim, dtype):
        rot, gauss, cb = self._get_params(dim, packed.device)
        return turboquant.reconstruct(packed, signs, norms, rot, gauss, cb,
                                      self._b_mse, dim, out_dtype=dtype)

    def _ensure_capacity_packed(self, layer_idx, needed, tensors):
        """Grow layer_idx's six buffers so capacity >= needed, reusing the
        parent's geometric-doubling policy. Each list has its own last dim and
        dtype, so the templates come from the freshly packed tensors."""
        cap = self.key_cache[layer_idx].shape[-2] if layer_idx < len(self.key_cache) else 0
        if layer_idx < len(self.key_cache) and cap >= needed:
            return
        new_cap = max(needed, max(1, cap) * self._GROWTH)
        for name, t in zip(self._BUFFER_LISTS, tensors):
            lst = getattr(self, name)
            B, H, _, D = t.shape
            buf = torch.zeros((B, H, new_cap, D), dtype=t.dtype, device=t.device)
            if layer_idx < len(lst):
                used = self._seq_len[layer_idx]
                buf[..., :used, :] = lst[layer_idx][..., :used, :]
                lst[layer_idx] = buf
            else:
                lst.append(buf)
        if layer_idx >= len(self._seq_len):
            self._seq_len.append(0)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k_idx, k_sign, k_norm = self._quantize(key_states)
        v_idx, v_sign, v_norm = self._quantize(value_states)
        tensors = (k_idx, v_idx, k_sign, v_sign, k_norm, v_norm)

        n_new = key_states.shape[-2]
        prev = self._seq_len[layer_idx] if layer_idx < len(self._seq_len) else 0
        self._ensure_capacity_packed(layer_idx, prev + n_new, tensors)

        s = slice(prev, prev + n_new)
        for name, t in zip(self._BUFFER_LISTS, tensors):
            getattr(self, name)[layer_idx][..., s, :] = t
        self._seq_len[layer_idx] = prev + n_new

        self._head_dims[layer_idx] = (key_states.shape[-1], value_states.shape[-1])
        return self.reconstruct_layer(layer_idx, dtype=key_states.dtype)

    def reconstruct_layer(self, layer_idx: int, dtype=torch.bfloat16):
        """Full-length (K, V) for one layer, reconstructed to `dtype`.

        The ONLY supported way to read this cache's contents: key_cache and
        value_cache hold packed indices, so anything that slices them directly
        gets indices interpreted as values.
        """
        k_dim, v_dim = self._head_dims[layer_idx]
        w = slice(0, self._seq_len[layer_idx])
        return (self._reconstruct(self.key_cache[layer_idx][..., w, :],
                                  self._k_signs[layer_idx][..., w, :],
                                  self._k_norms[layer_idx][..., w, :],
                                  k_dim, dtype),
                self._reconstruct(self.value_cache[layer_idx][..., w, :],
                                  self._v_signs[layer_idx][..., w, :],
                                  self._v_norms[layer_idx][..., w, :],
                                  v_dim, dtype))


def _layer_path_exists(model: nn.Module, path: str) -> bool:
    if not path:
        return False
    obj = model
    for part in path.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def _auto_detect_layer_names(model: nn.Module) -> dict:
    EMBED = {"embed_tokens", "word_embeddings", "wte", "embed", "token_embeddings"}
    NORM = {"norm", "final_layernorm", "ln_f", "final_norm", "layer_norm", "ln_out"}
    HEAD = {"lm_head", "output_layer", "embed_out", "output", "head"}

    best_path, best_len = "", 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
            best_len, best_path = len(mod), name

    if not best_path:
        return {}

    embed_path = norm_path = lm_head_path = None
    for name, mod in model.named_modules():
        if name.startswith(best_path + "."):
            continue
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if embed_path is None and leaf in EMBED and isinstance(mod, nn.Embedding):
            embed_path = name
        if norm_path is None and leaf in NORM and not isinstance(mod, (nn.Embedding, nn.Linear, nn.ModuleList)):
            norm_path = name
        if lm_head_path is None and leaf in HEAD and isinstance(mod, nn.Linear):
            lm_head_path = name

    return {
        "embed": embed_path,
        "layer_prefix": best_path,
        "norm": norm_path,
        "lm_head": lm_head_path,
    }


def stage_read_expert(shard_file: str, metas: dict, arena):
    """Prefill stager read: into a borrowed pinned arena slab when one is free
    (zero-clone, async-capable H2D later), else pageable load_expert_sparse
    fallback. Returns (views_dict, slab_idx_or_None). Caller owns the slab and
    must release it exactly once (consumer after H2D completes, or the stager/
    forward cleanup paths for never-consumed layers)."""
    from .persist.safetensor_model_persister import load_expert_sparse, load_expert_into_slab
    if arena is not None:
        sidx = arena.acquire()
        if sidx is not None:
            slab = arena.tensor(sidx)
            try:
                # tensor_align=False (B1): stage RAW — zero CPU copy, exactly
                # like decode's cache-miss path. The 16-byte alignment EXL3's
                # uint4* kernel needs is now applied on the GPU by
                # gpu_repack_aligned in the staged-consume path (moe_patch.py),
                # off the CPU memory bus the O_DIRECT DMA contends for. This
                # removes ~142 GB of per-prefill CPU copy_ (+285 GB bus traffic).
                views = load_expert_into_slab(shard_file, metas, slab,
                                              allow_direct=arena.pin,
                                              tensor_align=False)
            except Exception:
                arena.release(sidx)
                raise
            return views, sidx
    return load_expert_sparse(shard_file, metas), None


_pcie_topology_checked: set = set()

# HARDWARE CONSTANT: the worst pinned H2D bandwidth this box tolerates before a
# slot/riser/BIOS problem is the likely cause. ~8 GB/s sits below the Gen3 x16 /
# Gen4 x8 class. A property of the machine and its PCIe wiring, not of any model.
#
# The health test is MEASURED bandwidth, deliberately NOT link topology: idle
# GPUs downtrain to Gen1 for power save, and native-x8 cards (e.g. RTX 5060 Ti)
# can never report x16, so a gen/width rule raises false alarms on healthy
# hardware. This box measures ~20 GB/s while nvidia-smi reads "Gen1 x8".
PCIE_MIN_HEALTHY_GBPS = 8.0


def pcie_link_label(returncode: int, stdout: str) -> str:
    """Human-readable link state from nvidia-smi's gen,width row, or "unknown".

    The query asks for `pcie.link.gen.current,pcie.link.width.current` in that
    order, so the FIRST field is the generation. Never raises: this is a
    diagnostic and must not become the failure it exists to report.
    """
    if returncode != 0:
        return "unknown"
    try:
        gen, width = stdout.strip().splitlines()[0].split(", ")
        return f"Gen{gen} x{width}"
    except Exception:
        return "unknown"


def pcie_health_message(h2d_gbps: float, link: str, model_size_gb: float) -> str:
    """The line(s) the PCIe probe prints. Pure, so the decision is testable
    without a GPU or a subprocess."""
    if h2d_gbps < PCIE_MIN_HEALTHY_GBPS:
        return (f"⚠️  [Vaayu-Warning] Measured pinned H2D only {h2d_gbps:.1f} GB/s "
                f"(link under load: {link}).\n"
                f"   Streaming {model_size_gb:.1f}GB models will be PCIe-bound. "
                f"Check slot/riser/BIOS PCIe generation settings.")
    return (f"✅ [Vaayu] PCIe measured: {h2d_gbps:.1f} GB/s pinned H2D "
            f"(link under load: {link})")


def _check_pcie_topology(gpu_index: int = 0, model_size_gb: float = 132.0):
    """Measure real H2D bandwidth with a pinned burst copy, then report the
    uptrained link state. Idle GPUs downtrain to Gen1 (power save), so probing
    link registers without load produces false warnings; and native-x8 cards
    (e.g. RTX 5060 Ti) can never show x16 — width alone is not a defect signal.
    """
    if gpu_index in _pcie_topology_checked:
        return
    _pcie_topology_checked.add(gpu_index)
    if not torch.cuda.is_available():
        return
    try:
        # Burst copy: forces the link to uptrain and gives a measured number.
        _src = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
        _dst = torch.empty(256 * 1024 * 1024, dtype=torch.uint8,
                           device=f"cuda:{gpu_index}")
        _dst.copy_(_src)
        torch.cuda.synchronize(gpu_index)
        _t0 = time.perf_counter()
        for _ in range(3):
            _dst.copy_(_src, non_blocking=True)
        torch.cuda.synchronize(gpu_index)
        _h2d_gbps = 3 * _src.nbytes / 1e9 / (time.perf_counter() - _t0)
        del _src, _dst

        result = subprocess.run(
            ["nvidia-smi", f"-i={gpu_index}",
             "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        _link = pcie_link_label(result.returncode, result.stdout)
        print(pcie_health_message(_h2d_gbps, _link, model_size_gb))
    except Exception:
        pass


class VaayuBaseModel(GenerationMixin):

    # customize layer names here; subclasses or layer_names_dict kwarg override
    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }
        self.layer_names_dict.update(self._layer_names_override)



    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=None,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False, kv_cache_mode="auto", max_pinned_layers=0,
                 layer_names_dict=None, tokenizer_class=None, prefill_chunk_size=512,
                 drafter_model_path=None, draft_tokens=5, use_prompt_lookup=False,
                 attn_implementation=None, enable_prefix_cache=False,
                 prefix_compress=None,
                 auto_prefix_match=True, enable_kv_persist=False,
                 kv_persist_dir=None, kv_persist_sync=False,
                 enable_expert_usage=False, ram_cache_budget=None,
                 prefix_sessions=3):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM.

        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        max_seq_len : int, optional
            max seq length, by default None -- derives the largest context that
            fits in free VRAM (vaayu.vram_plan.plan_vram). A supplied value is
            validated against that derived ceiling and raises if it does not fit.
        layer_shards_saving_path : str, optional
            optional path to save layered shards model file, by default just save to the local cache of model, subdir named splitted_model will be saved
        profiling_mode : book, optional
            if to profile the model loading time, default to False
        compression: str, optinal
            setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
        hf_token: str, optional
            huggingface api token could be provided, by default None
        """


        # Diagnostic only (VAAYU_TRACE_ACT=1): per-layer activation norms during
        # prefill + the first decode step, to localize where hidden states go
        # wrong. Captured once here; every call site reads this, never os.environ
        # live, so the default path is unaffected. Off => zero cost.
        self._trace_act = os.environ.get("VAAYU_TRACE_ACT") == "1"
        # VAAYU_TRACE_EXPERT=<layer_idx>: per-expert magnitudes for ONE MoE
        # layer (the checks sync per expert, so model-wide is far too costly).
        _te = os.environ.get("VAAYU_TRACE_EXPERT")
        self._trace_expert_layer = int(_te) if _te and _te.lstrip("-").isdigit() else None

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()
        self.telemetry = VaayuTelemetry()
        self._token_count = 0

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self._supports_cache_class = True
        self.hf_quantizer = None

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. To use compression, please install bitsandbytes: `pip install bitsandbytes`')
            if not device.startswith("cuda"):
                raise ValueError(
                    "Compression (4bit/8bit) currently requires a CUDA GPU. "
                    "For CPU-only mode, use compression=None. "
                    "CPU compression support is planned for a future release."
                )


        self.compression = compression
        self.hf_token = hf_token
        self._forced_attn_impl = attn_implementation
        self.enable_prefix_cache = enable_prefix_cache
        # None = pooled KV stays bf16 (default). An int = TurboQuant-pack pooled
        # entries at that many bits/channel; 9 is the lowest width that passed
        # the quality gate on BOTH on-device models. Costs one pack per turn,
        # nothing per token.
        self.prefix_compress = prefix_compress
        self.auto_prefix_match = auto_prefix_match
        # Concurrent agent sessions the prefix pool must hold before the
        # expert cache may claim the rest. 3 is a policy default for an
        # agent with a few parallel streams, not a hardware figure.
        self.prefix_sessions = prefix_sessions
        self.enable_kv_persist = enable_kv_persist
        self.kv_persist_dir = kv_persist_dir
        self.kv_persist_sync = kv_persist_sync
        self.enable_expert_usage = enable_expert_usage

        # Save parameters
        self._layer_names_override = layer_names_dict or {}
        self._tokenizer_class = tokenizer_class
        self.set_layer_names_dict()


        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(model_local_path_or_repo_id,
                                                                                         layer_shards_saving_path,
                                                                                         compression=compression,
                                                                                         layer_names=self.layer_names_dict,
                                                                                         hf_token=hf_token,
                                                                                         delete_original=delete_original)
        self.running_device = device
        self.device = torch.device(self.running_device)
        if dtype == torch.float16 and device.startswith("cuda"):
            try:
                cc = torch.cuda.get_device_capability()
                if cc[0] >= 8:
                    self.running_dtype = torch.bfloat16
                    print(f"✅ GPU supports BF16 (compute capability {cc[0]}.{cc[1]}). Using bfloat16 for better stability.")
                else:
                    self.running_dtype = dtype
                    print(f"ℹ️ GPU compute capability {cc[0]}.{cc[1]} — using float16.")
            except Exception:
                self.running_dtype = dtype
        elif not device.startswith("cuda") and dtype == torch.float16:
            # CPU lacks native float16 ALUs — float32 is 10-50x faster on x86/ARM
            self.running_dtype = torch.float32
            print("ℹ️ CPU mode: using float32 (float16 is very slow on CPU).")
        else:
            self.running_dtype = dtype
        self.dtype = self.running_dtype

        if device.startswith("cuda"):
            gpu_index = int(device.split(":")[-1]) if ":" in device else 0
            try:
                model_size_bytes = sum(
                    os.path.getsize(p) for p in Path(self.checkpoint_path).glob("*.safetensors")
                )
                model_size_gb = model_size_bytes / (1024 ** 3)
                if model_size_gb == 0:
                    model_size_gb = 132.0
            except Exception:
                model_size_gb = 132.0
            _check_pcie_topology(gpu_index, model_size_gb)

        self.use_torch_compile = use_torch_compile
        self.kv_cache_mode = kv_cache_mode
        self.max_pinned_layers = max_pinned_layers
        self.prefill_chunk_size = prefill_chunk_size
        self._compiled_layer_fn = None
        if use_torch_compile:
            if not hasattr(torch, 'compile'):
                print("⚠️ torch.compile requires PyTorch 2.0+. Disabling.")
                self.use_torch_compile = False
            elif not device.startswith("cuda"):
                print("⚠️ torch.compile is most effective on CUDA. Disabling for CPU.")
                self.use_torch_compile = False
            else:
                print("✅ torch.compile enabled for layer forward passes")

        # Create model
        if hf_token is not None:
            self.config = AutoConfig.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)

        self.generation_config = self.get_generation_config()
        #print(f"using generation_config: {self.generation_config}")

        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        # Probe exact physical vocab/hidden dimensions from extracted shard metadata to prevent PyTorch embedding mismatches
        embed_path = Path(self.checkpoint_path) / f"{self.layer_names_dict['embed']}.safetensors"
        if os.path.exists(embed_path):
            import safetensors.torch
            try:
                with safetensors.torch.safe_open(embed_path, framework="pt", device="cpu") as f:
                    real_embed_key = f"{self.layer_names_dict['embed']}.weight"
                    if real_embed_key in f.keys():
                        real_shape = f.get_slice(real_embed_key).get_shape()
                        self.config.vocab_size = real_shape[0]
                        self.config.hidden_size = real_shape[1]
            except Exception:
                pass

        # Native fallback if probe failed
        if not hasattr(self.config, "vocab_size"):
            if hasattr(self.tokenizer, "vocab_size"):
                self.config.vocab_size = self.tokenizer.vocab_size
            else:
                self.config.vocab_size = len(self.tokenizer)

        # Pull everything from nested configurations if this is a VLM wrapper
        if hasattr(self.config, "text_config"):
            for k in dir(self.config.text_config):
                if not k.startswith("_") and not hasattr(self.config, k):
                    setattr(self.config, k, getattr(self.config.text_config, k))

        # Dynamically map custom aliases
        fallback_mappings = {
            "hidden_size": ["d_model", "n_embd", "n_emb"],
            "num_hidden_layers": ["num_layers", "n_layer", "n_layers"],
            "num_attention_heads": ["num_heads", "n_head", "n_heads"],
            "num_key_value_heads": ["num_kv_heads", "n_kv_head", "num_attention_heads"],
            "intermediate_size": ["ffn_dim", "inner_dim"],
            "rms_norm_eps": ["layer_norm_eps", "norm_eps"],
            "max_position_embeddings": ["n_positions", "max_seq_len", "seq_length"],
            "rope_theta": ["rotary_emb_base"],
            "hidden_act": ["activation_function"],
        }

        for target_key, fallbacks in fallback_mappings.items():
            if not hasattr(self.config, target_key):
                for fb in fallbacks:
                    if hasattr(self.config, fb):
                        setattr(self.config, target_key, getattr(self.config, fb))
                        break

        # Token bounds protection
        if not hasattr(self.config, "pad_token_id"):
            self.config.pad_token_id = None
        if getattr(self.config, "pad_token_id", None) is not None and getattr(self.config, "vocab_size", 0) <= self.config.pad_token_id:
            self.config.pad_token_id = None

        if not hasattr(self.config, "eos_token_id"):
            self.config.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        if not hasattr(self.config, "bos_token_id"):
            self.config.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1

        # Hard generic defaults to blindly satisfy transformers instantiation for totally blank models
        if not hasattr(self.config, "num_hidden_layers"): self.config.num_hidden_layers = 24
        if not hasattr(self.config, "num_attention_heads"): self.config.num_attention_heads = 16
        if not hasattr(self.config, "num_key_value_heads"): self.config.num_key_value_heads = getattr(self.config, "num_attention_heads", 16)
        if not hasattr(self.config, "intermediate_size"): self.config.intermediate_size = getattr(self.config, "hidden_size", 4096) * 4
        if not hasattr(self.config, "rms_norm_eps"): self.config.rms_norm_eps = 1e-5
        if not hasattr(self.config, "max_position_embeddings"): self.config.max_position_embeddings = 8192
        if not hasattr(self.config, "hidden_act"): self.config.hidden_act = "silu"
        if not hasattr(self.config, "rope_theta"): self.config.rope_theta = 10000.0

        if device.startswith("cuda"):
            from .persist.safetensor_model_persister import _should_use_direct_io, init_model_io
            self._use_direct_io = _should_use_direct_io(str(self.checkpoint_path))
            init_model_io(str(self.checkpoint_path), use_direct_io=self._use_direct_io)
        else:
            self._use_direct_io = False

        # Build loader strategy (auto-detects EXL3 vs standard safetensors)
        self.loader_strategy: LoaderStrategy = build_loader_strategy(
            str(self.checkpoint_path), self
        )

        self.init_model()

        # fill in any layer name keys that don't resolve on this model (unknown architectures)
        _detected = _auto_detect_layer_names(self.model)
        for key, val in _detected.items():
            if val and (key not in self.layer_names_dict or not _layer_path_exists(self.model, self.layer_names_dict[key])):
                self.layer_names_dict[key] = val

        # get layer count:
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        layers_count = len(model_attr)


        self.layer_names = [self.layer_names_dict['embed']] + [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in
                                                               range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.main_input_name = "input_ids"

        # model weights prefetch cuda stream
        self.prefetching = prefetching

        if self.compression is not None:
            self.prefetching = False
            print(f"not support prefetching for compression for now. loading with no prepetching mode.")

        # this operation should run only if gpu is available
        if prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        # MoE detection — must happen before pinning so setup_model has correct context
        self._detect_moe()

        # Build MoE key remapper (one-time, before any weight loading)
        if self.is_moe:
            from .moe_key_remap import build_moe_key_remap
            self._moe_key_remapper = build_moe_key_remap(
                self.model, str(self.checkpoint_path), self.layer_names_dict
            )
        else:
            self._moe_key_remapper = None

        # Pin shared layers (embed, norm, lm_head) permanently in VRAM.
        # Skips load/inject/free in forward() for these layers every token.
        self.pinned_layer_names: set = set()
        self.non_expert_pinned_layer_names: set = set()
        self.loader_strategy.setup_model(
            self.model, self.config, self.running_device, self.running_dtype,
            pinned_layer_names=self.pinned_layer_names,
        )
        
        # Refresh layers list to pick up newly replaced modules (e.g. ExLlamaV3Linear for lm_head)
        self.set_layers_from_layer_names()

        self.draft_tokens = draft_tokens if draft_tokens is not None else 5

        # Build offset index BEFORE _init_pinned_layers so Phase 2 can use
        # _load_non_expert_strategy (80 MB) instead of full layer load (2.9 GB).
        self._expert_offset_index = None
        if device.startswith("cuda") and self.is_moe:
            from .persist.expert_offset_index import ExpertOffsetIndex
            self._expert_offset_index = ExpertOffsetIndex(str(self.checkpoint_path))
            self._expert_offset_index.build_or_load()

        # Geometry is needed by plan_vram below (for its min_resident floor), so
        # it is computed BEFORE pinning rather than at cache-sizing time. Its
        # inputs are all ready: _detect_moe() ran above, set_layers_from_layer_names()
        # ran above that, the offset index just above.
        self._moe_geometry = None
        if self.is_moe and self._expert_offset_index is not None:
            self._moe_geometry = compute_moe_geometry(
                self._expert_offset_index,
                num_layers=len(self.layers),
                moe_top_k=self.moe_top_k,
                moe_num_experts=self.moe_num_experts,
            )

        self._vram_plan = None
        if device.startswith("cuda") and torch.cuda.is_available():
            from .vram_plan import (DEFAULT_ALLOC_FACTOR, VramShape,
                                    kv_bytes_per_token, plan_vram,
                                    reserve_floor_bytes, score_elem_bytes)
            _free_vram, _ = torch.cuda.mem_get_info()
            _n_heads = self.config.num_attention_heads
            _head_dim = getattr(self.config, "head_dim", None)
            if not isinstance(_head_dim, int):
                _head_dim = self.config.hidden_size // _n_heads
            _layer0 = Path(self.checkpoint_path) / \
                f"{self.layer_names_dict['layer_prefix']}.0.safetensors"
            try:
                _layer_slack = _layer0.stat().st_size
            except Exception:
                _layer_slack = int(0.6 * 1024 ** 3)   # hardware fallback
            # Bytes of lm_head weight that go resident when the logits are
            # computed -- a file stat on the checkpoint's own head shard, the
            # same way _layer_slack above is one, so no model number is typed
            # (§5). A tied-embedding checkpoint ships an EMPTY lm_head shard
            # (Llama-3.2-1B: 16 bytes) because the head IS the embedding, so
            # read the embedding shard instead -- decided by the config flag,
            # not by a size threshold. Missing shard -> 0, which under-reserves
            # by this one term rather than failing the load.
            _head_name = (self.layer_names_dict['embed']
                          if getattr(self.config, "tie_word_embeddings", False)
                          else self.layer_names_dict['lm_head'])
            try:
                _head_bytes = (Path(self.checkpoint_path)
                               / f"{_head_name}.safetensors").stat().st_size
            except Exception:
                _head_bytes = 0
            # kv_growth_factor/kv_capacity_buckets are algorithmic properties of
            # the KV cache class _create_kv_cache will build (Amendment 1,
            # task-6-brief.md), not of any model (§5) — derived by mirroring
            # _create_kv_cache's own mode resolution, not guessed.
            # Chunk-major GPU retention exists ONLY in the fused-EXL3 forward
            # (_chunk_major_gpu, moe_patch.py): it holds one layer's routed
            # experts across prefill chunks. Measured on Qwen3-235B 2026-08-15 at
            # a flat 1.514 GB = 128 experts x 11.8 MB, constant across
            # S=5000..12288 and reset per layer. The module-list and dense
            # forwards have no equivalent, so DETECT the layout rather than
            # assume it -- charging MiniMax's 256 experts would over-reserve
            # 2.28 GB and shrink its expert cache for nothing.
            #
            # NOT clamped to the expert VRAM cache budget, even though that is
            # the runtime bound: that budget is `free - reserve`, so clamping
            # would make the reserve depend on itself. Unclamped can only
            # over-reserve, and the clamp is ~6x from binding on this box.
            #
            # num_experts, not the count actually routed: a full prefill chunk
            # routes essentially all of them, and over-reserving is the safe
            # error for a value whose job is "every request up to this length
            # fits" (S5: both numbers come from _moe_geometry, never literals).
            _resident_expert_bytes = 0
            if self._moe_geometry is not None:
                # Local imports: both names are imported further down in
                # __init__, past this point, so referencing them here without
                # this would make them unbound locals. (It did: both MoE models
                # failed to load with UnboundLocalError, and the 836-test suite
                # could not see it -- nothing unit-tests this path.)
                from .moe_patch import detect_moe_blocks
                from .persist.loader_strategy import EXL3LoaderStrategy
                if getattr(self, "_detected_moe_blocks", None) is None:
                    self._detected_moe_blocks = detect_moe_blocks(self.model)
                if (isinstance(self.loader_strategy, EXL3LoaderStrategy)
                        and any(b.expert_style == "fused"
                                for b in self._detected_moe_blocks)):
                    _resident_expert_bytes = int(
                        self._moe_geometry["num_experts"]
                        * self._moe_geometry["expert_bytes"])
            _shape = VramShape(
                kv_per_token=kv_bytes_per_token(
                    num_hidden_layers=self.config.num_hidden_layers,
                    num_key_value_heads=getattr(self.config, "num_key_value_heads",
                                                _n_heads),
                    head_dim=_head_dim,
                    kv_cache_mode=self.kv_cache_mode),
                N=1, chunk=max(1, self.prefill_chunk_size),
                hidden=self.config.hidden_size, n_heads=_n_heads,
                vocab=self.config.vocab_size,
                elem_bytes=torch.finfo(self.running_dtype).bits // 8,
                # Kernel property, read off the model init_model() just
                # built (line ~1421, before this block): sdpa materializes an
                # S-sized score buffer and flash_attention_2 does not, a gap
                # that GROWS with S and flips OOM outcomes at large context
                # (todo row 10). Charging one kernel's number for both
                # under-reserves batched_generate, which requires sdpa.
                score_elem_bytes=score_elem_bytes(self._active_attn_impl),
                resident_expert_bytes=_resident_expert_bytes,
                head_bytes=_head_bytes,
                alloc_factor=DEFAULT_ALLOC_FACTOR,
                kv_growth_factor=FP8DynamicCache._GROWTH,
                kv_capacity_buckets=self._kv_cache_is_bucketed())
            _min_resident = (self._moe_geometry["expert_bytes"]
                             if self._moe_geometry else 0)
            # Kernel-conditional floor (todo row 8, decided 2026-08-15). Rows
            # 6/7 made the peak form kernel-aware, which cut every sdpa load's
            # derived context; this raises the floor just far enough that an
            # sdpa load derives what flash would have, paid out of the expert
            # VRAM cache / pinning budget. flash loads keep the base floor
            # unchanged. Derived per geometry -- see reserve_floor_bytes.
            _reserve_floor = reserve_floor_bytes(
                shape=_shape, base_floor=_VRAM_RESERVE_FLOOR_BYTES,
                free_vram=_free_vram, min_resident=_min_resident,
                layer_slack=_layer_slack)
            self._vram_plan = plan_vram(
                free_vram=_free_vram, shape=_shape,
                min_resident=_min_resident,
                requested_max_seq_len=max_seq_len,
                position_limit=getattr(self.config, "max_position_embeddings",
                                       2 ** 31 - 1),
                layer_slack=_layer_slack,
                reserve_floor=_reserve_floor)
            self._vram_shape = _shape      # Task 8's predicate re-uses it at N>1
            self.max_seq_len = self._vram_plan.max_seq_len
            # expert_vram_budget_for at this same free-VRAM snapshot -- the
            # ceiling the reserve implies, not yet corrected for what pinning
            # (below) actually consumes; the real post-pinning figure is
            # printed at the expert-cache-creation call site further down.
            from .vram_plan import expert_vram_budget_for as _expert_budget_for
            _expert_budget_planned = _expert_budget_for(self._vram_plan, _free_vram)
            _gib = 1024.0 ** 3
            print(f"[Vaayu-VRAM] {self._vram_plan.reason}; "
                  f"pinning_budget={self._vram_plan.pinning_budget / _gib:.2f} GiB; "
                  f"expert cache ceiling={_expert_budget_planned / _gib:.2f} GiB")
            # CLAUDE.md §5: warn rather than silently degrade when the plan
            # leaves nothing for pinning -- a real hardware limit, not a bug
            # to hide. (The parallel "expert budget under one expert" case is
            # warned at the expert-cache-creation call site below, where the
            # real post-pinning free VRAM is known.)
            if self._vram_plan.pinning_budget == 0:
                print(f"[Vaayu-VRAM] WARNING: pinning_budget is 0 — no shared "
                      f"or transformer layers can be pinned in VRAM. Every "
                      f"layer streams from disk/RAM on every token; expect "
                      f"very slow decode (a real hardware limit on this box, "
                      f"not a bug).")

        if self.max_seq_len is None:
            # CPU / no-CUDA path (§5: dense and non-EXL3/CPU paths must keep
            # working; CPU builds no VramPlan at all since there is no free-VRAM
            # figure to derive a ceiling from). Keep the historical default
            # rather than leaving max_seq_len unset.
            self.max_seq_len = 512

        if device.startswith("cuda"):
            self._init_pinned_layers()

        # RAM layer cache — measured AFTER _init_pinned_layers so pinned-layer RAM
        # consumption is already reflected in available memory, preventing over-budget.
        if device.startswith("cuda"):
            from .persist.ram_cache import (LayerRAMCache, auto_detect_cache_budget,
                                            describe_cache_budget)
            # MoE models skip pre-population entirely; the RAM layer cache budget
            # is wasted allocation. Their experts live in the PINNED expert RAM
            # cache instead, which is a different and much scarcer resource --
            # page-locked, unswappable, and rounded up to a power of two by the
            # allocator, hence its own 8.5 GiB ceiling rather than this one.
            if self.is_moe:
                _cache_budget = 0
                _budget_note = "0.0 GB budget (MoE: experts use the pinned expert RAM cache)"
            else:
                _cache_budget = auto_detect_cache_budget(override_bytes=ram_cache_budget)
                _budget_note = describe_cache_budget(override_bytes=ram_cache_budget)
            self.ram_cache = LayerRAMCache(_cache_budget)
            print(f"[Vaayu-Cache] {_budget_note}")
            self._prepopulate_ram_cache()
            self.loader_strategy._cache = self.ram_cache
        else:
            self.ram_cache = None

        # Speculative drafter — built HERE, above the MoE section, because the
        # expert VRAM cache below sizes itself from `mem_get_info() - reserve`
        # and the drafter is a real resident cost the cache must not double-spend
        # (bug 12a: 10.5 GB of cache + 2.47 GB of drafter + a 3.50 GiB reserve
        # overcommitted 15.47 GiB, and the drafter's first F.linear died with
        # CUBLAS_STATUS_INTERNAL_ERROR). Loading it first makes the cache's own
        # measurement account for it, with no size estimate anywhere.
        # The Fix #9 MoE gate that may DISABLE it runs further down, after the
        # RAM expert cache it needs exists and before the VRAM cache is sized.
        self.speculator = None
        if drafter_model_path or use_prompt_lookup:
            from .speculative import build_speculator
            self.speculator = build_speculator(
                drafter_model_path=drafter_model_path,
                base_model=self,
                device=device,
                use_prompt_lookup=use_prompt_lookup,
            )

        # MoE Expert-Aware Streaming: expert cache and patched forwards.
        # _expert_offset_index is initialized above (before _init_pinned_layers).
        self._expert_history: dict[int, list[int]] = {}
        self._expert_cache = None
        self._moe_blocks: list = []
        # Non-expert tensor cache: byte-budgeted LRU avoids re-reading ~44 MB per
        # non-pinned MoE layer per token. Capped to prevent unbounded RAM growth
        # on large models (e.g. Qwen3-235B with 94 layers).
        self._non_expert_tensor_cache = None  # initialized below for MoE
        self._expert_vram_cache = None  # initialized below for MoE (Fix #7)
        # Expert usage histogram (measurement only, ships dark). Default None
        # so the attribute always exists, on every load path (dense, CPU, or
        # no VRAM cache) — the hook's getattr must never be the only guard.
        self._expert_usage = None
        if self.running_device.startswith("cuda") and self.is_moe:
            # Hard-block guard (Task 5 follow-up) — must fire before any MoE
            # cache/executor/prefetch object is created. See
            # check_lookahead_flag_safe() docstring for the full rationale.
            check_lookahead_flag_safe()
            # Capture the flag decision ONCE, here, right after the guard above
            # raises if VAAYU_LOOKAHEAD=1 — so this is always False on this path
            # today. Read from this captured value at forward time (moe_patch.py),
            # never re-read os.environ live, so mutating the env var AFTER init
            # can't re-arm the known-corrupting Phase B prefetch path (final
            # whole-branch review finding: closes a guard-bypass window).
            self._lookahead_enabled = os.environ.get("VAAYU_LOOKAHEAD") == "1"
            # Diagnostic only (Task 3 GIL-contention investigation, not shipped
            # behavior): when set, logs wall-clock start/end timestamps around
            # the prefill stager's per-layer reads and the consumer's per-layer
            # wait/compute, so real overlap can be measured directly instead of
            # via a ptrace-based profiler (py-spy corrupts this process's
            # io_uring state — see io_uring_cqe_overflow_bug memory).
            self._overlap_trace = os.environ.get("VAAYU_OVERLAP_TRACE") == "1"
            from .persist.expert_cache import ExpertRAMCache, ExpertVRAMCache, LRUTensorCache, SlabArena
            from .moe_patch import detect_moe_blocks, patch_moe_blocks
            from .persist.loader_strategy import EXL3LoaderStrategy
            import psutil as _psutil

            # Detect the expert layout BEFORE sizing caches: which layouts can use
            # the pinned slab arena and which consume the prefill stager are two
            # separate questions (see the predicates below), and the arena is
            # allocated below — before patch_moe_blocks() would otherwise reveal
            # the layout. Detection is reused by patch_moe_blocks so the module
            # tree is walked exactly once.
            if getattr(self, "_detected_moe_blocks", None) is None:
                self._detected_moe_blocks = detect_moe_blocks(self.model)
            _styles_detected = [b.expert_style for b in self._detected_moe_blocks]
            _is_exl3 = isinstance(self.loader_strategy, EXL3LoaderStrategy)
            # Independent capabilities — see the two predicates' docstrings.
            # The module-list layout reads the arena but does not consume staged
            # layers, a combination the previous single gate could not express.
            self._moe_slab_capable = moe_supports_slab_arena(_styles_detected, _is_exl3)
            self._moe_stager_capable = moe_supports_prefill_stager(_styles_detected, _is_exl3)

            _mem = _psutil.virtual_memory()
            # RAM arbitration ledger (diagnostic): what is already resident before
            # any expert/prefix cache allocates. On a constrained box this load-time
            # footprint — model non-expert weights, pinned layers, checkpoint
            # buffers, CUDA/PyTorch host context — is what squeezes every cache that
            # sizes itself from "free RAM at my construction point" afterward.
            _snap = ram_snapshot()
            print(f"[Vaayu-RAM] ledger @ cache-sizing: avail={_snap['available'] / 1e9:.2f}GB "
                  f"rss={_snap['rss'] / 1e9:.2f}GB priv_dirty={_snap['private_dirty'] / 1e9:.2f}GB "
                  f"shared_clean={_snap['shared_clean'] / 1e9:.2f}GB swap={_snap['swap'] / 1e9:.2f}GB")

            _geo = self._moe_geometry
            # Ceiling DERIVED, not asserted: the expert arena is pinned before
            # compute_prefix_cache_budget measures what is left, so room for the
            # sessions we intend to guarantee is subtracted first. Measured
            # 2026-08-16 on Qwen3-235B: raising the stale 8.5 GiB literal took
            # the warm wall 36.6 -> 27.3 s with prefix reuse unchanged at 83 %.
            from .vram_plan import kv_bytes_per_token as _kv_per_tok
            try:
                _kv_tok = _kv_per_tok(
                    num_hidden_layers=self.config.num_hidden_layers,
                    num_key_value_heads=getattr(self.config, "num_key_value_heads",
                                                self.config.num_attention_heads),
                    head_dim=getattr(self.config, "head_dim",
                                     self.config.hidden_size
                                     // self.config.num_attention_heads),
                    kv_cache_mode=self.kv_cache_mode)
            except Exception:
                _kv_tok = 0        # D3: a sizing helper must not become the crash
            _ceiling = derive_expert_ceiling(
                available_bytes=_mem.available,
                prefix_sessions=self.prefix_sessions,
                kv_bytes_per_token=_kv_tok,
                max_seq_len=self.max_seq_len or 0,
                prefix_enabled=bool(getattr(self, "enable_prefix_cache", False)))
            print(f"[Vaayu-MoE] Expert cache ceiling {_ceiling / 1e9:.2f} GB "
                  f"(derived: available {_mem.available / 1e9:.2f} GB, reserving "
                  f"{self.prefix_sessions} prefix session(s) of "
                  f"{(_kv_tok * (self.max_seq_len or 0)) / 1e9:.2f} GB each)")
            _expert_budget, _ne_budget, _max_per_layer = compute_expert_budget(
                available_bytes=_mem.available,
                working_set_bytes=_geo["working_set_bytes"],
                num_moe_layers=_geo["num_moe_layers"],
                expert_bytes=_geo["expert_bytes"],
                moe_top_k=_geo["top_k"],
                moe_num_experts=_geo["num_experts"],
                ceiling_bytes=_ceiling,
            )
            self._non_expert_tensor_cache = LRUTensorCache(_ne_budget)
            print(f"[Vaayu-Cache] Non-expert tensor cache: {_ne_budget / 1e9:.1f} GB budget")
            # Pinned slab arena (zero-clone expert path): the whole cache budget
            # preallocated as fixed slabs sized to the model's largest expert
            # span (derived from the offset index — CLAUDE.md §5). +pool_size
            # extra slabs so in-flight loader threads never starve while the
            # cache is full. Pinned memory is unswappable — footprint equals
            # the budget the cache occupied anyway, but WARN for visibility.
            from .persist.safetensor_model_persister import compute_slab_bytes
            self._expert_slab_arena = None
            if not self._moe_slab_capable:
                # This layout's forward never reads the arena — it populates the
                # cache via ExpertRAMCache.put() (clone path). Allocating anyway
                # would pin a second full copy of the expert budget that nothing
                # reads. Skip compute_slab_bytes too: it rescans the whole offset
                # index for a number no one will use.
                _styles = sorted({b.expert_style for b in self._detected_moe_blocks})
                print(f"[Vaayu-MoE] Expert slab arena: SKIPPED — expert layout "
                      f"{_styles or ['none detected']} does not read the pinned "
                      f"slab arena. Freed ~{_expert_budget / 1e9:.1f} GB of pinned "
                      f"RAM; experts are cached via the clone path instead.")
                _slab_bytes = 0
            else:
                _slab_bytes = compute_slab_bytes(self._expert_offset_index)
            _pin = torch.cuda.is_available()
            if _slab_bytes > 0:
                # The pow2 clamp existed only because the backing store was ONE
                # pinned allocation, which the caching host allocator charges at
                # the next power of two. The arena is now several exact-pow2
                # chunks, each charged exactly, so there is nothing to clamp
                # away: take the full unclamped count (Plan L; measured
                # 2026-08-07, 9.14GB charged vs 17.18GB for one tensor, every
                # slab kept, H2D unchanged).
                _n_slabs = unclamped_slab_count(
                    _expert_budget, _slab_bytes, get_expert_pool_size())
                try:
                    self._expert_slab_arena = SlabArena(_n_slabs, _slab_bytes, pin=_pin)
                except RuntimeError as _slab_err:
                    print(f"[Vaayu-MoE] WARNING: pinned slab alloc failed ({_slab_err}); "
                          f"retrying pageable (H2D will be slower)")
                    try:
                        self._expert_slab_arena = SlabArena(_n_slabs, _slab_bytes, pin=False)
                    except RuntimeError as _pageable_err:
                        # Was silent (H5.5): the arena fell through to None with
                        # nothing printed, so a box with no slab arena at all
                        # looked identical to a healthy one until decode crawled.
                        # Same helper the rejected-degraded path below already
                        # uses -- this is the other way into the same state.
                        self._expert_slab_arena = None
                        print(slab_arena_exhausted_warning(_pageable_err))
                # A chunked arena keeps what it secured when a later chunk fails
                # (E5). Keep the degraded arena only while its cache share still
                # covers the working set; otherwise drop it and retry pageable,
                # which trades bandwidth for capacity rather than thrashing. A
                # degraded arena that is ALREADY pageable has nothing to trade --
                # degraded_arena_decision keeps it unconditionally in that case.
                _arena = self._expert_slab_arena
                if _arena is not None and _arena.n_slabs < _arena.requested_n_slabs:
                    _kept, _why = degraded_arena_decision(
                        _arena.n_slabs, _arena.requested_n_slabs, _slab_bytes,
                        get_expert_pool_size(), _geo["working_set_bytes"], _arena.pin)
                    print(f"[Vaayu-MoE] WARNING: slab arena allocated only "
                          f"{_arena.n_slabs} of {_arena.requested_n_slabs} slabs "
                          f"({_arena.n_slabs * _slab_bytes / 1e9:.2f} of "
                          f"{_arena.requested_n_slabs * _slab_bytes / 1e9:.2f}GB) — {_why}")
                    if not _kept:
                        self._expert_slab_arena = None
                        try:
                            self._expert_slab_arena = SlabArena(
                                _n_slabs, _slab_bytes, pin=False)
                        except RuntimeError as _pageable_err:
                            self._expert_slab_arena = None
                            print(slab_arena_exhausted_warning(_pageable_err))
                if self._expert_slab_arena is not None:
                    _arena = self._expert_slab_arena
                    _chunk_note = ""
                    if _arena.n_slabs < _arena.requested_n_slabs:
                        _chunk_note = (f" — DEGRADED: {_arena.n_slabs} of "
                                       f"{_arena.requested_n_slabs} slabs allocated")
                    print(f"[Vaayu-MoE] Expert slab arena: {_arena.n_slabs} x "
                          f"{_slab_bytes / 1e6:.1f}MB in {_arena.chunk_count} "
                          f"power-of-two chunks "
                          f"({'pinned' if _arena.pin else 'pageable'}, "
                          f"{_arena.charged_bytes / 1e9:.2f}GB "
                          f"charged for {_arena.n_slabs * _slab_bytes / 1e9:.2f}GB "
                          f"of slabs){_chunk_note}")
                    _snap = ram_snapshot()
                    print(f"[Vaayu-RAM] ledger @ post-arena: avail={_snap['available'] / 1e9:.2f}GB "
                          f"rss={_snap['rss'] / 1e9:.2f}GB priv_dirty={_snap['private_dirty'] / 1e9:.2f}GB "
                          f"shared_clean={_snap['shared_clean'] / 1e9:.2f}GB swap={_snap['swap'] / 1e9:.2f}GB")
            if self._expert_slab_arena is not None:
                _arena_bytes = cache_budget_for_arena(
                    self._expert_slab_arena.n_slabs,
                    self._expert_slab_arena.slab_bytes,
                    get_expert_pool_size())
                if _arena_bytes < _expert_budget:
                    # The trim fires whenever the arena's CACHE share is smaller
                    # than the requested budget — which includes the unclamped
                    # case, where the staging addend alone accounts for the gap.
                    # Do not call the arena "clamped" here: it often is not.
                    print(f"[Vaayu-MoE] Expert RAM cache budget trimmed "
                          f"{_expert_budget / 1e9:.2f} -> {_arena_bytes / 1e9:.2f} GB "
                          f"to match the slab arena's cache share (the arena, not "
                          f"the budget, is what binds)")
                    _expert_budget = _arena_bytes
                if _expert_budget <= 0:
                    # A real hardware limit, not a bug to hide (CLAUDE.md §5).
                    # put_slab handles a 0 budget safely, so the failure is
                    # silent otherwise: every expert streams from disk per token.
                    print(f"[Vaayu-MoE] WARNING: Expert RAM cache DISABLED — the "
                          f"{self._expert_slab_arena.n_slabs}-slab arena has nothing "
                          f"beyond the {get_expert_pool_size()} in-flight staging slabs "
                          f"the loader must keep free, so the cache gets 0 bytes. Every "
                          f"expert streams from disk on every token; expect very slow "
                          f"decode (a real hardware limit on this box, not a bug).")
            self._expert_cache = ExpertRAMCache(
                _expert_budget, max_per_layer=_max_per_layer,
                arena=self._expert_slab_arena,
            )

            # Fix #9 (REPORT §2.4): a verify pass routes (draft_tokens+1) positions
            # at once — per-layer unique experts ≈ positions × top_k, so the working
            # set is ~Nx the single-token one. When that cannot fit the expert cache,
            # speculation guarantees thrash and is ~I/O-neutral per accepted token.
            # Warn with the derived numbers and disable — loudly, never silently.
            if (self.speculator is not None and self.is_moe
                    and getattr(self, '_expert_cache', None) is not None
                    and getattr(self, '_moe_geometry', None) is not None):
                if not spec_decode_fits(self._moe_geometry, self.draft_tokens,
                                        self._expert_cache.budget_bytes):
                    _sws = compute_spec_working_set(self._moe_geometry, self.draft_tokens + 1)
                    print(f"[Vaayu-Spec] WARNING: speculative verify working set "
                          f"{_sws / 1e9:.1f} GB ((draft_tokens={self.draft_tokens}+1) positions × "
                          f"top-{self._moe_geometry['top_k']}) exceeds expert cache budget "
                          f"{self._expert_cache.budget_bytes / 1e9:.1f} GB — speculation would "
                          f"thrash the cache and gains ~nothing per accepted token on this "
                          f"hardware. DISABLING speculative decoding for this MoE model. "
                          f"(Reduce draft_tokens or use a smaller model to re-enable.)")
                    self.speculator = None
                    # Hand the drafter's VRAM back to the DRIVER before the
                    # cache is sized below. A plain drop leaves those bytes in
                    # PyTorch's caching-allocator pool, which mem_get_info()
                    # cannot see (the same driver-vs-allocator distinction as
                    # bug 11a, cutting the other way) -- so the expert cache
                    # would silently shrink by the drafter's footprint to hold
                    # memory nothing will ever use.
                    try:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            # VRAM-resident expert cache (Fix #7): hot experts skip per-token H2D.
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                from .vram_plan import expert_vram_budget_for
                _free_vram, _ = torch.cuda.mem_get_info()
                _vram_budget = expert_vram_budget_for(self._vram_plan, _free_vram)
                if _vram_budget >= self._moe_geometry["expert_bytes"]:
                    self._expert_vram_cache = ExpertVRAMCache(
                        _vram_budget,
                        track_evicted_keys=bool(getattr(self, "enable_expert_usage", False)),
                    )
                    print(f"[Vaayu-MoE] Expert VRAM cache: {_vram_budget / 1e9:.1f} GB "
                          f"(~{int(_vram_budget // max(1, self._moe_geometry['expert_bytes']))} "
                          f"experts resident on GPU)")
                else:
                    print(f"[Vaayu-MoE] WARNING: Expert VRAM cache disabled — free VRAM "
                          f"{_free_vram/1e9:.2f}GB leaves {_vram_budget/1e9:.2f}GB after the "
                          f"{self._vram_plan.reserve_bytes/1e9:.2f}GB reserve, under one "
                          f"{self._moe_geometry['expert_bytes']/1e6:.1f}MB expert. Every expert "
                          f"streams per token; expect very slow decode (a real hardware limit, "
                          f"not a bug).")

            # Expert usage histogram (measurement only, ships dark). Rows are
            # indexed by the model's own layer_idx; both dimensions derived (§5).
            if getattr(self, "enable_expert_usage", False) and self.is_moe:
                from vaayu.expert_usage import ExpertUsageTracker
                self._expert_usage = ExpertUsageTracker(
                    n_rows=int(self.config.num_hidden_layers),
                    n_experts=int(self._moe_geometry["num_experts"]),
                    device=self.running_device,
                )
                print(f"[Vaayu-Usage] expert usage histogram ON "
                      f"({self.config.num_hidden_layers} rows x "
                      f"{self._moe_geometry['num_experts']} experts)")

            _ws_gb = _geo["working_set_bytes"] / 1e9
            print(f"[Vaayu-MoE] Expert RAM cache: {_expert_budget / 1e9:.1f} GB budget "
                  f"(~{int(_expert_budget // max(1, _geo['expert_bytes']))} expert slots, "
                  f"max {_max_per_layer}/layer, "
                  f"expert={_geo['expert_bytes']/1e6:.1f}MB, working_set={_ws_gb:.2f}GB)")
            if _geo["working_set_bytes"] > _expert_budget:
                print(f"[Vaayu-MoE] WARNING: working set {_ws_gb:.2f}GB exceeds cache "
                      f"budget {_expert_budget/1e9:.2f}GB — decode will thrash. This "
                      f"model is too large for the expert cache on this hardware; "
                      f"expect low hit rate (a real limit, not a bug).")

            # Persistent executor for expert cache-miss loads, created once here instead
            # of per-layer-per-token inside sparse_forward. get_expert_pool_size() is
            # fixed for the model's lifetime (set once during init_model_io), so a
            # pool sized to it behaves identically to the old per-call
            # min(len(misses), get_expert_pool_size()) cap — .map() only dispatches
            # as many tasks as there are misses regardless of pool size.
            self._moe_executor = ThreadPoolExecutor(max_workers=get_expert_pool_size())
            atexit.register(self._moe_executor.shutdown, wait=False)

            # Dedicated executor for lookahead (Phase B) prefetch so predicted
            # next-layer reads can NEVER preempt on-path decode-miss reads on
            # _moe_executor (CR-2). Small: prefetch is opportunistic, not
            # latency-critical. Derived from the expert pool size, capped low.
            self._stager_executor = ThreadPoolExecutor(
                max_workers=max(1, get_expert_pool_size() // 4))
            atexit.register(self._stager_executor.shutdown, wait=False)

            self._moe_blocks = patch_moe_blocks(
                model=self.model,
                expert_cache=self._expert_cache,
                offset_index=self._expert_offset_index,
                checkpoint_path=str(self.checkpoint_path),
                top_k=self.moe_top_k,
                device=self.running_device,
                dtype=self.running_dtype,
                vaayu_ref=self,
                loader_strategy=self.loader_strategy,
                blocks=self._detected_moe_blocks,
            )
            # Lookahead predictor (Task 12): O(1) next-block lookup by layer idx.
            self._moe_block_by_idx = {b.layer_idx: b for b in self._moe_blocks}
            self._lookahead_state = {"hits": 0, "total": 0}
            # Cache-aware routing (advanced-opt #4). Default 0.0 => feature OFF
            # (output byte-identical to plain routing). >0 biases top-k toward
            # resident experts to cut NVMe read volume; enable only after a
            # perplexity A/B (see plan Task 6). Model-agnostic: no geometry here.
            self._cache_bias = float(os.environ.get("VAAYU_CACHE_BIAS", "0.0"))

            # NVMe controller warm-up (Task 7): prime the drive's DRAM cache so
            # token 1 isn't served cold. Fire-and-forget daemon; reads a
            # hardware-sized budget of the first experts' bytes into a throwaway
            # buffer and exits. Disabled with VAAYU_NO_WARMUP=1.
            if (os.environ.get("VAAYU_NO_WARMUP") != "1"
                    and getattr(self, "_expert_offset_index", None) is not None):
                # Hardware constant: NVMe controllers cache on the order of
                # hundreds of MB in on-device DRAM; 512 MB primes it without a
                # meaningful init delay. Property of the drive, not the model.
                _WARMUP_BYTES = 512 * 1024 ** 2
                _warm_ranges = compute_warmup_ranges(
                    self._expert_offset_index, str(self.checkpoint_path), _WARMUP_BYTES)

                def _nvme_warmup():
                    _scratch = bytearray(8 * 1024 ** 2)  # reused per read, discarded
                    _view = memoryview(_scratch)
                    for _shard, _off, _len in _warm_ranges:
                        try:
                            _fd = os.open(_shard, os.O_RDONLY)
                            try:
                                _remaining = _len
                                _pos = _off
                                while _remaining > 0:
                                    _chunk = min(_remaining, len(_scratch))
                                    _n = os.preadv(_fd, [_view[:_chunk]], _pos)
                                    if _n <= 0:
                                        break
                                    _remaining -= _n
                                    _pos += _n
                            finally:
                                os.close(_fd)
                        except OSError:
                            continue  # warm-up is best-effort; never fail init
                    print("[Vaayu-IO] NVMe warm-up complete")

                import threading as _thr
                _thr.Thread(target=_nvme_warmup, daemon=True).start()

            print(f"[Vaayu-MoE] {len(self._moe_blocks)} MoE blocks patched for sparse streaming")

            # Fix #7 follow-up: when the VRAM budget can't hold one token's
            # full working set, an unrestricted LRU deterministically hits 0%
            # (single-token self-eviction). Restrict to a layer prefix that
            # fits; WARN honestly (CLAUDE.md §5: warn, don't silently degrade).
            self._vram_cacheable_layers = None
            if self._expert_vram_cache is not None and self._moe_blocks:
                _moe_ids = [b.layer_idx for b in self._moe_blocks]
                self._vram_cacheable_layers = compute_vram_layer_range(
                    self._expert_vram_cache.budget_bytes, self._moe_geometry, _moe_ids)
                if self._vram_cacheable_layers is not None:
                    print(f"[Vaayu-MoE] WARNING: VRAM cache budget "
                          f"{self._expert_vram_cache.budget_bytes / 1e9:.1f}GB < per-token "
                          f"working set {self._moe_geometry['working_set_bytes'] / 1e9:.1f}GB — "
                          f"an unrestricted LRU would hit 0% (self-eviction). Restricting "
                          f"VRAM cache to the first {len(self._vram_cacheable_layers)}/"
                          f"{len(_moe_ids)} MoE layers (partial caching).")
        elif self.is_moe:
            print("[Vaayu-MoE] MoE model on non-CUDA device — expert-aware streaming disabled, "
                  "falling back to full-layer loading")
            if getattr(self, "enable_expert_usage", False):
                print("[Vaayu-Usage] WARNING: enable_expert_usage is set but no tracker will be "
                      "built — no VRAM expert cache exists on this device, so there would be "
                      "nothing to join eviction data against.")

        # Dense-only (Task 7 re-scope): MoE now gets the expert-range-scoped
        # warm-up above (~:938); this whole-layer version stays for dense so
        # dense keeps a working (if cruder) warm-up path, per CLAUDE.md §0
        # (dense must keep working; dense perf optimization itself is a
        # deferred future target, not in scope here). Same VAAYU_NO_WARMUP
        # flag as the new mechanism so exactly one warm-up path is active per
        # model type and the flag disables warm-up uniformly.
        if (device.startswith("cuda") and not self.is_moe
                and os.environ.get("VAAYU_NO_WARMUP") != "1"):
            self._warmup_nvme()

        # Cross-turn KV prefix cache (A1). Default off -> byte-identical to
        # today. RAM-backed pool sized from free system memory (hardware).
        # Model-agnostic: applies to MoE and dense alike, so it lives here,
        # after both branches converge, not inside the is_moe block above.
        self._prefix_cache = None
        if getattr(self, "enable_prefix_cache", False):
            # PrefixCacheStore drives to_cpu()/to_device()/resident_bytes() on
            # every cache it stores. Vaayu's own cache classes have them; the
            # bare transformers DynamicCache that kv_cache_mode "fp16" yields
            # (what "auto" resolves to for every non-EXL3 model) does not, and
            # store() would AttributeError. Since lookup() POPS an entry before
            # store() can put one back, failing per-turn would silently drain
            # the pool. Refuse up front and say why (CLAUDE.md §5: warn, never
            # silently degrade). Pre-existing limit of the prefix subsystem,
            # not of batched decode -- generate() calls the same store().
            from .batch import cache_supports_prefix_store
            _probe = self._create_kv_cache()
            if not cache_supports_prefix_store(_probe):
                print(f"[Vaayu-Prefix] WARNING: cross-turn prefix cache DISABLED — "
                      f"kv_cache_mode={self.kv_cache_mode!r} yields "
                      f"{type(_probe).__name__}, which does not implement "
                      f"to_cpu/to_device/resident_bytes. Load with "
                      f"kv_cache_mode='bf16' to enable it on this model.")
                self.enable_prefix_cache = False
            del _probe
        if getattr(self, "enable_prefix_cache", False):
            import psutil as _ps
            _free_ram = _ps.virtual_memory().available
            _pc_budget = compute_prefix_cache_budget(_free_ram)
            self._prefix_cache = self._build_prefix_store(_pc_budget)
            print(f"[Vaayu] cross-turn prefix cache ENABLED: "
                  f"{_pc_budget / 1e9:.2f} GB budget "
                  f"(free RAM {_free_ram / 1e9:.2f} GB - "
                  f"{_PREFIX_CACHE_RAM_RESERVE / 1e9:.2f} GB reserve)")
            if _pc_budget == 0:
                # Budget 0 does NOT disable the cache: store()'s eviction loop
                # keeps at least one entry regardless of budget
                # (`while current_bytes > budget and len(pool) > 1`), so a SINGLE
                # session's cross-turn reuse still works — measured 108s -> 10s
                # TTFT on a 2000-token shared prefix. What a 0 budget costs is
                # CONCURRENCY: the pool degrades to one resident session, so under
                # parallel fan-out (CLAUDE.md §0, multiple session_ids) every
                # concurrent stream but the most recent is evicted. Anything that
                # claims RAM before this point (notably the pinned expert slab
                # arena) shrinks the budget. Warn, don't hide (CLAUDE.md §5).
                print("[Vaayu] WARNING: prefix cache budget is 0 GB — single-session "
                      "reuse still works, but concurrent sessions (parallel fan-out) "
                      "collapse to one resident entry. Free RAM here is below the reserve.")

        # --- KV disk persistence tier (Subsystem A; ships dark) ---
        self._kv_writer = None
        if getattr(self, "enable_kv_persist", False) and self._prefix_cache is None:
            print("[KV] enable_kv_persist=True ignored: enable_prefix_cache is False")
        elif getattr(self, "enable_kv_persist", False):
            import shutil
            from .kv_persist import DiskWriter, KvDiskLoader
            persist_dir = getattr(self, "kv_persist_dir", None) or os.path.join(
                os.path.expanduser("~/.cache/vaayu"), "kv")
            os.makedirs(persist_dir, exist_ok=True)
            free = shutil.disk_usage(persist_dir).free
            self._kv_writer = DiskWriter(
                persist_dir, budget_bytes=free // 2,
                sync=getattr(self, "kv_persist_sync", False))
            atexit.register(self._kv_writer.close)
            KvDiskLoader(persist_dir).load_all(
                self._prefix_cache, self.config, self._prefix_cache.budget_bytes,
                writer=self._kv_writer)

        # RAM arbitration ledger (diagnostic), bracketing close: same categorized
        # snapshot as @ cache-sizing / @ post-arena, taken once model construction
        # (including the prefix cache, whether enabled or not) is complete.
        _snap = ram_snapshot()
        print(f"[Vaayu-RAM] ledger @ post-load: avail={_snap['available'] / 1e9:.2f}GB "
              f"rss={_snap['rss'] / 1e9:.2f}GB priv_dirty={_snap['private_dirty'] / 1e9:.2f}GB "
              f"shared_clean={_snap['shared_clean'] / 1e9:.2f}GB swap={_snap['swap'] / 1e9:.2f}GB")

    # if derived class needs to create generation config differently, like Mistrial, this function can be overridden
    def get_generation_config(self):
        # protective on generation config

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception as e:
            return GenerationConfig()

    # a chance to customize tokenizer
    def get_tokenizer(self, hf_token=None):
        if self._tokenizer_class is not None:
            tok_kwargs = {"use_fast": False, "trust_remote_code": True}
            if hf_token is not None:
                tok_kwargs["token"] = hf_token
            return self._tokenizer_class.from_pretrained(self.model_local_path, **tok_kwargs)
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    def get_use_better_transformer(self):
        return False

    @staticmethod
    def _apply_transformers_compat_patches():
        """
        Apply in-process shims for trust_remote_code models targeting a newer/different
        transformers API than the installed version. Never touches model files on disk.
        """
        # 1. OutputRecorder: some models import from utils.generic but it lives in
        #    utils.output_capturing in transformers ≥5.6.
        try:
            import transformers.utils.generic as _tf_generic
            if not hasattr(_tf_generic, "OutputRecorder"):
                from transformers.utils.output_capturing import OutputRecorder as _OR
                _tf_generic.OutputRecorder = _OR
        except Exception:
            pass

        # 2. ROPE 'default' type: removed from ROPE_INIT_FUNCTIONS in transformers ≥5.6.
        #    Standard RoPE (no scaling, attention_scaling=1.0).
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            if "default" not in ROPE_INIT_FUNCTIONS:
                def _rope_default(config, device=None, seq_len=None, layer_type=None, **_kw):
                    # Must honour partial rotary — see compute_rope_inv_freq.
                    return compute_rope_inv_freq(config, device), 1.0
                ROPE_INIT_FUNCTIONS["default"] = _rope_default
        except Exception:
            pass

        # 3. compute_default_rope_parameters: transformers ≥5.6 _init_weights calls this on
        #    any RotaryEmbedding with rope_type='default'. Patch PreTrainedModel._init_weights
        #    to inject the method if missing, rather than failing.
        try:
            from transformers.modeling_utils import PreTrainedModel
            import torch as _torch

            _orig_init_weights = PreTrainedModel._init_weights  # plain function in Python 3

            def _patched_init_weights(self_model, module):
                if (
                    "RotaryEmbedding" in type(module).__name__
                    and hasattr(module, "original_inv_freq")
                    and getattr(module, "rope_type", None) == "default"
                    and not hasattr(module, "compute_default_rope_parameters")
                ):
                    def _cdrp(config=None, _m=module):
                        cfg = config if config is not None else _m.config
                        # Must honour partial rotary — see compute_rope_inv_freq.
                        return compute_rope_inv_freq(cfg), 1.0
                    module.compute_default_rope_parameters = _cdrp
                return _orig_init_weights(self_model, module)

            PreTrainedModel._init_weights = _patched_init_weights
        except Exception:
            pass

    def _patch_fp8_quantizer_compat(self, quantization_config):
        """
        Fix FP8 quantizer incompatibilities for models with custom expert containers
        (e.g. MiniMax-M2's MiniMaxM2Experts). Applied in-process before preprocess_model().

        Problem: replace_with_fp8_linear tries to replace the whole experts ModuleList
        with FP8Experts (a batched-weight module). MiniMax-M2's expert structure is
        incompatible with that replacement — both structurally and for Vaayu's per-expert
        weight loading.

        Fix: add "experts" to modules_to_not_convert so the container is preserved.
        Individual nn.Linear inside each expert (w1/w2/w3) are still converted to
        FP8Linear, matching the per-tensor weight_scale_inv keys already on disk.

        Also adds num_experts config alias if missing (some quantizer code accesses
        config.num_experts but MiniMax uses num_local_experts).
        """
        # 1. Preserve expert containers — only replace individual Linear layers inside.
        # We replace the attribute entirely (list() coercion handles tuple/None) rather
        # than appending, to survive both tuple and Pydantic-style config objects.
        try:
            mnc = getattr(quantization_config, "modules_to_not_convert", None)
            if mnc is None:
                quantization_config.modules_to_not_convert = ["experts"]
                print("[Vaayu] FP8 compat: set modules_to_not_convert = ['experts']")
            elif "experts" not in mnc:
                quantization_config.modules_to_not_convert = list(mnc) + ["experts"]
                print(f"[Vaayu] FP8 compat: added 'experts' to modules_to_not_convert → {quantization_config.modules_to_not_convert}")
        except Exception as _e:
            print(f"[Vaayu] FP8 compat: FAILED to update modules_to_not_convert: {_e}")

        # 2. Config alias: some FP8 code accesses config.num_experts; MiniMax uses num_local_experts
        try:
            cfg = self.config.get_text_config() if hasattr(self.config, "get_text_config") else self.config
            if not hasattr(cfg, "num_experts") and hasattr(cfg, "num_local_experts"):
                cfg.num_experts = cfg.num_local_experts
        except Exception:
            pass

        # 3. deep-gemm v1 fails with "Unknown recipe" on some GPUs (e.g. Blackwell) for
        #    certain matrix shapes. Wrap w8a8_fp8_matmul to catch and fall back to Triton.
        self._patch_fp8_deepgemm_fallback()

    def _patch_fp8_deepgemm_fallback(self):
        """
        Patch w8a8_fp8_matmul to catch deep-gemm 'Unknown recipe' RuntimeErrors and
        automatically fall back to the Triton FP8 path. deep-gemm loads successfully on
        SM90+ (passes the capability check) but may have no compiled kernel recipe for
        the model's specific weight tile shapes (seen on Blackwell with deep-gemm v1).
        """
        try:
            import transformers.integrations.finegrained_fp8 as _fp8_mod
            
            # Proactively disable deep-gemm on Blackwell (SM120+) where deep-gemm v1
            # loads successfully but lacks compiled kernel recipes for matrix shapes.
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
                print("[Vaayu] FP8 compat: Blackwell GPU detected. Proactively disabling deep-gemm v1 to avoid 'Unknown recipe' crashes.")
                _fp8_mod._deepgemm_available = False
                _fp8_mod.deepgemm_fp8_matmul = None
                return

            _orig_w8a8 = _fp8_mod.w8a8_fp8_matmul

            def _w8a8_with_deepgemm_fallback(A, B, As, Bs, block_size, output_dtype=torch.float32):
                try:
                    return _orig_w8a8(A, B, As, Bs, block_size, output_dtype)
                except RuntimeError as e:
                    if "Unknown recipe" in str(e):
                        print("[Vaayu] deep-gemm 'Unknown recipe' — disabling deep-gemm, retrying with Triton")
                        _fp8_mod._deepgemm_available = False
                        _fp8_mod.deepgemm_fp8_matmul = None
                        return _orig_w8a8(A, B, As, Bs, block_size, output_dtype)
                    raise

            _fp8_mod.w8a8_fp8_matmul = _w8a8_with_deepgemm_fallback
            print("[Vaayu] FP8 compat: patched w8a8_fp8_matmul with deep-gemm 'Unknown recipe' fallback")
        except Exception as e:
            print(f"[Vaayu] FP8 compat: deepgemm fallback patch failed: {e}")

    def init_model(self):
        self._apply_transformers_compat_patches()
        self.model = None
        self._active_attn_impl = None

        if getattr(self, "_forced_attn_impl", None):
            attn_candidates = [self._forced_attn_impl]
        else:
            attn_candidates = []
            try:
                import flash_attn
                attn_candidates.append("flash_attention_2")
            except ImportError:
                pass
            attn_candidates.extend(["sdpa", "eager"])

        if self.get_use_better_transformer() and better_transformer_available:
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
                    self.model = BetterTransformer.transform(self.model)
                self._active_attn_impl = "BetterTransformer"
            except Exception:
                del self.model
                clean_memory()
                self.model = None

        if self.model is None:
            _last_exc = None
            for attn_impl in attn_candidates:
                try:
                    with init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(
                            self.config, attn_implementation=attn_impl, trust_remote_code=True
                        )
                    self._active_attn_impl = attn_impl
                    break
                except Exception as _e:
                    print(f"[Vaayu] attn_impl={attn_impl!r} failed: {_e}")
                    _last_exc = _e
                    if self.model is not None:
                        del self.model
                    clean_memory()
                    self.model = None

        if self.model is None:
            raise RuntimeError(
                "Failed to create model with any attention implementation"
            ) from _last_exc

        print(f"✅ Attention implementation: {self._active_attn_impl}")
        try:
            for mod in self.model.modules():
                if 'Attention' in type(mod).__name__:
                    print(f"   Attention module type: {type(mod).__name__}")
                    break
        except Exception:
            pass

        quantization_config = getattr(self.config, "quantization_config", None)

        if quantization_config is not None:
            quant_method = None
            if isinstance(quantization_config, dict):
                quant_method = quantization_config.get("quant_method")
            else:
                quant_method = getattr(quantization_config, "quant_method", None)
                
            if quant_method == "exl3":
                print("[Vaayu] Skipping AutoHfQuantizer for exl3 (managed by LoaderStrategy)")
            else:
                self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
                device_map = self.hf_quantizer.update_device_map(None)
                self._patch_fp8_quantizer_compat(self.hf_quantizer.quantization_config)
                self.hf_quantizer.preprocess_model(model = self.model, device_map = device_map)

        self.model.eval()
        self.model.tie_weights()

        self._path_remappings = {}
        self.set_layers_from_layer_names()

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer,
                                        dtype=self.running_dtype)

        if 'rotary_pos_emb' in self.layer_names_dict:
            # for glm keep rotary_pos_emb in gpu
            self.load_rotary_pos_emb_to_device()

    def diagnose_attention(self):
        print(f"Configured attention: {self._active_attn_impl}")
        if torch.cuda.is_available():
            print(f"Flash SDP available: {torch.backends.cuda.flash_sdp_enabled()}")
            print(f"Memory-efficient SDP available: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
            print(f"Math SDP available: {torch.backends.cuda.math_sdp_enabled()}")
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Compute capability: {torch.cuda.get_device_capability()}")

    def _resolve_and_map_path(self, target_path):
        import torch
        attr = self.model
        parts = target_path.split(".")
        current_path = ""
        resolved_full_path = ""

        # Attempt standard traversal
        success = True
        for part in parts:
            if hasattr(attr, part):
                attr = getattr(attr, part)
                current_path += ("." if current_path else "") + part
            else:
                success = False
                break
                
        if success:
            return attr

        # If it failed, search the model tree for the final node (e.g. embed_tokens, layers, norm, lm_head)
        target_tail = parts[-1]
        for name, mod in self.model.named_modules():
            if name.endswith(target_tail):
                # Ensure we avoid picking an audio/vision component arbitrarily if language is present
                if "audio" in name or "vision" in name:
                    continue
                # For layers, ensure it's a ModuleList
                if target_tail in ("layers", "blocks") and not isinstance(mod, torch.nn.ModuleList):
                    continue
                
                # Found a replacement! Map it so parameter injection succeeds later
                self._path_remappings[target_path] = name
                return mod
                
        # If absolute failure, return original root (will likely crash later, but preserves legacy fallback)
        return self.model

    def set_layers_from_layer_names(self):
        self.layers = []
        self.layers.append(self._resolve_and_map_path(self.layer_names_dict["embed"]))
        
        layer_list = self._resolve_and_map_path(self.layer_names_dict["layer_prefix"])
        if isinstance(layer_list, (list, tuple, torch.nn.ModuleList)):
            self.layers.extend(list(layer_list))
        else:
            self.layers.append(layer_list)

        self.layers.append(self._resolve_and_map_path(self.layer_names_dict["norm"]))
        self.layers.append(self._resolve_and_map_path(self.layer_names_dict["lm_head"]))

    def load_rotary_pos_emb_to_device(self):
        state_dict = load_layer(self.checkpoint_path, self.layer_names_dict['rotary_pos_emb'])
        self.move_layer_to_device(state_dict)

    # ── MoE detection ────────────────────────────────────────────────────────

    def _detect_moe(self) -> None:
        """Detect Mixture-of-Experts architecture from model config."""
        arch = (getattr(self.config, "architectures", None) or [""])[0]
        self.is_moe = (
            hasattr(self.config, "num_local_experts")
            or hasattr(self.config, "num_experts")
            or "Moe" in arch
            or "MoE" in arch
            or "Mixtral" in arch
        )
        if self.is_moe:
            self.moe_num_experts = getattr(
                self.config, "num_local_experts",
                getattr(self.config, "num_experts", 8)
            )
            self.moe_top_k = getattr(
                self.config, "num_experts_per_tok",
                getattr(self.config, "top_k", 2)
            )
            print(
                f"[Vaayu] MoE detected: {self.moe_num_experts} experts, "
                f"top-{self.moe_top_k} active per token"
            )
            print(
                "[Vaayu] Expert-aware streaming (Phase 3): pending — "
                "currently streams full MoE layer (conservative baseline)"
            )
        else:
            self.moe_num_experts = 0
            self.moe_top_k = 0

    # ── VRAM pinning ─────────────────────────────────────────────────────────

    def _get_vram_pinning_budget(self) -> int:
        """
        VRAM budget available for permanently pinned layers.

        Sourced from self._vram_plan.pinning_budget (vaayu.vram_plan.plan_vram),
        built once at load time from the same reserve the expert VRAM cache
        also subtracts (expert_vram_budget_for) -- see vram_plan.py's module
        docstring for why the two budgets must share one reserve number.

        Reads self._vram_plan directly, NOT via getattr: __init__ always sets
        it (to None or a real VramPlan), so a missing attribute here is a
        genuine bug, not a legitimate "no plan" state -- unlike
        _assert_context_fits's getattr, which is a diagnostic guard bound by
        the D3 rule that a diagnostic must never itself crash. A budget
        computation should fail loudly instead; do not "harmonize" the two.
        """
        if not torch.cuda.is_available():
            return 0
        if not str(self.device).startswith("cuda"):
            return 0
        return self._vram_plan.pinning_budget if self._vram_plan else 0

    def _assert_context_fits(self, n_tokens: int, N: int = 1) -> None:
        """Refuse an oversized request HERE, where the prompt length is known.

        Without this, the wall surfaces as an OOM inside the layer loop ~30
        frames away -- exactly the diagnostic failure D3 fixed for RoPE.

        A diagnostic guard, not a source of failure in its own right (D3): it
        must never itself be the thing that crashes. Returns quietly (does
        not raise) when there is no plan to check against -- CPU, CUDA
        unavailable, or _vram_shape somehow unset -- rather than raising an
        unrelated error in any of those cases.
        """
        plan = getattr(self, "_vram_plan", None)
        shape = getattr(self, "_vram_shape", None)
        if plan is None or shape is None:
            return
        if not torch.cuda.is_available():
            return
        import dataclasses

        from .vram_plan import (VramPlanError, _too_big_message,
                                 admission_headroom, predict_peak_vram)
        shape = dataclasses.replace(shape, N=int(N))
        try:
            free_vram, _ = torch.cuda.mem_get_info()
        except Exception:
            return
        # Compare against free_vram alone, NOT free_vram + resident.
        # mem_get_info() already reports free VRAM excluding this process's
        # own allocations (verified empirically), and predict_peak_vram's
        # terms are KV + activations + scores + logits -- there is no
        # model-weights term, so `need` is a TRANSIENT, not an absolute peak.
        # Adding resident back in would double-count it and pretend the
        # pinned weights/expert cache are not resident, inflating the
        # admitted budget by exactly that amount.
        #
        # ...but free_vram is not the budget either when a VRAM expert cache
        # exists: it is entitled to everything above the reserve and fills
        # LAZILY, during the very prefill being admitted. See
        # admission_headroom.
        # ...and mem_get_info() is a DRIVER-level call: it cannot see the
        # blocks PyTorch's caching allocator holds reserved-but-unallocated,
        # which are reusable by this very prefill. Measured on Qwen turn 2
        # (bug 11a): driver free 4.09 GiB with 1.57 GiB sitting in that pool,
        # refusing a request needing 4.88 that succeeds when admitted.
        try:
            reusable_pool = (torch.cuda.memory_reserved()
                             - torch.cuda.memory_allocated())
        except Exception:
            reusable_pool = 0      # D3: a diagnostic must never be the crash
        headroom = admission_headroom(
            free_vram=free_vram, plan=plan,
            expert_cache_active=getattr(self, "_expert_vram_cache", None) is not None,
            reusable_pool=reusable_pool)
        need = predict_peak_vram(shape, int(n_tokens))
        if need > headroom:
            # The floor the load-time plan actually reserved one of (an expert,
            # for MoE; nothing, for dense) -- NOT a literal 0, or a MoE refusal
            # prints "min_resident=0.00 GiB" even though one expert is resident.
            _moe_geo = getattr(self, "_moe_geometry", None)
            _min_resident = _moe_geo["expert_bytes"] if _moe_geo else 0
            raise VramPlanError(_too_big_message(
                requested=int(n_tokens), s_max=plan.s_max,
                free_vram=free_vram, shape=shape,
                min_resident=_min_resident, headroom=headroom))

    def _kv_cache_is_bucketed(self) -> bool:
        """Whether _create_kv_cache (below) would return a geometric-doubling
        ring buffer -- FP8DynamicCache or its TurboQuantKVCache subclass --
        rather than a plain transformers DynamicCache that holds exactly S
        tokens with no over-allocation (VramShape.kv_capacity_buckets,
        Amendment 1, task-6-brief.md). Mirrors _create_kv_cache's own mode
        resolution exactly, since the real cache is not constructed until
        first use but the VRAM plan (built at load time, see __init__) needs
        the answer before that. Not a guess: every branch here matches a
        branch in _create_kv_cache."""
        mode = self.kv_cache_mode
        if mode == "auto":
            from .persist.loader_strategy import EXL3LoaderStrategy
            mode = "bf16" if isinstance(self.loader_strategy, EXL3LoaderStrategy) else "fp16"
        if mode == "fp16":
            return False
        if mode == "bf16":
            return True
        if mode == "turboquant" or mode.startswith("turboquant:"):
            return True
        # fp8 (or a bare/unrecognised passthrough): FP8DynamicCache, unless
        # this PyTorch build lacks float8 support, in which case
        # _create_kv_cache falls back to a plain DynamicCache.
        return hasattr(torch, "float8_e4m3fn")

    def _create_kv_cache(self):
        """Return a KV cache instance sized and typed for the configured kv_cache_mode."""
        mode = self.kv_cache_mode

        if mode == "auto":
            # bf16, not fp8: fp8 mode stores an fp8 copy AND the dequantized
            # mirror attention actually reads, measured 3.016 bytes/element
            # against bf16's 2.000. bf16 is lossless, 33.7% smaller, and decodes
            # slightly faster on both on-device models (Plan E RESULTS). KV disk
            # persistence handles this layout too (QUANT_SIG_BF16, no scales).
            from .persist.loader_strategy import EXL3LoaderStrategy
            mode = "bf16" if isinstance(self.loader_strategy, EXL3LoaderStrategy) else "fp16"

        if mode == "fp16":
            return DynamicCache()

        if mode == "bf16":
            # Ring buffer without the fp8 store or the dequantized mirror:
            # 2.0 bytes/element vs 3.016 for fp8 mode, and lossless.
            print("[Vaayu] KV Cache: bf16 ring buffer (no quantization, ~33% "
                  "less VRAM than fp8 mode)")
            return FP8DynamicCache(quantize=False)

        if mode == "turboquant" or mode.startswith("turboquant:"):
            bits_total = int(mode.split(":", 1)[1]) if ":" in mode else 4
            # head_dim is not known until the first update(), so quote the
            # index rate only; the norms add 4 bytes per (token, head) on top.
            print(f"[Vaayu] KV Cache: TurboQuant, {bits_total} bits/channel "
                  f"({bits_total / 8:.3f} bytes/element of indices + 4 bytes per "
                  f"token-head of norms, vs bf16's 2.0). LOSSY — see the Plan H "
                  f"RESULTS quality gate.")
            return TurboQuantKVCache(bits_total=bits_total)

        if mode == "fp8":
            if not hasattr(torch, "float8_e4m3fn"):
                print("[Vaayu] FP8 KV cache unavailable (PyTorch < 2.1). Falling back to FP16.")
                return DynamicCache()

            print("[Vaayu] KV Cache: FP8 quantized (~50% VRAM savings)")
            return FP8DynamicCache()

        # H1: fp8 used to be the FALL-THROUGH, so any unrecognised mode string
        # -- a typo included -- silently got a different cache format than the
        # caller asked for. vram_plan.kv_elem_bytes already refuses the same
        # value, so accepting it here made the planner and the cache disagree
        # about the very same config. Refuse in both places, and say so the
        # same way.
        from .vram_plan import VramPlanError
        raise VramPlanError(
            f"Unknown kv_cache_mode {self.kv_cache_mode!r}. Known: "
            f"['auto', 'bf16', 'fp16', 'fp8'] or 'turboquant[:bits]'. "
            f"Refusing to guess -- an unrecognised mode used to fall through to "
            f"the fp8 cache silently."
        )

    def _init_pinned_layers(self) -> None:
        """
        Pre-load layers into VRAM using two-phase asymmetric pinning.

        Phase 1 pins shared layers (embed, norm, lm_head) with continue on budget miss.
        Phase 2 pins transformer layers 0…N with break on budget miss; only runs when
        max_pinned_layers != 0 (0 = shared-only, -1 = fill budget, N = first N layers).
        """
        budget = self._get_vram_pinning_budget()
        if budget <= 0:
            return

        layer_prefix = self.layer_names_dict["layer_prefix"]

        # Phase 1: shared layers — continue on budget miss (each has independent size)
        used = 0
        for layer_name in [self.layer_names_dict["embed"],
                           self.layer_names_dict["norm"],
                           self.layer_names_dict["lm_head"]]:
            try:
                tensors = self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)
                # A tied-weight checkpoint has no lm_head tensor; the split still
                # contains an empty-but-valid shard for it, so this load SUCCEEDS
                # with zero tensors (bug D1). Pinning that marks the layer resident
                # while it is still meta-initialized, and the forward then never
                # loads it -- the model emits uniform-random logits.
                if should_use_tied_lm_head(layer_name, self.layer_names_dict.get("lm_head"),
                                           tensors, load_failed=False):
                    print(f"[Vaayu-Pin] {layer_name} shard holds no tensors (tied weights); "
                          f"loading them from {self.layer_names_dict['embed']} instead.")
                    tensors = rekey_embed_as_lm_head(
                        self.loader_strategy.load_from_disk(
                            self.layer_names_dict["embed"], self.checkpoint_path),
                        self.layer_names_dict["embed"], layer_name)
                if not tensors:
                    # NOT `size == 0`: model.norm legitimately measures 0.00 GB.
                    print(f"[Vaayu-Pin] WARNING: {layer_name} yielded no tensors — refusing to "
                          f"mark it pinned. It would stay meta-initialized and emit garbage "
                          f"(CLAUDE.md §5: warn, never silently degrade).")
                    continue
                size = sum(t.nbytes for t in tensors.values())
                if used + size > budget:
                    print(f"[Vaayu-Pin] skip {layer_name}: budget exhausted "
                          f"({used / 1e9:.2f}/{budget / 1e9:.2f} GB)")
                    continue
                self.loader_strategy.inject_to_gpu(
                    self.model, layer_name, tensors,
                    self.running_device, self.running_dtype,
                    self.hf_quantizer, getattr(self, "_path_remappings", {}),
                )
                self.pinned_layer_names.add(layer_name)
                used += size
                print(f"[Vaayu-Pin] pinned {layer_name}: "
                      f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)")
            except Exception as e:
                print(f"[Vaayu-Pin] skip {layer_name}: {e}")

        # Phase 1.5: Pin MoE router gates permanently in VRAM (MoE models only).
        # Gates are tiny (< 4 MB per layer) but must be in VRAM before the patched
        # MoE forward() can run the router. Detection is header-driven: any tensor
        # key matching r'\.(gate|router)\.weight$' but NOT containing 'experts.' is
        # a router weight. The tighter regex avoids matching gate_proj (MLP projection).
        if self.is_moe:
            import json as _json
            import re as _re
            import struct as _struct
            _gate_re = _re.compile(r'\.(gate|router)\.weight$')
            for shard_name in self.layer_names:
                if not shard_name.startswith(layer_prefix + "."):
                    continue
                shard_path = Path(self.checkpoint_path) / (shard_name + ".safetensors")
                if not shard_path.exists():
                    continue
                try:
                    with open(shard_path, "rb") as _f:
                        _hdr_len = _struct.unpack("<Q", _f.read(8))[0]
                        _hdr = _json.loads(_f.read(_hdr_len))
                    gate_keys = {
                        k for k in _hdr
                        if k != "__metadata__"
                        and _gate_re.search(k)
                        and "experts." not in k
                    }
                    if not gate_keys:
                        continue
                    from safetensors import safe_open
                    gate_tensors = {}
                    with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                        for k in gate_keys:
                            if k in sf.keys():
                                gate_tensors[k] = sf.get_tensor(k)
                    if not gate_tensors:
                        continue
                    size = sum(t.nbytes for t in gate_tensors.values())
                    if used + size > budget:
                        print(f"[Vaayu-Pin] skip gate {shard_name}: budget exhausted")
                        continue
                    
                    if getattr(self, "_moe_key_remapper", None) is not None:
                        gate_tensors = self._moe_key_remapper.remap_state_dict(gate_tensors)
                        
                    self.loader_strategy.inject_to_gpu(
                        self.model, shard_name, gate_tensors,
                        self.running_device, self.running_dtype,
                        self.hf_quantizer, getattr(self, "_path_remappings", {}),
                    )
                    if not hasattr(self, '_moe_gate_pinned'):
                        self._moe_gate_pinned = set()
                    self._moe_gate_pinned.add(shard_name)
                    used += size
                    print(f"[Vaayu-Pin] pinned gate {shard_name}: "
                          f"{size / 1e6:.1f} MB  (cumulative {used / 1e9:.2f} GB)")
                except Exception as e:
                    print(f"[Vaayu-Pin] gate pin failed for {shard_name}: {e}")

        # Phase 2: transformer layers — break on budget miss (homogeneous size)
        if self.max_pinned_layers != 0:
            effective_max_pinned = -1 if self.is_moe else self.max_pinned_layers
            model_attr = self.model
            for attr_name in layer_prefix.split("."):
                model_attr = getattr(model_attr, attr_name)
            total_layers = len(model_attr)

            for i in range(total_layers):
                if not self.is_moe and effective_max_pinned > 0 and i >= effective_max_pinned:
                    break
                layer_name = f"{layer_prefix}.{i}"
                try:
                    if self.is_moe and getattr(self, "_expert_offset_index", None) is not None:
                        tensors = self._load_non_expert_strategy(layer_name)
                    else:
                        tensors = self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)
                    size = sum(t.nbytes for t in tensors.values())
                    if used + size > budget:
                        print(f"[Vaayu-Pin] budget exhausted at {layer_name} "
                              f"({used / 1e9:.2f}/{budget / 1e9:.2f} GB)")
                        break
                    self.loader_strategy.inject_to_gpu(
                        self.model, layer_name, tensors,
                        self.running_device, self.running_dtype,
                        self.hf_quantizer, getattr(self, "_path_remappings", {}),
                    )
                    _is_non_expert_pin = (
                        self.is_moe and getattr(self, "_expert_offset_index", None) is not None
                    )
                    if _is_non_expert_pin:
                        self.non_expert_pinned_layer_names.add(layer_name)
                    else:
                        self.pinned_layer_names.add(layer_name)
                    used += size
                    _pin_label = "non-expert-pinned" if _is_non_expert_pin else "pinned"
                    print(f"[Vaayu-Pin] {_pin_label} {layer_name}: "
                          f"{size / 1e9:.2f} GB  (cumulative {used / 1e9:.2f} GB)")
                except Exception as e:
                    print(f"[Vaayu-Pin] skip {layer_name}: {e}")

        pinned_xformer = sum(1 for n in self.pinned_layer_names
                             if n.startswith(layer_prefix + "."))
        pinned_shared = len(self.pinned_layer_names) - pinned_xformer
        non_expert_xformer = len(self.non_expert_pinned_layer_names)
        print(
            f"[Vaayu-Pin] {len(self.pinned_layer_names)} fully-pinned "
            f"({pinned_xformer} transformer + {pinned_shared} shared), "
            f"{non_expert_xformer} non-expert-pinned transformer, "
            f"{used / 1e9:.2f} GB reserved"
        )

    def _prepopulate_ram_cache(self) -> None:
        """Load cacheable layers from NVMe into RAM before the first inference token.

        Runs synchronously in __init__ (no inference, no DDR4 contention). Temporarily
        disconnects the async cache path on loader_strategy so load_from_disk returns
        plain tensors; put_sync() clones them inline and stores in the cache. Reconnects
        after all layers are loaded. Token 1 then gets pure cache hits with no clone
        overhead and no pinned-buffer race.

        For MoE models: skipped entirely. MoE inference uses _load_non_expert_strategy
        (80 MB per layer) which bypasses this cache, so pre-loading full 2.9 GB shards
        wastes ~84 GB of disk reads and ~8.7 GB of RAM with data never accessed.
        """
        if getattr(self, 'is_moe', False):
            print("[Vaayu-Cache] MoE model: skipping RAM cache pre-population "
                  "(expert cache handles active weights; non-expert tensors loaded on-demand)")
            return

        import time as _time
        t0 = _time.perf_counter()

        # Transformer layers not already pinned to VRAM
        prefix = self.layer_names_dict["layer_prefix"]
        cacheable = [
            name for name in self.layer_names
            if name not in self.pinned_layer_names and name.startswith(prefix)
        ]

        # Disable async cache path so load_from_disk returns raw tensors without
        # submitting background clones that would race with put_sync below.
        self.loader_strategy._cache = None
        n_cached = 0

        for layer_name in cacheable:
            if self.ram_cache.current_usage >= self.ram_cache.budget_bytes:
                break
            try:
                tensors = self.loader_strategy.load_from_disk(
                    layer_name, str(self.checkpoint_path))
            except Exception as e:
                print(f"[Vaayu-Cache] pre-pop skip {layer_name}: {e}")
                continue
            if self.ram_cache.put_sync(layer_name, tensors):
                n_cached += 1
            else:
                # put_sync failed = cache full; all remaining layers same size, stop loading.
                break

        elapsed = _time.perf_counter() - t0
        cached_gb = self.ram_cache.current_usage / 1e9
        print(f"[Vaayu-Cache] pre-populated {n_cached} layers "
              f"({cached_gb:.1f} GB) in {elapsed:.1f}s")
        # loader_strategy._cache is reconnected by the caller immediately after return

    def _warmup_nvme(self) -> None:
        import threading

        def _do_warmup():
            warmup_count = min(10, len(self.layer_names))
            warmup_names = self.layer_names[:warmup_count]

            # Use plain file reads — NOT io_uring_read_into.
            # The persistent io_uring ring is not thread-safe for concurrent SQE
            # submission; using it here races with inference and corrupts CQE counts.
            # Plain reads still touch NVMe blocks and prime the controller's internal
            # DRAM cache, which is the only goal.
            # Buffer sized from the actual files being warmed (CLAUDE.md §5: no
            # hardcoded model sizes), so any model's shards fit.
            _max_bytes = 0
            for _n in warmup_names:
                try:
                    _max_bytes = max(_max_bytes, os.path.getsize(
                        str(Path(self.checkpoint_path) / (_n + ".safetensors"))))
                except OSError:
                    pass
            if _max_bytes == 0:
                return
            buf = torch.empty(_max_bytes, dtype=torch.uint8)

            for name in warmup_names:
                path = str(Path(self.checkpoint_path) / (name + ".safetensors"))
                try:
                    with open(path, 'rb') as f:
                        f.readinto(buf.numpy())
                except Exception as e:
                    print(f"[Vaayu] Warm-up aborted on layer {name}: {e}")
                    break

            del buf
            print(f"[Vaayu] NVMe warm-up complete ({warmup_count} layers pre-read)")

        t = threading.Thread(target=_do_warmup, daemon=True)
        t.start()
        self._warmup_thread = t

    # ── Prefill lookahead stager ──────────────────────────────────────────────

    def _prefill_stage_worker(self) -> None:
        """Background thread (prefill only): reads EVERY expert of each MoE
        layer, in layer order, at most stage_depth layers ahead of the compute
        loop. During prefill routing touches ~all experts (P(unused) ≈ e^-27
        for a long prompt), so layer L+1's disk needs are known without its
        router — the consumer (sparse_forward) waits on the staged dict instead
        of issuing its own reads: zero duplicate I/O, disk never idles.

        Publishing protocol: self._prefill_staged[layer_idx] is written COMPLETE,
        then self._prefill_stage_events[layer_idx].set(). On any failure the
        event is still set (with a possibly-empty dict) so consumers fall back
        to their own reads instead of deadlocking.
        """
        from concurrent.futures import as_completed
        try:
            for _block in sorted(self._moe_blocks, key=lambda b: b.layer_idx):
                if self._prefill_stop.is_set():
                    return
                self._prefill_stage_sem.acquire()  # cap layers staged ahead
                _layer_name = _block.layer_name
                _shard = str(Path(self.checkpoint_path) / f"{_layer_name}.safetensors")
                _metas_by_e = self._expert_offset_index.index.get(_layer_name, {})
                _arena = self._expert_cache.arena if self._expert_cache is not None else None
                _staged: dict[int, dict] = {}
                _slabs: dict[int, "int | None"] = {}
                if getattr(self, "_overlap_trace", False):
                    print(f"[VAAYU-OVERLAP] stage_start layer={_block.layer_idx} "
                          f"t={time.perf_counter():.6f} n_experts={len(_metas_by_e)}")
                _futs = {
                    self._moe_executor.submit(stage_read_expert, _shard, _m, _arena): _e
                    for _e, _m in _metas_by_e.items()
                }
                _aborted = False
                for _f in as_completed(_futs):
                    if self._prefill_stop.is_set():
                        _aborted = True
                        for _g in _futs:
                            _g.cancel()
                        break
                    try:
                        _views, _sidx = _f.result()
                        _staged[_futs[_f]] = _views
                        _slabs[_futs[_f]] = _sidx
                    except Exception as _stage_err:
                        print(f"[Vaayu-MoE] prefill stage read failed "
                              f"layer={_block.layer_idx} expert={_futs[_f]}: {_stage_err}")
                if getattr(self, "_overlap_trace", False):
                    print(f"[VAAYU-OVERLAP] stage_end layer={_block.layer_idx} "
                          f"t={time.perf_counter():.6f}")
                if _aborted:
                    # Drain futures that were already running (cancel() only stops
                    # queued ones) and give every acquired slab back to the arena.
                    for _f, _e in _futs.items():
                        if _e in _slabs:
                            continue
                        try:
                            _views, _sidx = _f.result()
                            if _sidx is not None and _arena is not None:
                                _arena.release(_sidx)
                        except Exception:
                            pass
                    if _arena is not None:
                        for _sidx in _slabs.values():
                            if _sidx is not None:
                                _arena.release(_sidx)
                    return
                self._prefill_staged[_block.layer_idx] = _staged
                self._prefill_staged_slabs[_block.layer_idx] = _slabs
                self._prefill_stage_events[_block.layer_idx].set()
        finally:
            # Never leave a consumer blocked on an event that will not arrive.
            for _ev in self._prefill_stage_events.values():
                _ev.set()

    # ── Strategy helper ───────────────────────────────────────────────────────

    def _load_from_strategy(self, layer_name: str) -> dict:
        """Thin wrapper so ThreadPoolExecutor can call the strategy load."""
        return self.loader_strategy.load_from_disk(layer_name, self.checkpoint_path)

    def _load_non_expert_strategy(self, layer_name: str) -> dict:
        """For MoE layers: load only non-expert tensors (~44 MB instead of full layer).

        Results cached in _non_expert_tensor_cache (byte-budgeted LRU).
        After the first token, most non-expert reads are cache hits with no disk I/O.
        On RAM-constrained machines, LRU eviction keeps the working set warm.
        """
        cache = getattr(self, '_non_expert_tensor_cache', None)
        if cache is not None:
            cached = cache.get(layer_name)
            if cached is not None:
                return cached
        from .persist.safetensor_model_persister import load_non_expert_tensors
        from pathlib import Path as _Path
        file_path = str(_Path(self.checkpoint_path) / (layer_name + ".safetensors"))
        result = load_non_expert_tensors(file_path, layer_name, self._expert_offset_index)
        if cache is not None:
            cache.put(layer_name, result)
        return result

    def _load_tied_lm_head_from_embed(self, layer_name):
        """Load the embedding shard and re-key its tensors as lm_head's."""
        embed_layer_name = self.layer_names_dict['embed']
        # Sometimes layer names are passed with dots depending on the execution context, so strip it for comparison but fetch the exact embed equivalent
        if layer_name.endswith("."):
            embed_layer_name += "."
        load_layer_output = load_layer(self.checkpoint_path, embed_layer_name, self.profiling_mode)

        # Replace the 'model.embed_tokens' prefix with 'lm_head' in the returned state dict
        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
        else:
            state_dict = load_layer_output

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace(self.layer_names_dict['embed'], self.layer_names_dict['lm_head'])
            new_state_dict[new_k] = v

        if not new_state_dict:
            # Failing honestly beats degrading silently (CLAUDE.md §5): an
            # unweighted lm_head does not crash, it emits uniform-random
            # logits, which is exactly how bug D1 survived a smoke test.
            raise RuntimeError(
                f"Cannot supply weights for {layer_name}: its shard is empty or missing "
                f"(tied weights) and the embedding shard '{embed_layer_name}' yielded no "
                f"tensors either. The checkpoint split at {self.checkpoint_path} is incomplete.")

        if self.profiling_mode:
            return (new_state_dict, compression_time)
        return new_state_dict

    def load_layer_to_cpu(self, layer_name):

        t = time.time()

        # For models with tied weights, the lm_head shard carries no tensors
        # because lm_head is implicitly tied to embed_tokens. Safely fallback to
        # embed_tokens. Both shapes of that count -- an absent shard (raises) and
        # an empty-but-valid one (does NOT raise; see should_use_tied_lm_head).
        _lm_head_name = self.layer_names_dict.get('lm_head')
        try:
            load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        except Exception as e:
            if not should_use_tied_lm_head(layer_name, _lm_head_name, None, load_failed=True):
                raise e
            print(f"Failed to load {layer_name}, attempting to use tied weights from {self.layer_names_dict['embed']}. (original error: {repr(e)})")
            load_layer_output = self._load_tied_lm_head_from_embed(layer_name)
        else:
            _loaded = load_layer_output[0] if self.profiling_mode else load_layer_output
            if should_use_tied_lm_head(layer_name, _lm_head_name, _loaded, load_failed=False):
                print(f"{layer_name} shard holds no tensors (tied weights); loading them "
                      f"from {self.layer_names_dict['embed']} instead.")
                load_layer_output = self._load_tied_lm_head_from_embed(layer_name)

        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time

            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)

            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        return state_dict

    def move_layer_to_device(self, state_dict):
        layers = []
        
        # Apply MoE key remapping (checkpoint format -> live model format)
        if getattr(self, "_moe_key_remapper", None) is not None:
            state_dict = self._moe_key_remapper.remap_state_dict(state_dict)
        
        # Apply path remappings to state dict keys so parameter injection succeeds 
        # on architectures heavily wrapped in VLMs (e.g., Gemma4)
        remapped_state_dict = {}
        for param_name, param_tensor in state_dict.items():
            new_param_name = param_name
            for old_path, new_path in getattr(self, "_path_remappings", {}).items():
                if param_name.startswith(old_path):
                    new_param_name = param_name.replace(old_path, new_path, 1)
                    break
            remapped_state_dict[new_param_name] = param_tensor

        for param_name in remapped_state_dict:
            layers.append(param_name)

        _fp8_available = False
        if self.hf_quantizer is not None:
            try:
                from transformers.quantizers.quantizers_utils import get_module_from_name
                from transformers.integrations.finegrained_fp8 import FP8Linear
                _fp8_available = True
            except ImportError:
                pass

        for param_name in layers:
            tensor = remapped_state_dict[param_name]
            if self.hf_quantizer is not None and _fp8_available:
                try:
                    module, _ = get_module_from_name(self.model, param_name)
                    is_fp8_module = isinstance(module, FP8Linear)
                except Exception:
                    is_fp8_module = False
                target = tensor.to(self.running_device, non_blocking=True) if is_fp8_module \
                    else tensor.to(self.running_device, dtype=self.running_dtype, non_blocking=True)
            else:
                target = tensor.to(self.running_device, dtype=self.running_dtype, non_blocking=True)

            try:
                set_module_tensor_to_device(self.model, param_name, self.running_device, value=target)
            except Exception as e:
                print(f"[Vaayu] Warning: could not inject {param_name}: {e}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return layers

    # make GenerationMixin happy
    def can_generate(self):
        return True

    def set_experts_implementation(self, experts_implementation) -> None:
        # Transformers calls this to switch Mixtral between "grouped_mm" / "batched_mm"
        # expert dispatch during decode. Vaayu handles expert dispatch via moe_patch.py,
        # so this is intentionally a no-op.
        pass

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            # Capture past length BEFORE normalizing the cache object to None,
            # because Vaayu manages its own KV cache layer-by-layer and does not
            # consume the HF DynamicCache directly.
            past_length = self.get_past_key_values_cache_seq_len(past_key_values)

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_past_key_values_cache_seq_len(self, past_key_values):
        # transformers >= 4.36 returns a DynamicCache object instead of a tuple-of-tuples.
        # Use duck-typing first so this works even if cache_utils failed to import.
        if hasattr(past_key_values, 'get_seq_length'):
            return past_key_values.get_seq_length()
        if cache_utils_installed and isinstance(past_key_values, Cache):
            return past_key_values.get_seq_length()
        return past_key_values[0][0].shape[2]
    def get_sequence_len(self, seq):
        return seq.shape[1]

    def get_rotary_emb(self):
        """Return the model-level rotary embedding module if the architecture exposes one.

        Transformers >= 4.45 computes RoPE once in the top-level model forward and passes
        position_embeddings=(cos, sin) to every decoder layer.  For models that follow this
        pattern the module lives at model.model.rotary_emb.
        """
        try:
            return self.model.model.rotary_emb
        except AttributeError:
            return None

    def _compute_full_position_embeddings(self, position_ids):
        """Compute RoPE (cos, sin) for all positions directly from model config.

        Avoids using the meta-initialized rotary_emb module whose inv_freq buffer
        contains uninitialized CUDA memory after init_empty_weights, which would
        produce garbage / overflowed values and corrupt the hidden states.
        """
        try:
            # Honours partial rotary (rotary_dim < head_dim) — see
            # compute_rope_inv_freq. Emitting a full-head_dim cos/sin here makes
            # apply_rotary_pos_emb rotate dims meant to pass through untouched.
            inv_freq = compute_rope_inv_freq(self.config, position_ids.device)
        except AttributeError:
            return None

        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.no_grad():
            freqs = (inv_freq_expanded.float() @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(dtype=self.running_dtype)
            sin = emb.sin().to(dtype=self.running_dtype)
        return cos, sin

    def _precompute_rope(self, position_ids):
        """Return (cos, sin) for every position, or None if this architecture
        computes RoPE inside each decoder layer.

        None is a REAL answer, not a failure: models with no top-level
        rotary_emb expect `get_pos_emb_args` to pass nothing. Precompute
        failure must therefore raise rather than return None — every modern
        decoder layer takes `position_embeddings` as a required argument, so
        returning None on failure surfaces a VRAM exhaustion as
        `...DecoderLayer.forward() missing 1 required positional argument`
        roughly 30 frames away, naming the wrong subsystem (bug D3). The
        handler this replaced printed "RoPE will be computed per-layer"; no
        such per-layer path has ever existed.
        """
        if self.get_rotary_emb() is None:
            return None
        try:
            # Natively bypass init_empty_weights uninitialized buffer by creating a local un-mocked instance.
            # Guarantees perfectly matched dimensionality outputs specific to the framework (e.g. concatenated Llama vs chunked GptOss)
            rotary_emb_cls = type(self.get_rotary_emb())
            real_rotary = rotary_emb_cls(config=self.config).to(device=self.running_device, dtype=self.running_dtype)

            # Provide a dummy sequence tensor; many modern RoPE forwards require x purely for querying .device and .dtype
            dummy_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
            dummy_x = torch.empty(1, 1, position_ids.shape[1], dummy_dim, device=self.running_device, dtype=self.running_dtype)

            return real_rotary(dummy_x, position_ids)
        except Exception as e:
            print(f"Native RoPE instantiation failed ({e}), falling back to manual fallback.")
            try:
                return self._compute_full_position_embeddings(position_ids)
            except Exception as e2:
                # Free what we can so the caller's error path has room to run,
                # then fail honestly (CLAUDE.md §5) with the real cause chained.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Nothing in this message may itself raise: a diagnostic that
                # dies re-creates the very bug it reports (D3).
                _npos = position_ids.shape[1] if position_ids is not None else "unknown"
                raise RuntimeError(
                    f"Could not precompute RoPE position embeddings for "
                    f"{_npos} positions on {self.running_device}. Both the "
                    f"native rotary path (failed: {e}) and the manual computation (failed: "
                    f"{e2}) were exhausted. This is most often VRAM exhaustion — reduce "
                    f"prefill_chunk_size, max_seq_len, or the expert VRAM budget. There is no "
                    f"per-layer RoPE fallback to degrade to."
                ) from e2

    def get_pos_emb_args(self, len_p, len_s):
        # If position_embeddings were pre-computed for the full sequence (new transformers API),
        # slice out just the tokens being processed in this call.
        pos_emb = getattr(self, '_full_position_embeddings', None)
        if pos_emb is not None:
            cos, sin = pos_emb
            return {'position_embeddings': (cos[:, len_p:len_p + len_s], sin[:, len_p:len_p + len_s])}
        return {}

    def get_past_key_value_args(self, past_key_values):
        return {'past_key_values': past_key_values}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        if full_attention_mask is None:
            return {}
        return {'attention_mask': full_attention_mask[:, :, -len_s:, -len_p - len_s:]}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):

        return {'position_ids': full_position_ids[:, len_p:len_p + len_s]}


    def run_lm_head(self, layer, seq):
        return layer(seq).float()

    def run_norm(self, layer, seq):
        return layer(seq)

    def _chunked_prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        past_key_values=None,
    ):
        chunk_size = self.prefill_chunk_size
        total_len  = input_ids.shape[1]
        n_chunks   = (total_len + chunk_size - 1) // chunk_size
        print(f"[Vaayu] Chunked prefill: {total_len} tokens → {n_chunks} chunks of {chunk_size}")

        # Layer-major path (MoE): chunks become batch elements of ONE forward, so
        # every expert is read once per prefill instead of once per chunk (a
        # chunk-major 4k prompt otherwise re-reads the full expert set 8x).
        # Falls back to the legacy per-chunk loop for non-MoE, explicit-mask
        # attention, or hidden-state/attention outputs.
        if (getattr(self, 'is_moe', False)
                and self._active_attn_impl in ("flash_attention_2", "sdpa")
                and not output_attentions and not output_hidden_states):
            chunks = [input_ids[0, s:min(s + chunk_size, total_len)]
                      for s in range(0, total_len, chunk_size)]
            self._chunk_major_lens = [int(c.shape[0]) for c in chunks]
            self._chunk_major_n = len(chunks)
            try:
                output = self.forward(
                    input_ids=chunks,        # list of 1-D chunk tensors
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            finally:
                self._chunk_major_lens = None
                self._chunk_major_n = None
                # Defensive: if an exception unwound mid-layer, moe_patch.py's
                # last-chunk teardown never ran, leaving a stale per-eidx GPU
                # retention dict from an aborted layer. Clear it here too so a
                # later forward can't mistake it for the current layer's cache.
                self._chunk_major_gpu = None
                self._chunk_major_gpu_bytes = 0
            if not return_dict:
                return tuple(v for v in [
                    output.logits,
                    output.past_key_values if use_cache else None,
                    output.hidden_states,
                    output.attentions,
                ] if v is not None)
            if not use_cache:
                output.past_key_values = None
            return output

        output = None

        for start in range(0, total_len, chunk_size):
            end        = min(start + chunk_size, total_len)
            chunk_ids  = input_ids[:, start:end]
            chunk_mask = attention_mask[:, :end] if attention_mask is not None else None

            output = self.forward(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                past_key_values=past_key_values,
                use_cache=True,  # must accumulate KV cache across chunks regardless of caller's use_cache
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            past_key_values = output.past_key_values

            if torch.cuda.is_available():
                # RUNNING high-water mark across the whole prefill, not this
                # chunk's own peak: max_memory_allocated() is one PROCESS-GLOBAL
                # counter, and it is deliberately NOT cleared here. Clearing
                # it per chunk truncated every external measurement of a
                # chunked prefill to its final chunk -- which is how every dense point in
                # scripts/fit_vram_calibration.py's MEASURED table came to be a
                # last-chunk peak (llama-3.2-1b at S=32768: 2.409 GB recorded,
                # 8.13 GB actual). The counter belongs to whoever called
                # torch's peak-stats reset last; the engine must not touch
                # it. (Naming that function literally here would trip the
                # package-wide scan in tests/test_vram_peak_measurement.py.)
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[Vaayu] Chunk [{start}:{end}] done — peak VRAM: {peak:.2f} GB")

        if not return_dict:
            return tuple(v for v in [
                output.logits,
                output.past_key_values if use_cache else None,
                output.hidden_states,
                output.attentions,
            ] if v is not None)
        if not use_cache:
            output.past_key_values = None
        return output

    @torch.no_grad()
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        _exec_start = time.time()
        self.telemetry.begin_token(self._token_count)

        # Feature B: capture caller-supplied batched mask/positions BEFORE the
        # sdpa branch nulls them. When present, forward builds a per-stream 4D
        # padding+causal mask and uses per-stream positions (left-padded batch).
        _caller_attention_mask = attention_mask
        _caller_position_ids = position_ids

        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()

        if (self.prefill_chunk_size > 0
                and isinstance(input_ids, torch.Tensor)
                and input_ids.shape[1] > self.prefill_chunk_size):
            _past_kv_len = (self.get_past_key_values_cache_seq_len(past_key_values)
                            if past_key_values is not None else 0)
            if _past_kv_len == 0:
                return self._chunked_prefill(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        # Chunk-major prefill (Task 11): input_ids is a LIST of 1-D chunk tensors;
        # each becomes a batch element. KV appends per element within each layer,
        # so element j attends over chunks 0..j-1 + itself — same causal semantics
        # as the legacy per-chunk loop, but each layer's experts load only once.
        _chunk_lens = getattr(self, "_chunk_major_lens", None)
        if getattr(self, '_batch_decode_active', False):
            # Feature B: input_ids' leading dim is N independent streams (real
            # batch), not sequential chunks of one sequence — the per-row split
            # below would call layer() once per stream at batch=1 each, while
            # Task 5's mask/position tensors keep their real batch-N shape,
            # silently mismatching inside attention. Keep the whole N-batch
            # tensor as a single element so layer() is called once with its
            # true batch dimension intact.
            batch = [input_ids.to(self.running_device)]
        else:
            batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])
        _seq_total = (sum(_chunk_lens) if _chunk_lens
                      else (input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else 0))

        # Create attention mask for the largest input, and position ids to use KV cache
        if self._active_attn_impl in ("flash_attention_2", "sdpa"):
            if getattr(self, '_batch_decode_active', False) and _caller_attention_mask is not None and _caller_attention_mask.dim() == 2:
                from .batch import build_batch_attention_mask
                _q_len = input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else 0
                check_batch_kv_invariant(_caller_attention_mask, _q_len, past_key_values)
                attention_mask = build_batch_attention_mask(
                    _caller_attention_mask, _q_len,
                    dtype=self.running_dtype, device=self.running_device)
            else:
                attention_mask = None
        else:
            attention_mask = None
        _past_len_for_pos = (self.get_past_key_values_cache_seq_len(past_key_values)
                             if past_key_values is not None else 0)
        # Bug fix: must use _seq_total (already correctly sums ALL chunks for
        # layer-major list-typed input_ids), not re-derive from input_ids.shape
        # which silently evaluates to 0 for a list, undersizing the RoPE cache
        # to max_seq_len and crashing later chunks with an empty cos/sin slice.
        _pos_range        = compute_pos_range(self.max_seq_len, _past_len_for_pos, _seq_total)
        if self._active_attn_impl not in ("flash_attention_2", "sdpa"):
            # Sized from past+new tokens actually being processed, NOT from
            # _pos_range (which is max(max_seq_len, past+seq_total) -- since
            # max_seq_len is now DERIVED rather than a hardcoded 512, it is
            # frequently >> past+seq_total, and _pos_range never drops below it).
            # get_attention_mask_args only ever reads the bottom-right
            # full_attention_mask[:, :, -len_s:, -len_p-len_s:] block, so sizing
            # the mask any larger than the max (len_p+len_s) seen this forward is
            # pure waste -- and reusing _pos_range for it silently reintroduced
            # the exact 268 MB/12.9 GB transient this branch's comment claims to
            # avoid. _mask_size is that max: for a single (non-chunk-major) call
            # len_p+len_s == _past_len_for_pos+_seq_total exactly; for layer-major
            # chunked prefill, per-chunk len_p+len_s (via compute_chunk_past_lens)
            # is monotonically increasing and tops out at the same total on the
            # last chunk -- so no chunk's slice can ever exceed this size.
            _mask_size = max(1, _past_len_for_pos + _seq_total)
            attention_mask = torch.ones(_mask_size, _mask_size)
            attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
            attention_mask = attention_mask.to(self.running_device)
        # _pos_range is constant (== max_seq_len) unless the generation outgrows it;
        # allocating a fresh GPU arange every token is pure waste. Cache like the
        # RoPE block below.
        if getattr(self, '_batch_decode_active', False) and _caller_position_ids is not None:
            position_ids = _caller_position_ids.to(self.running_device)
        else:
            if getattr(self, '_position_ids_cache_key', None) != _pos_range:
                self._position_ids_cache = torch.arange(
                    _pos_range, dtype=torch.long, device=self.running_device)[None, :]
                self._position_ids_cache_key = _pos_range
            position_ids = self._position_ids_cache

        # Pre-compute RoPE position embeddings for the full sequence length.
        # Transformers >= 4.45 moved this computation to the top-level model forward;
        # each decoder layer now receives position_embeddings=(cos, sin) rather than
        # recomputing them internally.
        #
        # We gate on whether the architecture exposes a top-level rotary_emb (the
        # signal that it uses this new API), but compute the values from the model
        # config rather than calling the meta-initialized module.  After
        # init_empty_weights(), rotary_emb.inv_freq is an uninitialized CUDA tensor
        # (garbage values) which would produce NaN/overflow and corrupt hidden states.
        #
        # _pos_range is constant (== self.max_seq_len) for any generation that doesn't
        # outgrow max_seq_len, but this block used to recompute from scratch (new module
        # instance + H2D transfer + full forward) on every single token regardless. Cache
        # the result keyed on the values it was actually computed from; only recompute
        # when one of them changes (e.g. the rare case of outgrowing max_seq_len).
        _rope_cache_key = (_pos_range, self.running_device, self.running_dtype)
        if getattr(self, '_batch_decode_active', False) and _caller_position_ids is not None:
            # Per-stream positions: never use the shared-arange cache; compute
            # RoPE directly from [N, S] positions (handles the batch dim natively).
            self._full_position_embeddings = self._compute_full_position_embeddings(position_ids)
            self._rope_cache_key = None
        elif getattr(self, '_rope_cache_key', None) != _rope_cache_key:
            self._full_position_embeddings = self._precompute_rope(position_ids)
            self._rope_cache_key = _rope_cache_key

        if use_cache and past_key_values is None:
            past_key_values = self._create_kv_cache()
        all_hidden_states = [[] for _ in range(len(self.layers))] if output_hidden_states else None
        all_self_attns = [[] for _ in range(len(self.layers))] if output_attentions else None

        # Capture KV length once before the layer loop. len_p stays constant for the
        # full forward pass — all layers process the same positions. Re-querying
        # get_seq_length() inside the loop would return a growing value as each layer
        # appends its K/V to the cache, causing cos/sin slices to walk off the end of
        # _full_position_embeddings and produce empty tensors (silent position corruption).
        _len_p = past_key_values.get_seq_length() if past_key_values is not None else 0
        # Chunk-major: element j's effective past = past0 + lengths of chunks < j.
        _past_lens = compute_chunk_past_lens(_chunk_lens, _len_p) if _chunk_lens else None

        _io_wait_s = 0.0    # time blocked on future.result() / load_layer_to_cpu
        _gpu_load_s = 0.0   # time spent on move_layer_to_device (CPU→GPU transfer)
        _compute_s = 0.0    # time spent on actual layer.forward() GPU compute
        self._expert_disk_s = 0.0   # Fix #6: real expert NVMe time this token
                                    # (accumulated by sparse_forward in moe_patch)

        # MoE dispatch sub-timing (mask sync / disk load / H2D / object construct / GEMM),
        # gated behind profiling_mode so it's zero-cost when disabled.
        self._moe_profile = (
            {"mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0, "gemm": 0.0,
             "route": 0.0, "attach": 0.0, "detach": 0.0, "scatter": 0.0}
            if self.profiling_mode and self.is_moe else None
        )

        # MoE: precompute set of shard names that contain MoE experts so we can
        # filter expert tensors out of inject_to_gpu. sparse_forward handles experts.
        _moe_shard_names: set[str] = (
            {b.layer_name for b in self._moe_blocks} if self._moe_blocks else set()
        )
        if _moe_shard_names:
            import re as _re_fwd
            _expert_tensor_re = _re_fwd.compile(r'\.experts\.')
        else:
            _expert_tensor_re = None

        # Persistent executor: created once, reused every token. The old per-token
        # `with ThreadPoolExecutor(...)` paid thread create/join on every forward.
        # Only layer-weight load futures run here; each is consumed by
        # future.result() at its own layer, so no cross-token leakage.
        if getattr(self, "_fwd_executor", None) is None:
            _exec_workers = 2 if self.is_moe else 4
            self._fwd_executor = ThreadPoolExecutor(max_workers=_exec_workers)
            atexit.register(self._fwd_executor.shutdown, wait=False)
        executor = self._fwd_executor
        with torch.inference_mode():

            # Prefill flag: sparse_forward variants read this to skip cache puts
            # during prompt ingestion (prefill routes ~all experts once; caching
            # them is one full-model clone memcpy for content decode replaces).
            self._moe_prefill_pass = self.is_moe and compute_prefill_pass(
                seq_len=int(_seq_total),
                has_speculator=self.speculator is not None,
                draft_tokens=getattr(self, "draft_tokens", 0) or 0,
            )
            # First decode pass after prefill: reset cache stats once so decode
            # hit-rate telemetry excludes prefill's ~100%-miss pollution. (The old
            # guard here tested `_token_count == 0 and shape[1] == 1`, which is
            # unsatisfiable after prefill increments _token_count — it never ran.)
            _reset_now, self._decode_stats_reset = decode_stats_reset(
                is_moe=self.is_moe,
                has_expert_cache=self._expert_cache is not None,
                is_prefill_pass=self._moe_prefill_pass,
                already_reset=getattr(self, "_decode_stats_reset", False))
            if _reset_now:
                self._expert_cache.reset_stats()
                if getattr(self, '_expert_vram_cache', None) is not None:
                    self._expert_vram_cache.reset_stats()

            # Prefill lookahead pipeline: stream upcoming layers' full expert
            # sets while earlier layers compute. Long prompts only (short ones
            # don't route all experts — staging would over-read).
            self._prefill_stager_active = False
            # _moe_stager_capable: only the EXL3 fused forward consumes
            # _prefill_staged. For any other layout — including module-list,
            # which uses the arena but passes staged_layer=None — the stager
            # would read every expert of every layer into memory no consumer
            # ever reads: pure wasted I/O and RAM. See moe_supports_prefill_stager.
            if (self._moe_prefill_pass and self._moe_blocks
                    and getattr(self, "_moe_stager_capable", False)
                    and self._expert_cache is not None
                    and getattr(self, "_moe_executor", None) is not None
                    and compute_stage_gate(int(_seq_total), self._moe_geometry)):
                import threading as _threading
                _depth = compute_stage_depth(
                    self._expert_cache.budget_bytes, self._moe_geometry)
                self._prefill_staged = {}
                self._prefill_staged_slabs = {}
                self._prefill_stage_events = {
                    b.layer_idx: _threading.Event() for b in self._moe_blocks}
                self._prefill_stage_sem = _threading.Semaphore(_depth)
                self._prefill_stop = _threading.Event()
                self._prefill_stager_active = True
                self._staged_consume_count = {}
                _threading.Thread(
                    target=self._prefill_stage_worker,
                    daemon=True, name="vaayu-prefill-stager",
                ).start()
                print(f"[Vaayu-MoE] prefill stager: depth={_depth} layers ahead")

            # ── Seed prefetch: skip pinned layers (already in VRAM) ──────────
            future = None
            if self.prefetching:
                _all_pinned = self.pinned_layer_names | self.non_expert_pinned_layer_names
                for _seed_name in self.layer_names:
                    if _seed_name not in _all_pinned:
                        if _seed_name in _moe_shard_names and getattr(self, '_expert_offset_index', None) is not None:
                            future = executor.submit(self._load_non_expert_strategy, _seed_name)
                        else:
                            future = executor.submit(self._load_from_strategy, _seed_name)
                        break

            try:
                for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                                   desc=f'running layers({self.running_device})',
                                                   total=len(self.layers),
                                                   disable=(self._token_count > 0)):

                    _lt_file_bytes = 0
                    _lt_cache_hit = False
                    _lt_header_parse_us = 0
                    _lt_io_submit_us = 0
                    _lt_io_wait_us = 0
                    _lt_tensor_reconstruct_us = 0
                    _lt_gpu_transfer_us = 0
                    _lt_compute_us = 0

                    _non_expert_pinned = self.non_expert_pinned_layer_names
                    if layer_name in self.pinned_layer_names or layer_name in _non_expert_pinned:
                        # ── Pinned (full or non-expert): already in VRAM — zero disk/transfer cost ──
                        _lt_cache_hit = True
                        moved_layers = []
                        # If no future is outstanding, kick off the next non-pinned load
                        # so disk read overlaps with this layer's GPU compute.
                        if self.prefetching and future is None:
                            _all_pinned_fwd = self.pinned_layer_names | _non_expert_pinned
                            for _n in self.layer_names[i + 1:]:
                                if _n not in _all_pinned_fwd:
                                    if _n in _moe_shard_names and getattr(self, '_expert_offset_index', None) is not None:
                                        future = executor.submit(self._load_non_expert_strategy, _n)
                                    else:
                                        future = executor.submit(self._load_from_strategy, _n)
                                    break

                    elif self.prefetching:
                        # ── Non-pinned, prefetching ──────────────────────────────
                        if self.profiling_mode:
                            t = time.time()
                        _tl = time.perf_counter()
                        _t = time.time()
                        state_dict = future.result()
                        _io_wait_s += time.time() - _t
                        _lt_io_wait_us = int((time.perf_counter() - _tl) * 1e6)
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('load_safe_tensor_cpu_wait', elapsed_time)
                        _lt_fpath = str(Path(self.checkpoint_path) / (layer_name + ".safetensors"))
                        _lt_disk = _get_layer_disk_timings(_lt_fpath)
                        _lt_header_parse_us = _lt_disk.get("header_parse_us", 0)
                        _lt_io_submit_us = _lt_disk.get("io_submit_us", 0)
                        _lt_io_wait_us = _lt_disk.get("io_wait_us", _lt_io_wait_us)
                        _lt_tensor_reconstruct_us = _lt_disk.get("tensor_reconstruct_us", 0)
                        _lt_file_bytes = sum(v.nbytes for v in state_dict.values())

                        # Submit next non-pinned layer's load BEFORE GPU transfer
                        # so disk read and GPU H2D transfer overlap.
                        future = None
                        if self.profiling_mode:
                            t = time.time()
                        _all_pinned_elif = self.pinned_layer_names | self.non_expert_pinned_layer_names
                        for _n in self.layer_names[i + 1:]:
                            if _n not in _all_pinned_elif:
                                if _n in _moe_shard_names and getattr(self, '_expert_offset_index', None) is not None:
                                    future = executor.submit(self._load_non_expert_strategy, _n)
                                else:
                                    future = executor.submit(self._load_from_strategy, _n)
                                break
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('kick_off_load_cpu', elapsed_time)

                        if self.profiling_mode:
                            t = time.time()
                        _tl = time.perf_counter()
                        _t = time.time()
                        _sd = state_dict
                        if _expert_tensor_re is not None and layer_name in _moe_shard_names:
                            _sd = {k: v for k, v in state_dict.items()
                                   if not _expert_tensor_re.search(k)}
                        moved_layers = self.loader_strategy.inject_to_gpu(
                            self.model, layer_name, _sd, self.running_device, self.running_dtype,
                            self.hf_quantizer, getattr(self, "_path_remappings", {})
                        )
                        _gpu_load_s += time.time() - _t
                        _lt_gpu_transfer_us = int((time.perf_counter() - _tl) * 1e6)
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('create_layer_from_state_dict', elapsed_time)

                    else:
                        # ── Non-pinned, no prefetch ──────────────────────────────
                        _tl = time.perf_counter()
                        _t = time.time()
                        if layer_name in _moe_shard_names and getattr(self, '_expert_offset_index', None) is not None:
                            state_dict = self._load_non_expert_strategy(layer_name)
                        else:
                            state_dict = self.load_layer_to_cpu(layer_name)
                        _io_wait_s += time.time() - _t
                        _lt_io_wait_us = int((time.perf_counter() - _tl) * 1e6)
                        _lt_fpath = str(Path(self.checkpoint_path) / (layer_name + ".safetensors"))
                        _lt_disk = _get_layer_disk_timings(_lt_fpath)
                        _lt_header_parse_us = _lt_disk.get("header_parse_us", 0)
                        _lt_io_submit_us = _lt_disk.get("io_submit_us", 0)
                        _lt_io_wait_us = _lt_disk.get("io_wait_us", _lt_io_wait_us)
                        _lt_tensor_reconstruct_us = _lt_disk.get("tensor_reconstruct_us", 0)
                        _lt_file_bytes = sum(v.nbytes for v in state_dict.values())
                        if self.profiling_mode:
                            t = time.time()
                        _tl = time.perf_counter()
                        _t = time.time()
                        _sd = state_dict
                        if _expert_tensor_re is not None and layer_name in _moe_shard_names:
                            _sd = {k: v for k, v in state_dict.items()
                                   if not _expert_tensor_re.search(k)}
                        moved_layers = self.loader_strategy.inject_to_gpu(
                            self.model, layer_name, _sd, self.running_device, self.running_dtype,
                            self.hf_quantizer, getattr(self, "_path_remappings", {})
                        )
                        _gpu_load_s += time.time() - _t
                        _lt_gpu_transfer_us = int((time.perf_counter() - _tl) * 1e6)
                        if self.profiling_mode:
                            elapsed_time = time.time() - t
                            self.profiler.add_profiling_time('create_layer_from_safe_tensor', elapsed_time)

                    # Run layer

                    for j, seq in enumerate(batch):

                        if layer_name == self.layer_names_dict['embed']:
                            _tl = time.perf_counter()
                            batch[j] = layer(seq)
                            _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
                        elif layer_name == self.layer_names_dict['norm']:
                            #batch[j] = layer(seq[torch.arange(n_seq), batch_eos[j]][:, None])
                            _tl = time.perf_counter()
                            batch[j] = self.run_norm(layer, seq)
                            _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)

                            if output_hidden_states:
                                all_hidden_states[i].append(batch[j])
                        elif layer_name == self.layer_names_dict['lm_head']:
                            # Chunk-major: only the LAST chunk's logits are consumed
                            # (generate() reads [:, -1]); skip the vocab projection
                            # for earlier chunks.
                            if _chunk_lens is not None and j < len(batch) - 1:
                                continue
                            _tl = time.perf_counter()
                            batch[j] = self.run_lm_head(layer, seq)
                            _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
                        else:

                            if output_hidden_states:
                                all_hidden_states[i].append(seq)

                            len_s = self.get_sequence_len(seq)
                            _lp = _past_lens[j] if _past_lens is not None else _len_p

                            kwargs = {
                                'use_cache': use_cache,
                                **self.get_past_key_value_args(past_key_values),
                                **self.get_pos_emb_args(_lp, len_s),
                                **self.get_attention_mask_args(attention_mask, _lp, len_s),
                                **self.get_position_ids_args(position_ids, _lp, len_s),
                            }

                            _MAX_OOM_RETRIES = 2
                            layer_out = None
                            for _oom_retry in range(_MAX_OOM_RETRIES + 1):
                                try:
                                    if self.use_torch_compile:
                                        if self._compiled_layer_fn is None:
                                            self._compiled_layer_fn = torch.compile(
                                                lambda m, x, **kw: m(x, **kw),
                                                mode="reduce-overhead",
                                                fullgraph=False,
                                            )
                                        _tl = time.perf_counter(); _t = time.time(); layer_out = self._compiled_layer_fn(layer, seq, **kwargs); _compute_s += time.time() - _t; _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
                                    else:
                                        _tl = time.perf_counter()
                                        _t = time.time()
                                        layer_out = layer(seq, **kwargs)
                                        _compute_s += time.time() - _t
                                        _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
                                    break
                                except torch.cuda.OutOfMemoryError as _oom_err:
                                    if _oom_retry == _MAX_OOM_RETRIES:
                                        raise RuntimeError(
                                            f"CUDA OOM on {layer_name} after "
                                            f"{_MAX_OOM_RETRIES} non-destructive retries "
                                            f"(empty_cache + gc). The KV cache is NOT truncated "
                                            f"to recover — see below. Reduce prefill_chunk_size, "
                                            f"the batch width N, the context length, or the "
                                            f"expert VRAM budget."
                                        ) from _oom_err
                                    print(f"[Vaayu] ⚠️ OOM on {layer_name} (attempt {_oom_retry + 1}/{_MAX_OOM_RETRIES}). "
                                          f"Clearing cache and retrying...")
                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    # This handler used to halve the KV cache here
                                    # (`_oom_seq_len // 2` + _trim_kv_cache_front) and retry.
                                    # REMOVED 2026-08-09 -- it was the root cause of two
                                    # mask/KV mismatches, both landing on exact `// 2`:
                                    # F3 Task 6's unreproduced 1127-vs-1690 (563 = 1127 // 2)
                                    # and Plan K's 7770-vs-3885 at N=3 x 7.7k.
                                    #
                                    # It is not merely a shape bug. crop_front drops the
                                    # OLDEST tokens, so every survivor's position shifts
                                    # relative to both the caller's attention mask and the
                                    # RoPE positions already baked into the cached keys --
                                    # and the caller is never told, so its mask still
                                    # describes the full sequence. A run that "recovered"
                                    # and completed had silently answered from a truncated,
                                    # position-shifted context. Worse per-layer: the OOM
                                    # lands at layer i, so layers before it keep the
                                    # pre-crop length while later ones get post-crop plus
                                    # the retried chunk, leaving _seq_len ragged.
                                    #
                                    # Silently shrinking the user's context to survive is
                                    # the degradation CLAUDE.md §5 forbids, and D3 already
                                    # settled that a VRAM wall must raise and name the real
                                    # levers rather than fabricate a recovery.
                            new_seq = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                            if output_attentions and isinstance(layer_out, tuple) and len(layer_out) > 1:
                                all_self_attns[i].append(layer_out[1])

                            batch[j] = new_seq

                    # VAAYU_TRACE_ACT: hidden-state health after this layer.
                    # Limited to the first two forward passes (prefill + first
                    # decode step) — enough to localize a defect, and keeps the
                    # log readable instead of n_layers x n_tokens lines.
                    if self._trace_act and self._token_count <= 1:
                        print(f"[VAAYU-ACT] tok={self._token_count} "
                              f"i={i:>3} {layer_name} {act_stats(batch[0])}",
                              flush=True)

                    if output_hidden_states:
                        all_hidden_states += (torch.cat(batch, 0),)

                    # Reclaim VRAM — skip for pinned layers (they stay resident)
                    if layer_name not in self.pinned_layer_names and layer_name not in self.non_expert_pinned_layer_names:
                        self.loader_strategy.free_from_gpu(
                            self.model, layer, moved_layers, self.hf_quantizer
                        )

                    self.telemetry.record_layer(LayerTiming(
                        layer_name=layer_name,
                        file_bytes=_lt_file_bytes,
                        header_parse_us=_lt_header_parse_us,
                        io_submit_us=_lt_io_submit_us,
                        io_wait_us=_lt_io_wait_us,
                        tensor_reconstruct_us=_lt_tensor_reconstruct_us,
                        gpu_transfer_us=_lt_gpu_transfer_us,
                        compute_us=_lt_compute_us,
                        cache_hit=_lt_cache_hit,
                    ))

                    # Prefill only: decode calls this every 32 tokens instead (below) —
                    # per-token empty_cache() defeats the CUDA caching allocator.
                    if self._token_count == 0 and i % 40 == 39:
                        clean_memory()
            finally:
                if getattr(self, "_prefill_stager_active", False):
                    self._prefill_stop.set()
                    self._prefill_stage_sem.release()  # unblock a waiting worker
                    self._prefill_stager_active = False
                    # Return every never-consumed staged slab to the arena.
                    # synchronize() first: in-flight async H2D from a partially
                    # consumed layer must finish before its pinned slab is reused
                    # by a CPU-side io_uring write (stream ordering doesn't cover
                    # that).
                    _leftover = getattr(self, "_prefill_staged_slabs", None)
                    if _leftover:
                        # Rebind to a fresh dict BEFORE iterating (single atomic
                        # attribute assignment under the GIL) so the worker
                        # thread's per-layer publish can no longer discover this
                        # dict via self._prefill_staged_slabs once iteration
                        # starts — avoids "dict changed size during iteration"
                        # from a concurrent unlocked publish.
                        self._prefill_staged_slabs = {}
                        _arena = self._expert_cache.arena if self._expert_cache is not None else None
                        if _arena is not None:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            for _layer_slabs in list(_leftover.values()):
                                for _sidx in _layer_slabs.values():
                                    if _sidx is not None:
                                        _arena.release(_sidx)

        self.telemetry.end_token()
        self._token_count += 1
        # Decode-phase memory hygiene, throttled: every 32 tokens, host-only —
        # empty_cache() would defeat the CUDA caching allocator (Task 2).
        if self._token_count % 32 == 0:
            clean_memory(cuda=False)
        # Chunk-major elements have unequal lengths — only the last chunk's
        # logits exist/matter (see lm_head skip above).
        logits = batch[-1] if _chunk_lens is not None else torch.cat(batch, 0)

        if output_attentions:
            all_self_attns = all_self_attns[0:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states[0:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)

        _exec_end = time.time()
        _total = _exec_end - _exec_start
        _cache_stats = self.ram_cache.stats() if getattr(self, 'ram_cache', None) is not None else "disabled"
        _expert_cache_stats = self._expert_cache.stats() if self._expert_cache is not None else "n/a"
        _expert_cache_delta = self._expert_cache.delta_stats() if self._expert_cache is not None else "n/a"
        _vram_stats = self._expert_vram_cache.delta_stats() if getattr(self, '_expert_vram_cache', None) else "n/a"
        _expert_disk_s = getattr(self, '_expert_disk_s', 0.0)
        _compute_only_s = max(0.0, _compute_s - _expert_disk_s)
        _other_s = _total - _io_wait_s - _gpu_load_s - _compute_s
        _la = getattr(self, "_lookahead_state", None)
        _la_str = (f"  lookahead_acc={_la['hits']}/{_la['total']}"
                   f"={_la['hits']/_la['total']:.0%}" if _la and _la["total"] else "")
        try:
            with open("/proc/self/status") as _rf:
                _rss_kb = next((int(_l.split()[1]) for _l in _rf if _l.startswith("VmRSS:")), 0)
        except OSError:
            _rss_kb = 0
        _rss_str = f"  rss={_rss_kb / 1e6:.2f}GB"
        print(f"[VAAYU-IO] token summary: total={_total:.1f}s  io_wait={_io_wait_s:.1f}s  expert_disk={_expert_disk_s:.1f}s  gpu_load={_gpu_load_s:.1f}s  compute={_compute_only_s:.1f}s  other={_other_s:.1f}s  cache={_cache_stats}  expert_cache={_expert_cache_stats}  this_token={_expert_cache_delta}  vram={_vram_stats}{_la_str}{_rss_str}")
        if self._moe_profile is not None:
            _mp = self._moe_profile
            print(f"[VAAYU-MOE] layer_totals: mask_sync={_mp['mask_sync']:.3f}s  disk_load={_mp['disk_load']:.3f}s  h2d={_mp['h2d']:.3f}s  construct={_mp['construct']:.3f}s  gemm={_mp['gemm']:.3f}s  route={_mp['route']:.3f}s  attach={_mp['attach']:.3f}s  detach={_mp['detach']:.3f}s  scatter={_mp['scatter']:.3f}s")

        if not return_dict:
            return tuple(v for v in [logits,
                                     past_key_values if use_cache else None,
                                     tuple(all_hidden_states) if all_hidden_states is not None else None,
                                     tuple(all_self_attns) if all_self_attns is not None else None] if v is not None)
        if self.profiling_mode:
            forward_elapsed_time = time.process_time() - forward_start
            forward_elapsed_time_wall = time.time() - forward_start_wall
            self.profiler.print_profiling_time()


            print(f"total infer process time(including all above plus gpu compute): {forward_elapsed_time:.04f}")
            print(f"total infer wall time(including all above plus gpu compute): {forward_elapsed_time_wall:.04f}")

            self.profiler.clear_profiling_time()


        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_self_attns is not None else None,
        )

    def _trim_kv_cache(self, cache, target_seq_len: int) -> None:
        """Trim KV cache tensors in-place to target_seq_len tokens."""
        if cache is None:
            return
        if isinstance(cache, FP8DynamicCache):
            # Ring buffer (Fix #8): the retained [0:target_seq_len) tokens are
            # already the valid data, so trimming the tail is a length-only
            # update — no tensor slicing/copy needed.
            cache.crop(target_seq_len)
        elif hasattr(cache, 'key_cache'):
            for i in range(len(cache.key_cache)):
                if len(cache.key_cache[i].shape) >= 3:
                    cache.key_cache[i] = cache.key_cache[i][:, :, :target_seq_len, :]
                    cache.value_cache[i] = cache.value_cache[i][:, :, :target_seq_len, :]

    def _trim_kv_cache_front(self, cache, target_seq_len: int) -> None:
        """Remove the oldest tokens from the KV cache to free VRAM."""
        if cache is None:
            return
        if isinstance(cache, FP8DynamicCache):
            # Ring buffer (Fix #8): valid data starts at index 0, so dropping the
            # oldest tokens requires shifting the retained window down to the
            # front before shrinking the length. Rare (OOM path only), so the
            # one copy-shift here is acceptable.
            cache.crop_front(target_seq_len)
        elif hasattr(cache, 'key_cache'):
            for i in range(len(cache.key_cache)):
                if cache.key_cache[i] is not None and len(cache.key_cache[i].shape) >= 3:
                    cache.key_cache[i]   = cache.key_cache[i][:, :, -target_seq_len:, :]
                    cache.value_cache[i] = cache.value_cache[i][:, :, -target_seq_len:, :]

    # A1 cross-turn prefix cache concurrency invariant: one in-flight request
    # per session id. Two concurrent generate() calls with the SAME session_id
    # would both check out / mutate the same cache object and corrupt each
    # other. Parallel fan-out MUST use distinct session ids. PrefixCacheStore
    # .lookup()'s checkout (Task 4) makes a second concurrent same-id call a
    # miss (full prefill) rather than a corruption, so the failure mode is a
    # perf loss, not wrong output.
    def generate(self, *args, session_id=None, **kwargs):
        self._batch_decode_active = False
        self._batch_n = 1

        # Fail fast here, where the prompt length is known (Task 8) -- before
        # this can surface ~30 frames deep as an OOM in the layer loop.
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            # kwargs.get(..., 0): every in-repo caller passes max_new_tokens
            # explicitly. A caller relying on self.generation_config's default
            # instead would under-count here -- HF's own merge of kwargs with
            # generation_config happens deeper inside super().generate(),
            # replicating it here was judged not worth the duplication for a
            # gap no current call site hits.
            self._assert_context_fits(input_ids.shape[-1] + kwargs.get("max_new_tokens", 0))

        pc = getattr(self, "_prefix_cache", None)
        if pc is None:
            result = super().generate(*args, **kwargs)
            self.telemetry.dump_json("vaayu_telemetry.json")
            return result

        # A1 cross-turn reuse. Batch=1 per call (parallel streams = separate
        # calls with distinct session ids).
        ids_list = input_ids[0].tolist()

        # device= so a compressed entry is rebuilt straight onto the GPU; the
        # store raises rather than handing back a packed container.
        kv, match_len = pc.lookup(session_id, ids_list, device=self.running_device)
        if kv is None:
            kv = self._create_kv_cache()
            print(f"[Vaayu-Prefix] MISS session={session_id} "
                  f"prompt_len={input_ids.shape[1]} — full prefill")
        else:
            match_len = self._adopt_prefix_hit(
                kv, match_len, input_ids.shape[1], session_id)
            print(f"[Vaayu-Prefix] HIT session={session_id} "
                  f"match_len={match_len}/{input_ids.shape[1]} — "
                  f"prefill only {input_ids.shape[1] - match_len} new positions")
        kwargs["past_key_values"] = kv

        resolved_sid = session_id if session_id is not None else _auto_session_id(ids_list)
        try:
            result = super().generate(*args, **kwargs)
        except BaseException:
            # lookup() above POPPED this entry. Without putting it back, an
            # aborted turn -- an SSE client disconnecting mid-stream, a
            # cancellation, an OOM -- drains the pool permanently.
            from .prefix_cache import restore_on_abort
            restore_on_abort(pc, resolved_sid, ids_list, kv, match_len)
            raise
        # `kv` was mutated in place through the forward passes -> holds the full
        # final KV.
        self._persist_and_store(pc, resolved_sid, kv, result[0].tolist())
        return result

    def _build_prefix_store(self, budget_bytes):
        """Construct the RAM prefix pool, honouring prefix_compress."""
        bits = getattr(self, "prefix_compress", None)
        if bits is not None and bits < 9:
            print(f"[Vaayu-Prefix] WARNING: prefix_compress={bits} is below the 9 "
                  f"bits/channel that passed the quality gate on both on-device "
                  f"models (see 2026-08-02-compress-on-evict-kv-RESULTS.md; 8 bits "
                  f"measured 0.0612 worst-layer attention-score error on MiniMax "
                  f"against a 0.05 bar). Restored prefixes will be measurably wrong.")
        # getattr, matching the construction this replaced: callers in tests
        # build partial objects.
        return PrefixCacheStore(budget_bytes,
                                auto_prefix_match=getattr(self, "auto_prefix_match", True),
                                compress_bits=bits)

    def _persist_and_store(self, pc, resolved_sid, kv, token_ids):
        """Persist the finished KV, THEN hand it to the prefix pool.

        Order matters and is locked by test. kv_persist reads kv.key_cache
        directly, while pc.store() may TurboQuant-compress that very same object
        (prefix_compress=). Persisting second would write packed indices into a
        file whose header says bf16 values -- silent garbage on restore.
        kv_persist.to_bytes() does its own .cpu(), so persisting from the
        still-on-GPU cache is fine.
        """
        if self._kv_writer is not None:
            # kv.get_seq_length(), not len(token_ids): the last sampled token is
            # never fed back through a forward pass (generation stops right after
            # sampling it), so it never went through cache.update() -- the cache's
            # true occupied length is exactly len(token_ids) - 1. Using the longer
            # value here indexes one position past what the cache actually holds.
            self._kv_writer.enqueue_append(
                resolved_sid, kv, self.config, token_ids, kv.get_seq_length())
        pc.store(resolved_sid, token_ids, kv)
        self.telemetry.dump_json("vaayu_telemetry.json")

    def _adopt_prefix_hit(self, kv, match_len: int, prompt_len: int,
                          session_id) -> int:
        """Take ownership of a prefix-cache HIT. Returns the clamped match_len.

        Every caller of PrefixCacheStore.lookup() owes the same three steps, and
        H7 was one caller doing only two of them: generate() cropped the on-disk
        record along with the cache, speculative_generate cropped only the
        cache. Both persist through _persist_and_store, whose enqueue_append
        packs `range(st["nrec"], length)` off the WRITER's own nrec -- so a
        missing truncate leaves the bytes between match_len and the stale nrec
        holding the PREVIOUS turn's data. A corrupted record, undetectable until
        restore. Sharing the sequence is the fix: a fourth caller cannot forget
        a step that is not theirs to remember.

        The session_id guard is deliberate, not an oversight to mirror: with
        session_id=None the store resolves an auto id by HASHING the prompt, so
        each turn is a different record and there is nothing to truncate --
        truncating would crop some unrelated session's file.
        """
        kv.to_device(self.running_device)
        if match_len >= prompt_len:
            match_len = prompt_len - 1   # leave >=1 position to run
        kv.crop(match_len)
        if self._kv_writer is not None and session_id is not None:
            self._kv_writer.enqueue_truncate(session_id, match_len)
        return match_len

    @torch.no_grad()
    def speculative_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 20,
        streamer=None,
        temperature: float = 0.0,
        session_id=None,
        stopping_criteria=None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Custom generation loop using speculative decoding.

        Replaces model.generate() when self.speculator is set.
        The drafter guesses DRAFT_TOKENS candidates; Vaayu verifies all in one
        forward pass and accepts the longest correct prefix.

        Returns:
            generated: (1, prompt_len + new_tokens) tensor of all token IDs
        """
        assert self.speculator is not None, "speculative_generate called with no speculator"

        # This loop honours NOTHING about sampling -- say so rather than hand
        # back greedy text under a sampled request's name (CLAUDE.md: warn, do
        # not silently degrade).
        _ignored = speculative_ignored_sampling(temperature, kwargs)
        if _ignored:
            print(f"[Vaayu-Spec] WARNING: speculative decoding is greedy; "
                  f"{', '.join(_ignored)} IGNORED for this generation. "
                  f"Load without drafter_model_path to use them.")

        # The drafter keeps its KV cache ACROSS draft rounds (Fix #9), which is
        # only correct within one generation: `_drafter_seen` is a length into
        # THIS input_ids. Carried into the next call it slices a different
        # sequence, and the drafter drafts from another prompt's context.
        # Measured: three identical greedy calls in one process returned three
        # different sequences; resetting made them byte-identical.
        self.speculator.reset()

        self._batch_decode_active = False
        self._batch_n = 1

        # Fail fast here too (Task 8), same entry-point policy as generate().
        # This loop is single-stream (N=1, the default); the per-step verify
        # forward routes draft_tokens+1 positions at once, but that transient
        # is far below a full prefill chunk and the final KV length is capped
        # by prompt_len + max_new_tokens regardless of how many draft/verify
        # rounds it takes to get there -- the same total this predicate checks
        # for generate().
        self._assert_context_fits(input_ids.shape[-1] + max_new_tokens)

        pc = getattr(self, "_prefix_cache", None)
        kv, ids_list = None, input_ids[0].tolist()
        if pc is not None:
            kv, match_len = pc.lookup(session_id, ids_list, device=self.running_device)
            if kv is None:
                kv = self._create_kv_cache()
            else:
                self._adopt_prefix_hit(
                    kv, match_len, input_ids.shape[1], session_id)

        # A HIT crops `kv` to match_len < prompt length. generate()'s HIT path
        # gets away with passing the untouched prompt because HF's own
        # generate() loop calls prepare_inputs_for_generation (above), which
        # slices input_ids down to what the cache doesn't already hold. This
        # loop calls forward() directly with no such step, so do it here: feed
        # only the tail beyond kv's current length, or forward() would
        # re-append the already-cached prefix on top of itself.
        prefill_ids = input_ids[:, kv.get_seq_length():] if kv is not None else input_ids

        # Step 1: Prefill — forward() auto-chunks if prefill_chunk_size > 0
        prefill_out = self.forward(
            input_ids=prefill_ids,
            past_key_values=kv,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = prefill_out.past_key_values
        # Seed the first new token from prefill logits
        first_token = prefill_out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([input_ids, first_token], dim=1)
        tokens_generated = 1

        if streamer is not None:
            # Put the PROMPT first, exactly as generate() does. A streamer built
            # with skip_prompt=True (what stream_generate always builds) swallows
            # its first put() and clears the flag; without this line the call it
            # eats is `first_token` -- the first GENERATED token -- so streamed
            # text and the id sink are both short by exactly one token.
            streamer.put(input_ids.cpu())
            streamer.put(first_token.cpu())

        eos_ids = self.config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_set = set(eos_ids) if eos_ids else set()

        if first_token.item() in eos_set or tokens_generated >= max_new_tokens:
            if streamer is not None:
                streamer.end()
            if pc is not None:
                resolved = session_id if session_id is not None else _auto_session_id(ids_list)
                self._persist_and_store(pc, resolved, kv, generated[0].tolist())
            return generated

        while tokens_generated < max_new_tokens:
            # This loop is hand-written, so transformers' criteria machinery
            # never runs here: check it ourselves or a cancelled request keeps
            # drafting and verifying with nobody left to read the tokens.
            if stopping_criteria is not None and bool(stopping_criteria(generated, None).any()):
                break

            # Step 2: Draft K tokens
            draft_ids, draft_state = self.speculator.draft(
                generated, past_key_values, num_tokens=self.draft_tokens
            )
            k = draft_ids.shape[1]

            # Step 3: Verify — one Vaayu pass over [last_token, d_0, ..., d_{k-1}]
            # past_key_values has prompt+accepted tokens; verify adds k+1 more
            past_len_before = past_key_values.get_seq_length()
            verify_input = torch.cat([generated[:, -1:], draft_ids], dim=1)
            verify_out = self.forward(
                input_ids=verify_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            verifier_logits = verify_out.logits   # shape: (1, k+1, vocab_size)
            past_key_values = verify_out.past_key_values

            # Step 4: Accept/reject longest correct prefix
            accepted_ids, num_accepted = self.speculator.verify_and_accept(
                generated, draft_ids, verifier_logits, draft_state
            )

            # Step 5: Trim verifier KV back to past_len + num_accepted
            # (verify pass grew cache by k+1; we only keep num_accepted new positions)
            self._trim_kv_cache(past_key_values, past_len_before + num_accepted)

            # Step 6: Sync drafter cache to match accepted length
            self.speculator.rollback_cache(draft_state, num_accepted)

            # Step 7: Extend generated sequence (cap at max_new_tokens)
            remaining = max_new_tokens - tokens_generated
            if num_accepted > remaining:
                accepted_ids = accepted_ids[:, :remaining]
                num_accepted = remaining
            generated = torch.cat([generated, accepted_ids], dim=1)
            tokens_generated += num_accepted

            if streamer is not None:
                streamer.put(accepted_ids.cpu())

            accept_rate_str = f"{num_accepted / k:.0%}" if k > 0 else "n/a (no match)"
            print(f"[Vaayu-Spec] drafted={k} accepted={num_accepted} "
                  f"total_new={tokens_generated}/{max_new_tokens} "
                  f"accept_rate={accept_rate_str}")

            if accepted_ids[0, -1].item() in eos_set:
                break

        if streamer is not None:
            streamer.end()
        if pc is not None:
            resolved = session_id if session_id is not None else _auto_session_id(ids_list)
            self._persist_and_store(pc, resolved, kv, generated[0].tolist())
        self.telemetry.dump_json("vaayu_telemetry.json")
        return generated

    @torch.no_grad()
    def batched_generate(self, requests, reclaim: bool = False):
        """Static multi-query batched decode. Yields (stream_id, token_id) live.
        requests: list[vaayu.batch.BatchRequest] (already admitted, or pass raw and
        call plan_batch first). MoE forward is unchanged; batching is this loop
        plus the gated mask/position path in forward (Task 5).

        Contract: callers must fully consume this generator (or explicitly
        call .close() on it). Abandoning it mid-iteration leaves the model in
        batched-decode mode (self._batch_decode_active/_batch_n) until Python
        garbage-collects the generator and runs the finally block below.
        generate()/speculative_generate() defensively reset both flags at
        entry so a leaked flag can't corrupt those call paths, but relying on
        GC timing here is still not clean design."""
        from .batch import (sample_step, batch_position_ids, batch_common_prefix,
                            broadcast_cache_rows, slice_cache_row,
                            prefix_store_row, resolve_matched_len,
                            batch_kv_cost, geo_from_model, _VRAM_SAFETY_BYTES)
        assert self._active_attn_impl == "sdpa", (
            f"batched_generate requires sdpa attention, got {self._active_attn_impl}")
        N = len(requests)
        device = self.running_device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        # Fail fast here, at the real batch width (Task 8): the load-time
        # plan is sized for N=1, and a fan-out of N streams multiplies KV,
        # activations and logits, so admission must re-evaluate rather than
        # assume the load-time answer still holds.
        self._assert_context_fits(
            max(len(r.prompt_ids) + r.max_new_tokens for r in requests), N=N)

        # Plan K: perform the reclaim this batch was ADMITTED under. Admitting
        # against reclaimable VRAM without reclaiming it just moves the OOM.
        #
        # ONCE, here at the batch boundary, never on the decode path below --
        # a per-step reclaim would re-evict the cache every token, which is
        # Plan H's mistake in a different costume (right mechanism, hot path).
        # Pinned by test.
        #
        # Runs before the prefix lookup's to_device() so the freed VRAM is
        # still free when the reused KV lands on the GPU; reclaiming after it
        # would evict experts to make room for bytes already allocated.
        # batch_kv_cost is the SAME expression plan_batch admitted against --
        # if these two diverge, admission is lying and the gap surfaces as an
        # OOM instead of a deferral.
        if reclaim:
            _vram = getattr(self, "_expert_vram_cache", None)
            if _vram is not None:
                _target = (batch_kv_cost(requests, *geo_from_model(self))
                           + _VRAM_SAFETY_BYTES)
                _freed = _vram.evict_until_free(_target)
                print(f"[Vaayu-Batch] reclaim: freed {_freed / 1e9:.2f} GB of expert "
                      f"VRAM cache for N={N} (free target {_target / 1e9:.2f} GB)")

        pc = getattr(self, "_prefix_cache", None)
        common_ids = batch_common_prefix(requests) if pc is not None else []
        kv, m = (pc.lookup(None, common_ids, device=device)
                 if common_ids else (None, 0))
        if kv is not None:
            # Every stream must keep >=1 token to run, same clamp generate()
            # applies at :3714 and for the same reason.
            m = min(m, min(len(r.prompt_ids) for r in requests) - 1)
        if kv is None or m <= 0:
            kv, m = None, 0
        else:
            kv.to_device(device)
            m = resolve_matched_len(kv, m)
            broadcast_cache_rows(kv, N)
            print(f"[Vaayu-Prefix] BATCH HIT n={N} match_len={m}/"
                  f"{len(common_ids)} — prefilling only the divergent tails")

        tails = [r.prompt_ids[m:] for r in requests]
        max_R = max(len(t) for t in tails)
        input_ids = torch.full((N, max_R), pad_id, dtype=torch.long, device=device)
        tail_mask = torch.zeros((N, max_R), dtype=torch.long, device=device)
        for i, t in enumerate(tails):
            input_ids[i, max_R - len(t):] = torch.tensor(t, device=device)
            tail_mask[i, max_R - len(t):] = 1
        pad_mask = torch.cat(
            [torch.ones((N, m), dtype=torch.long, device=device), tail_mask], dim=1)
        position_ids = batch_position_ids(pad_mask)

        _total_prompt = sum(len(r.prompt_ids) for r in requests)
        self._batch_prefix_stats = {
            "matched": m, "common": len(common_ids),
            "prefill_tokens": _total_prompt,
            "coverage": (N * m / _total_prompt) if _total_prompt else 0.0,
        }

        eos = self.config.eos_token_id
        eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        stop_sets = [set(r.stop_ids) | eos_set for r in requests]
        active = [True] * N
        produced = [0] * N

        # Feature B: only batched_generate ever sets this. It gates the padding
        # mask / per-stream positions / batched-RoPE branches in forward() (Task
        # 5) — without it, forward() silently falls through to the single-stream
        # path on a padded batch, producing wrong results. Must stay True for
        # every forward() call below (prefill AND decode loop) and must always
        # be cleared on exit, so it never leaks into a later single-stream call.
        self._batch_decode_active = True
        self._batch_n = N
        self.telemetry.reset_batch_overlap()
        past = None   # bound in finally's store-back guard; stays None if the
                       # generator is torn down before the prefill forward returns
        try:
            # Prefill, CHUNKED. Running the whole batch prompt as one forward
            # made long contexts unreachable at any N: measured on MiniMax,
            # N=1 fits at 4k tokens and OOMs at 7 768 (tried 5.79 GiB with
            # 5.13 free), N=3 fits at 2k and OOMs at 4k. The constraint is
            # prefill ACTIVATIONS, scaling with N x context — which is why no
            # admission policy could help there (Plan K Task 3).
            #
            # forward()'s own chunked path is NOT reusable here: its MoE fast
            # path turns chunks INTO batch elements, so it needs the batch
            # dimension this caller is already using. Hence chunking here,
            # where pad_mask and the per-stream positions already exist, with
            # forward's own chunking disabled for each call.
            #
            # Per chunk, with past = m + start and q = end - start:
            #   mask       pad_mask[:, :m + end]      width == past + q
            #   positions  position_ids[:, :m + end]  get_pos_emb_args slices
            #                                         [len_p : len_p + len_s],
            #                                         so positions are absolute
            #                                         from 0, never chunk-local.
            # check_batch_kv_invariant re-checks the mask/KV agreement on every
            # one of these calls.
            #
            # Cost, stated because it is real: the layer-major optimisation that
            # reads each expert once per PREFILL rather than once per chunk is
            # the very trick that cannot compose with a batch, so a batched
            # chunked prefill re-reads the expert set per chunk. That buys the
            # ability to run at all.
            _saved_chunk = self.prefill_chunk_size
            self.prefill_chunk_size = 0
            _step = _saved_chunk if _saved_chunk and _saved_chunk > 0 else max_R
            try:
                _acc = kv
                for _start in range(0, max_R, _step):
                    _end = min(_start + _step, max_R)
                    out = self.forward(
                        input_ids=input_ids[:, _start:_end],
                        attention_mask=pad_mask[:, :m + _end],
                        position_ids=position_ids[:, :m + _end],
                        past_key_values=_acc,
                        use_cache=True, return_dict=True)
                    _acc = out.past_key_values
                    # Bind after each SUCCESSFUL chunk, not before the loop: a
                    # chunk that raises must not leave `past` pointing at a
                    # cache the finally-block would then store back as if the
                    # prefill had completed. Binding it as we go still lets the
                    # store-back return the popped pool entry rather than lose
                    # it — store()'s crop plus lookup()'s resolve_matched_len
                    # make a short entry safe, an over-claimed one is what they
                    # exist to handle.
                    past = _acc
            finally:
                self.prefill_chunk_size = _saved_chunk
            logits = out.logits[:, -1, :]           # right edge aligned by left-pad

            next_tokens = torch.empty((N, 1), dtype=torch.long, device=device)
            for i in range(N):
                tok = sample_step(logits[i], requests[i].temperature, requests[i].top_p)
                next_tokens[i, 0] = tok
                yield (i, tok)
                produced[i] = 1
                if tok in stop_sets[i] or produced[i] >= requests[i].max_new_tokens:
                    active[i] = False

            while any(active):
                pad_mask = torch.cat(
                    [pad_mask, torch.ones((N, 1), dtype=torch.long, device=device)], dim=1)
                position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)
                out = self.forward(
                    input_ids=next_tokens, past_key_values=past,
                    attention_mask=pad_mask, position_ids=position_ids,
                    use_cache=True, return_dict=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                for i in range(N):
                    if not active[i]:
                        next_tokens[i, 0] = pad_id      # dead slot, masked; static HoL
                        continue
                    tok = sample_step(logits[i], requests[i].temperature, requests[i].top_p)
                    next_tokens[i, 0] = tok
                    yield (i, tok)
                    produced[i] += 1
                    if tok in stop_sets[i] or produced[i] >= requests[i].max_new_tokens:
                        active[i] = False
        finally:
            # Store-back lives in finally, not after the while loop: lookup()
            # already POPPED the pooled entry, and only store() puts one back.
            # batched_generate is a generator (its own docstring treats early
            # abandonment as real) -- a consumer `break` raises GeneratorExit
            # at the currently-suspended yield, which unwinds straight past a
            # post-loop store-back and permanently deletes the pool's best
            # entry. `past is not None` guards the case where teardown hits
            # before the prefill forward() call returns (nothing to store yet).
            if past is not None and pc is not None and common_ids:
                # Exactly ONE entry per batch regardless of N: writing N would
                # be the K=7-against-a-5-slot-pool configuration that measured
                # ZERO hits (capacity is a cliff, not a slope).
                # Row choice must be the LONGEST prompt, never row 0 -- see
                # prefix_store_row's docstring for why row 0 would silently
                # store padding as if it were the shared prefix.
                #
                # try/except Exception (never bare/BaseException -- GeneratorExit
                # must keep propagating): the cleanup below must run unconditionally
                # on exit, and a failure here must never replace whatever exception
                # is already unwinding through this finally (e.g. a CUDA OOM in
                # decode, where store()'s to_cpu pinned alloc can plausibly fail
                # again) -- that would be the exact diagnostic-laundering failure
                # this repo already fixed elsewhere (bug D3). Not re-raising means
                # the entry is simply lost, which lookup()'s pop already made true
                # the moment this batch started.
                #
                # That swallowing is why this failed SILENTLY for a whole sweep:
                # slice_cache_row used to clone on-device and OOM'd here at 7.7k
                # context, so every pool-on cell reported a 1.08x "hit" that was
                # a full re-prefill. The warning below is the only signal, and it
                # is worth reading whenever a prefix hit looks too small.
                try:
                    one = slice_cache_row(past, prefix_store_row(requests))
                    one.crop(len(common_ids))
                    pc.store(_auto_session_id(common_ids), list(common_ids), one)
                except Exception as e:
                    print(f"[Vaayu-Prefix] WARNING: batch store-back failed "
                          f"({e!r}) -- lookup() already popped this entry, so "
                          f"it is now LOST from the pool.")
            self._batch_decode_active = False
            self._batch_n = 1
            self.telemetry.dump_json("vaayu_telemetry.json")
