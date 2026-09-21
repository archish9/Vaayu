# MoE Expert-Aware Streaming

Vaayu's expert-aware streaming eliminates the primary bottleneck of running
Mixture-of-Experts (MoE) models on consumer hardware: loading all expert weights
when only 2 of N are ever used per token.

---

## 1. Problem Statement

For a MoE model like Kimi-K2.6 (256 experts, top-2) or Mixtral 8x7B (8 experts, top-2),
each layer shard on disk contains ALL expert weights. Vaayu's base streaming engine reads
the entire shard for every token, even though 99.2% of the data is unused.

| Model | Experts | top-k | Shard size | Useful bytes | Waste |
|-------|---------|-------|------------|--------------|-------|
| Mixtral 8x7B | 8 | 2 | ~1.3 GB | ~325 MB | 75% |
| Kimi-K2.6 | 256 | 2 | ~9.7 GB | ~77 MB | 99.2% |

**Latency math at 6.5 GB/s NVMe:**
- Without expert streaming: 594 GB ÷ 6.5 GB/s = **91.3 s/token**
- With expert streaming + 80% cache hit: ~3 GB ÷ 6.5 GB/s = **~0.5 s/token NVMe stall**

---

## 2. Architecture

### 2.1 Three-Layer Defense Against NVMe Stalls

```
Token N, MoE layer K:
                                                 
  Router (VRAM-pinned) → active_experts=[3,7]
         │
         ├─ ExpertRAMCache.get(K, 3) → HIT  (0ms, pure RAM read)
         ├─ ExpertRAMCache.get(K, 7) → MISS → load_expert_sparse() → io_uring read
         │                                    ~0.6s stall for 4 GB expert at 6.5 GB/s
         │
  Run expert FFN [3,7] on GPU
         │
  Write _expert_history[K] = [3,7]
         │
  Background thread: _prefetch_experts(K+1, _expert_history[K])
         ↑ overlaps with layer K+1 shared attention compute
```

**Layer 1: ExpertRAMCache** — LRU cache in RAM keyed by `(layer_idx, expert_idx)`.
70–90% of tokens reuse the same experts (MoE temporal locality), so most requests
are instant RAM reads.

**Layer 2: Speculative Prefetch** — After each MoE layer, the current token's
routing history predicts which experts the *next* MoE layer will need. These are
pre-loaded in background while GPU computes attention. ~80–95% prediction accuracy
in autoregressive generation.

**Layer 3: Sparse io_uring** — On cache miss, `load_expert_sparse()` issues a
precise byte-range read for only the active expert tensors using the existing
io_uring infrastructure. No full-shard read.

### 2.2 Monkey-Patch Approach

MoE blocks are patched at init time rather than restructuring the forward loop.
This avoids disk format changes and follows Vaayu's layered design principle.

Generic detection in `detect_moe_blocks()` finds any module containing both a gate Linear and an expert ModuleList — regardless of the MoE model architecture.

This covers Mixtral (`block_sparse_moe`), DeepSeek (`mlp`), Kimi, Qwen MoE,
and any architecture following the `.experts.N.` naming convention.

---

## 3. Components

### 3.1 ExpertRAMCache — `vaayu/persist/expert_cache.py`

```python
class ExpertRAMCache:
    def __init__(self, budget_bytes: int)
    def get(self, layer_idx: int, expert_idx: int) -> dict | None
    def has(self, layer_idx: int, expert_idx: int) -> bool
    def put(self, layer_idx: int, expert_idx: int, tensors: dict) -> None
    def stats(self) -> str
```

- Key: `(layer_idx, expert_idx)` tuple
- Eviction: LRU via `OrderedDict` — `popitem(last=False)` removes oldest
- `put()` clones tensors synchronously and returns shallow copy on `get()`
- Thread-safe: single `threading.Lock` guards all mutations
- Budget: `psutil.virtual_memory().available - 8 GB reserve`

### 3.2 ExpertOffsetIndex — `vaayu/persist/expert_offset_index.py`

```python
@dataclass
class ExpertTensorMeta:
    file_offset: int   # absolute byte offset from file start
    byte_length: int
    shape: list
    dtype: str         # safetensors format e.g. "BF16"

class ExpertOffsetIndex:
    def build_or_load(self) -> None
    def get_expert_offsets(self, layer_name: str, expert_idx: int) -> dict | None
    def moe_layer_names(self) -> list[str]
```

- Scans all `.safetensors` headers at init; builds `{layer: {expert: {key: meta}}}`
- Regex `r'\.experts\.(\d+)\.'` detects expert tensors across all architectures
- JSON cache at `<checkpoint>/vaayu_expert_offsets.json`
- Staleness: cache stale if any shard is newer than JSON (uses `max()` of shard mtimes)

### 3.3 load_expert_sparse — `vaayu/persist/safetensor_model_persister.py`

```python
def load_expert_sparse(
    file_path: str,
    expert_metas: dict[str, ExpertTensorMeta],
    use_direct_io: bool | None = None,
) -> dict[str, torch.Tensor]
```

- Groups adjacent byte ranges (gap ≤ 4096) into single io_uring call
- Falls back to `preadv` when io_uring unavailable
- Uses LOCAL buffer per call (NOT the shared double-buffer — avoids race with prefetch pipeline)
- Returns `{tensor_key: cloned_cpu_tensor}`

### 3.4 MoE Block Patcher — `vaayu/moe_patch.py`

```python
def detect_moe_blocks(model: nn.Module) -> list[MoEBlockInfo]
def patch_moe_blocks(model, expert_cache, offset_index, checkpoint_path,
                     top_k, device, dtype, vaayu_ref) -> list[MoEBlockInfo]
def _inject_expert_weights(expert_mod, raw, expert_idx, layer_name, device, dtype) -> bool
```

**Sparse forward closure** per MoE block:
1. Run pinned router gate → `unique_experts`
2. `ExpertRAMCache.get()` for each → load missing via `load_expert_sparse()`
3. **Phase A**: Inject ALL unique expert weights to device (once per forward, not per token)
4. **Phase B**: Token × top_k accumulation with correct batch handling (B dimension)
5. **Phase C**: Unload all injected experts back to meta
6. Write `vaayu_ref._expert_history[layer_idx] = unique_experts`

**Weight injection** (`_inject_expert_weights`): uses `re.sub(r'^.*?\.experts\.\d+\.', '', param_key)` to strip key prefix generically — works for all architectures without hardcoding module names.

**Routing weight normalization**: top-k softmax weights are divided by their sum so they sum to 1.0, matching actual MoE behavior.

**Direct assignment**: `moe_module.forward = sparse_fwd` (NOT `.__get__()`) — instance attribute assignment bypasses Python's descriptor protocol, so `sparse_fwd(hidden_states)` receives only `hidden_states` without a module self injection.

### 3.4b Whole-layer streaming for fused-3D checkpoints — `vaayu/moe_patch.py`

Some checkpoints do not store one tensor per expert. They store **one tensor per projection
per layer**, with the expert count as the leading axis (`...mlp.experts.gate_up_proj_blocks`
of shape `[num_experts, …]`), and their experts module is handed to a grouped-GEMM kernel
that does the routing **inside** the kernel — `gpt-oss-20b` (MXFP4) is the example on this
box. There is no per-expert delivery to make for such a model: the kernel wants every
expert's bytes whether one token routed to them or five hundred did.

`_make_sparse_moe_forward` therefore has a fourth branch, `_make_whole_layer_sparse_forward`,
chosen when **both** are true:

- the expert offset index holds whole-layer tensors for the layer
  (`ExpertOffsetIndex.fused_layer_tensors`), i.e. the CHECKPOINT is fused, and
- the live experts module carries fused parameters (`fused_param_experts()`).

Both conditions matter. A module-only test is not enough: `Qwen3NextExperts` also carries
fused parameters while its checkpoint stores one tensor per expert, and it must stay on the
per-expert forward.

Per visit the branch reads the layer's tensors with `load_fused_layer_tensors()`, hands them
to the quantizer's own weight conversions (`hf_quantizer.get_weight_conversions()` — empty
for a checkpoint that needs none, so this is inert elsewhere), calls the block's own forward
unchanged, and releases the layer's expert attributes afterwards.

**What this costs you, and it is not small.** The expert RAM cache, the VRAM expert cache and
the top-k budget (`DECODE_TOP_K`) all exist to exploit sparsity, and none of them applies
here — every token moves the whole expert set. On gpt-oss-20b that is 423.6 MB per layer,
10.17 GB per token, and the per-token telemetry line reports `expert_cache=0hit/0miss`.
In exchange the model runs in ~1.7 GB of VRAM instead of the ~13.8 GB a resident load needs.

**Attention note for this family:** a model whose attention carries *sinks* cannot use the
flash kernel shipped here unless the GPU is Hopper — the kernel raises *"S aux is currently
only supported for Hopper GPUs"* at the first token — so Vaayu skips flash for it and runs
eager attention, which is what stock transformers picks for it too.

### 3.5 Router Gate Pinning — `vaayu/vaayu_base.py`

`_init_pinned_layers()` Phase 1.5 (MoE only): scans shard headers for keys matching
`r'\.(gate|router)\.weight$'`. Loads using `safetensors.safe_open` (NOT `load_from_disk`)
to avoid loading the full shard. Injects permanently into VRAM. Tracks in `self._moe_gate_pinned`
(separate from `pinned_layer_names` to avoid inflating the pinned-layer counter).

### 3.6 Forward Loop — `vaayu/vaayu_base.py`

Three additions (non-MoE paths untouched):
1. `_expert_history.clear()` at top of each `forward()` call
2. After each MoE layer: read `_expert_history[current_idx]`, submit `_prefetch_experts(next_idx, pred)` to existing `ThreadPoolExecutor`
3. Token summary print includes `expert_cache=Nhit/Mmiss hit_rate=X%`

---

## 4. Initialization Sequence

```
VaayuBaseModel.__init__():
  1. _detect_moe()                         ← sets self.is_moe, .moe_num_experts, .moe_top_k
  2. _init_pinned_layers()                 ← Phase 1.5 pins router gates via safe_open (MoE only)
  3. ExpertOffsetIndex.build_or_load()     ← scan headers or load JSON cache
  4. ExpertRAMCache(budget_bytes)          ← allocate RAM cache
  5. patch_moe_blocks(model, ...)          ← replace MoE forward() methods
```

Dense models (is_moe=False): steps 2 (Phase 1.5), 3, 4, 5 are skipped entirely.

---

## 5. Debugging Guide

### Log Messages

| Log line | Meaning |
|----------|---------|
| `[Vaayu-MoE] MoE detected: N experts, top-K active per token` | Detection succeeded |
| `[Vaayu-Pin] pinned gate model.layers.N: X MB` | Router gate pinned to VRAM |
| `[Vaayu-MoE] Building expert offset index...` | First run — scanning headers |
| `[Vaayu-MoE] Expert offset index loaded from cache` | JSON cache hit |
| `[Vaayu-MoE] Expert RAM cache: X GB budget` | Budget allocated |
| `[Vaayu-MoE] N MoE blocks patched for sparse streaming` | Patching complete |
| `[VAAYU-IO] ... expert_cache=Nhit/Mmiss hit_rate=X%` | Per-token cache stats |
| `[Vaayu-MoE] warn: no offset for layer=... expert=N` | Expert not in index |
| `[Vaayu-MoE] prefetch failed ...` | Background prefetch error (non-fatal) |
| `[Vaayu-MoE] warn: multiple expert lists at ...` | Ambiguous MoE architecture |

### Verification Checklist

- [ ] `vaayu_expert_offsets.json` exists in checkpoint directory after first run
- [ ] Expert cache stats in token summary show hit_rate > 70% from token 2
- [ ] NVMe read volume per token drops vs baseline (monitor with `iostat -x 1`)
- [ ] Dense model produces identical output and timing as before
- [ ] `[Vaayu-MoE] N MoE blocks patched` count matches expected layer count

### Common Issues

**"No offset for layer=... expert=N"**
Delete `vaayu_expert_offsets.json` to force rebuild. Check that the shard file exists and contains `.experts.N.` tensor keys.

**Hit rate stays at 0%**
Check that `[Vaayu-Pin] pinned gate` log lines appear. Without pinned router, the MoE forward may error before routing.

**patch_moe_blocks finds 0 blocks**
`detect_moe_blocks()` couldn't find MoE structure. The model needs an `nn.Linear` named `gate` or `router` as direct child of the MoE block. Print `[(n, type(m).__name__) for n, m in model.named_modules()]` to inspect.

**"MoE model on non-CUDA device — expert-aware streaming disabled"**
Feature only activates on CUDA. CPU inference falls back to full-layer streaming.

---

## 6. Key Design Decisions and Why

**Why monkey-patch instead of modifying the forward loop?**
The forward loop in `vaayu_base.py` is model-agnostic. Modifying it for MoE would require architecture detection in the hot path. Monkey-patching lets each MoE block carry its own sparse logic, cleanly isolated.

**Why a separate ExpertRAMCache instead of reusing LayerRAMCache?**
LayerRAMCache is keyed by layer name string (whole layer). ExpertRAMCache is keyed by `(layer_idx, expert_idx)` tuple. Different granularity, different eviction semantics. Sharing would require a union key type and muddy the eviction policy.

**Why local buffer in load_expert_sparse instead of shared pinned buffer?**
The shared double-buffer is used by the main prefetch pipeline (`_load_pinned`). If `load_expert_sparse` reallocated it mid-read from a background thread, the prefetch pipeline's in-flight data would be corrupted. Local-per-call buffers eliminate the race entirely. Expert reads are small enough that allocation cost is negligible.

**Why direct assignment (`moe_module.forward = fn`) not `fn.__get__()`?**
`__get__` binding creates a bound method — Python injects the module instance as the first positional argument. Direct instance attribute assignment bypasses the descriptor protocol entirely, so the function receives only `hidden_states` as the first argument.

**Why separate `_moe_gate_pinned` set instead of `pinned_layer_names`?**
Adding `model.layers.0.gate` to `pinned_layer_names` inflates the `pinned_xformer` counter (which counts names starting with `layer_prefix + "."`) and creates a double-injection path if Phase 2 also pins the full shard. Separate tracking avoids both problems.
