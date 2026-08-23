# Speeding up Vaayu token generation with Rust

Written 2026-08-23. Self-contained: every number below is either read off this
repo's own recorded measurements (`todo.md`, `plans/settled_and_method_notes.md`) or derived from them in
one line of arithmetic, and each item says how to falsify it.

**Read this first.** Vaayu's MoE decode is **72-74 % disk-bound**. Rust cannot
make an EXL3 CUDA kernel faster and cannot make NVMe faster. So the only Rust
levers that can move the token rate at all are the ones that (a) close the gap
between the bandwidth the disk path *achieves* and the bandwidth the drive
*delivers*, or (b) stop Python from standing in the way of the I/O threads.
Everything else in this document is a rounding error, and is labelled as such.

---

## 0. The one number that decides everything

Recorded decode figures (archived rows 14 / 14-OLD in `plans/settled_and_method_notes.md`, measured on-device,
decode-only counters, 64 tokens):

| model | NVMe expert reads/tok | bytes/tok | decode | **achieved read bandwidth** |
|---|---|---|---|---|
| Qwen3-235B | 422 | 4.99 GB | 0.820 tok/s | **4.09 GB/s** |
| MiniMax-M2.7 | 295 | 2.62 GB | 1.501 tok/s | **3.93 GB/s** |

The box's NVMe peak, stated as a hardware constant in `todo.md`'s Environment section, is **~5.5 GB/s**.

So the disk path is running at **71-74 % of the drive's rate** while being the
thing that costs 72-74 % of the token. **That gap is the entire Rust
opportunity, and it is worth ~1.25-1.4x on MoE decode if it closes completely.**

    Qwen   0.820 tok/s at 4.09 GB/s  ->  ~1.10 tok/s at 5.5 GB/s
    MiniMax 1.501 tok/s at 3.93 GB/s ->  ~2.05 tok/s at 5.5 GB/s

Those are ceilings, not predictions. **Measure the achieved bandwidth first**
(§5). If a fresh measurement shows decode already sitting at ~5.4 GB/s, then
R1 and R2 below are refuted and this whole document closes with them —
per `todo.md`'s stop rule, do not re-scope to find a win.

Dense (Llama-3.3-70B, ~37 GB read per token, 0.155 tok/s = ~5.7 GB/s) is
already at the wall. **Rust has nothing to offer streamed dense.** Its lever is
speculative decoding, already built and measured at 5.57x (archived row 12; carried in `todo.md`'s CARRIED FORWARD).

---

## 1. Is it Python's garbage collector?

**There is no evidence in this repo that it is, and it is the cheapest thing on
this list to rule in or out — do that before writing a line of Rust.**

Why it is *plausible*:

* Each MoE layer forward allocates a burst of short-lived Python containers —
  the `raw` tensor dict, the `groups` dict from `_group_exl3_tensors_by_weight`,
  the `_contribs` list, one `Future` per expert miss, the views dict returned by
  `load_expert_into_slab`. At ~8 experts x 94 layers that is on the order of
  **10^4 container allocations per token**, all in gen-0.
* The *surviving* graph is enormous: the whole `nn.Module` tree, thousands of
  live tensor views held by `ExpertRAMCache`/`ExpertVRAMCache`/`SlabArena`, the
  offset index, the prefix pool. A gen-2 pass has to walk all of it.
* Default thresholds are `(700, 10, 10)`, so gen-2 runs roughly every 70,000
  gen-0 allocations — i.e. **every few tokens**, each time traversing the full
  resident graph.

Why it is probably *not the main story*: 26-28 % of the token is all that is
left after disk, and most of that is GPU kernel time and H2D. GC would have to
be eating a large share of a small slice.

**The test (30 minutes, no GPU risk, no Rust):**

```python
# in vaayu_base after the load completes, before the first forward
import gc
gc.collect()
gc.freeze()                 # move the permanent graph out of gen-2 scanning
gc.set_threshold(50_000, 50, 50)   # arm B: far fewer gen-0/gen-2 passes
```

Instrument it honestly rather than guessing:

```python
import gc, time
_gc = {"n": 0, "s": 0.0, "t0": 0.0}
def _cb(phase, info):
    if phase == "start": _gc["t0"] = time.perf_counter()
    else: _gc["n"] += 1; _gc["s"] += time.perf_counter() - _gc["t0"]
gc.callbacks.append(_cb)
```

Run 64 decode tokens, print `_gc["s"] / wall`. **That fraction is the entire
prize.** If it is under 2 %, GC is not the problem and this section closes.

Three constraints on how you measure it, all already paid for by this repo:

* **Never attach py-spy** (`MEMORY.md`, "py-spy Corrupts io_uring"): ptrace-based
  profilers corrupt `vaayu_rust`'s io_uring state. In-process instrumentation
  only.
* **Same-box, same-day A/B or it does not count** (`todo.md`, Environment).
* Run each arm under its own `systemd-run --user --scope -p MemoryMax=22G
  -p MemorySwapMax=0`, discarding each checkpoint's first touch.

**And note the shape of the answer: `gc.freeze()` is a Python fix, not a Rust
one.** Rust helps GC pressure only *indirectly*, by moving the per-token object
churn out of Python entirely — which is R2 below.

### The more likely Python-side limiter: the GIL, not the GC

The I/O threads (`_moe_executor`, `get_expert_pool_size()` workers) release the
GIL correctly inside `vaayu_rust` (`py.allow_threads`, verified in
`vaayu_rust/src/lib.rs`). But **everything on either side of that call holds
it**: the `_EXPERT_BUF_POOL.get()`, the `ExpertRAMCache` OrderedDict LRU
bookkeeping, the `.view(dtype).view(shape)` slicing, the future resolution — and
the main thread's forward loop is competing for the same GIL the whole time.

Telling GC and GIL apart: with `gc.disable()` in place, if the token rate does
not move but `sys.setswitchinterval(0.0005)` (or `0.02`) *does*, the contention
is the GIL. That is the signal that R2 is worth its cost.

---

## 2. Ranked Rust work

Ordered by (measured ceiling) / (effort). Each row states what kills it.

### R1 — Strip the hot-path `eprintln!` from `persistent_read` — **do this first**

`vaayu_rust/src/lib.rs` prints **two unconditional lines per read**, one at
submit and one at completion, on the decode hot path:

```rust
eprintln!("[vaayu_rust] persistent  ring={}  start  buf_index={:?} ...");
...
eprintln!("[vaayu_rust] persistent  ring={}  done  elapsed={:.3}ms", ...);
```

At Qwen's 422 reads/token that is **~850 unbuffered `write(2)` syscalls per
token**, and `eprintln!` takes a lock on stderr each time — a lock the I/O
threads contend on with each other. It is also why `todo.md` warns that logs run
**300 MB - 1.3 GB** and must be `grep -a`'d, never `cat`'d.

**The new server makes this materially worse.** `vaayu/server/logs.py`
redirects fd 2 into a pipe read by a Python pump thread that **takes the GIL**
for every chunk it tees. So as of the server work, every expert read now also
buys a pipe write plus GIL-contending Python thread wakeups, competing directly
with the forward loop.

* **Fix:** gate both lines behind an env check resolved once
  (`VAAYU_RUST_TRACE`) or the `log` crate at `trace` level.
* **Cost:** ~10 lines of Rust plus a rebuild.
* **Expected:** small on its own (order 0.5-1 % of the token) — **but it is a
  prerequisite for R2 and R3**, because you cannot honestly profile a path that
  logs twice per read, and the log volume itself perturbs the measurement.
* **Refuted if:** removing it changes neither tok/s nor the achieved-bandwidth
  number. Keep the change anyway; the log volume alone justifies it.

### R2 — `io_uring_read_many`: one submission for a whole layer's misses

**This is the flagship item — it is the one aimed squarely at the 4.09-vs-5.5
GB/s gap.**

Today one contiguous tensor group = one blocking `io_uring_read_into` call, made
from a `ThreadPoolExecutor` worker (`_dispatch_expert_misses` in
`vaayu/moe_patch.py`, `load_expert_into_slab` /`load_expert_sparse` in
`vaayu/persist/safetensor_model_persister.py`). Concurrency is
`get_expert_pool_size()` Python threads, each holding one ring's mutex across
its own `submit_and_wait`.

The consequence: **between a worker's completion and its next submission the
ring goes idle**, and that idle window is a Python future resolving, a
`_EXPERT_BUF_POOL.put()`/`get()`, LRU bookkeeping, and a GIL acquisition. With
~422 such windows per token, a few hundred microseconds each is the whole
missing 25-30 % of bandwidth.

**Proposal:** a `read_many(Vec<(fd_or_path, offset, len, buf_ptr)>)` pyfunction
that pushes every SQE for a layer's misses onto one ring and waits for all
CQEs in one `allow_threads` call. Python's role collapses from ~422
dispatches/token to **~94 (one per layer)**.

* **Effort:** moderate. The submission loop already exists; it becomes a loop
  over descriptors instead of over chunks of one descriptor. The Python side
  needs `_dispatch_expert_misses` to gather the layer's descriptor list before
  calling, instead of fanning out to the executor.
* **Ceiling:** if it closes the bandwidth gap entirely, Qwen 0.820 -> ~1.10
  tok/s, MiniMax 1.501 -> ~2.05 tok/s (§0).
* **Refuted if:** §5's probe shows decode already at ~5.4 GB/s, or if a
  prototype's achieved bandwidth does not move. **Close the row on that result;
  do not re-scope.**
* **Watch for:** the SQ is sized at `queue_depth` (32). A layer's descriptor
  list can exceed it — submit in SQ-sized waves inside the one call, and fix the
  `SQ full` leak first (Bug **B6** in `todo.md`; B3 — short reads counted as success — is the other prerequisite).

### R3 — Move the expert-cache miss path into Rust ("get-or-read")

`ExpertRAMCache.get` / `acquire_slab` / `put_slab` / `_evict_global_oldest`
(`vaayu/persist/expert_cache.py`) are Python `OrderedDict` LRU operations run
**~750 times per token** (94 layers x 8 experts), each holding the GIL while the
forward loop wants it.

Fold lookup + slab acquisition + read + admission into a single GIL-free Rust
entry point. The `SlabArena` is already one pinned allocation with integer slab
indices — Rust only needs the base pointer, the slab stride, and the
LRU/eviction state, none of which needs to be a Python object.

* **Effort:** high. This is the largest item here and it touches the eviction
  policy that archived row 14 tuned against measured hit rates. Do it **only**
  if §1's GIL test says contention is real.
* **Composes with R2:** together they turn "422 Python dispatches + 750 LRU
  operations" into "94 Rust calls".
* **Refuted if:** the GIL probe in §1 shows switch-interval changes do not move
  the token rate.

### R4 — Prefill repack memcpys (`gpu_repack_aligned`, `tensor_align=True`)

`load_expert_into_slab`'s aligned mode does a per-tensor `dst_slice.copy_(chunk)`
in Python, and `gpu_repack_aligned` does the same into a fresh buffer. This is
the **prefill stager's** path, not decode's (decode's miss path is already
zero-copy straight into the slab — see the function's docstring).

A Rust `memcpy` under `allow_threads` lets the repack overlap the H2D stream
instead of serialising against it. **TTFT lever, not a decode lever** — and
prefix caching already took 136-377x off TTFT, so this only bites on a cold
turn.

* **Effort:** low-moderate. **Priority:** low, for that reason.

### R5 — Things that are already Rust, or that Rust cannot help

* **Tokenizer / detokenizer** — HF `tokenizers` is already Rust. No win.
  (But see Bug B2 in `todo.md`: the streaming path re-tokenizes the *entire*
  output once per request purely to count tokens. Fixing that bug deletes the
  work outright — a strictly better outcome than making it fast.)
* **Sampling / top-k / top-p** — GPU tensors. Rust would add a D2H round trip.
* **EXL3 matmuls, the router GEMM, attention** — CUDA kernels. Out of scope.
* **Streamed dense decode** — already at ~5.7 GB/s, i.e. at the wall (§0).
* **`safetensors` header parse, offset index build** — load-time, once.

---

## 3. Free Python wins to take before any Rust

These are cheaper than every item in §2 and should be measured in the same
arm, so the Rust numbers are not credited with them.

1. **`gc.freeze()` after load** (§1). One line. Removes the permanent module
   tree and the cache view graph from every gen-2 traversal.
2. **Precompute the expert-tensor role table at load.**
   `_group_exl3_tensors_by_weight` and `_classify_expert_tensors`
   (`vaayu/moe_patch.py`) loop `_EXPERT_WEIGHT_ROLES` patterns over every tensor
   key **per expert, per layer, per token** — pure Python string matching for a
   mapping that is fixed the moment the checkpoint is opened. Build
   `{tensor_key: role}` once at load and index it.
3. **`EngineHandle.tool_format` is a property that re-scans the whole chat
   template string on every access**, and `generate()`/`stream()` touch it 2-3
   times per request. Compute once in `__init__`.
4. **Kill the `len(tokenizer(full))` re-tokenization** in
   `vaayu/server/engine.py` — it is both a bug (Bug B2) and per-request work.

---

## 4. What will NOT make tokens faster, stated so nobody re-derives it

* **Rewriting the decode loop in Rust.** The loop is `for layer in layers:
  layer(x)`. Its Python cost is ~94 dispatches per token against a ~1.2 s token.
* **Replacing PyTorch.** The compute is EXL3 CUDA kernels reached through
  torch; there is no Python arithmetic to remove.
* **`multiprocessing` instead of threads.** The pinned slab arena and the CUDA
  context are per-process. Crossing that boundary costs more than the GIL does.
* **Bigger `queue_depth` on the ring.** `persistent_read` already splits one
  ~11 MB expert read into `qd` (32) chunked SQEs, so a single read already fills
  the ring. The idle time is *between* reads, which is what R2 targets.
* **Raising the expert cache further.** Already done and closed — see
  archived row 14, ceiling now derived, 2.41x end-to-end, and the trade against the
  prefix pool is measured. Do not re-open it as a Rust item.

---

## 5. The probe that gates all of this

One arm. Run it before writing any Rust.

```
systemd-run --user --scope -p MemoryMax=27G -p MemorySwapMax=0 \
  ./venv/bin/python test_quick.py > /tmp/bwprobe.log 2>&1
```

Instrument, decode-only (reset counters after prefill), 64 tokens, per model:

1. **Achieved read bandwidth** = (bytes read from NVMe during decode) / (decode
   wall). Compare against ~5.5 GB/s. **This single number decides R2 and R3.**
2. **GC fraction** = `gc.callbacks` accumulated seconds / decode wall (§1).
3. **GIL sensitivity** = tok/s at `sys.setswitchinterval(0.0005)` vs `0.02`.
4. **Ring idle fraction** = with R1's trace *enabled deliberately for this one
   arm*, sum of `elapsed` across all `persistent_read` calls / (decode wall x
   pool size). `1 - that` is the idle window R2 is trying to reclaim.

Write the result into `todo.md` under the R-row it settles, verdict included —
**including a refutation**, which closes its row per the standing stop rule.
