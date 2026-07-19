# Vaayu KV Prefix-Cache Disk Persistence — Design Spec (2026-07-19)

**Status:** Approved design, ready for `superpowers:writing-plans`.
**Subsystem:** A of a 3-part roadmap (A = disk persistence tier; B = persistent
expert usage histogram; C = auto-sized VRAM expert cache). B and C get their own
spec+plan cycles and are independent of this one.
**Prior art / inspiration:** JustVugg/colibri `.coli_kv` (single-file C engine).
Colibri ships the same idea for an MLA model; this spec adapts the technique to
Vaayu's model-agnostic, FP8-KV, HuggingFace-attention stack.

---

## 1. Purpose & context

Vaayu is the inference engine for an autonomous coding agent doing loop-engineering
(repeated agentic iterations). It runs as a **persistent server**. Successive loop
iterations re-send largely overlapping context (same system prompt + mostly-unchanged
files, differing only in a small tail).

The **cross-turn KV prefix cache** (Feature A1, spec
`2026-07-12-vaayu-prefix-caching-design.md`, plan
`2026-07-12-vaayu-prefix-caching-plan.md`) reuses a session's KV across turns while
the server stays up: it holds per-session `FP8DynamicCache` entries in a byte-budgeted
pinned-RAM pool (`PrefixCacheStore`), moving an entry to GPU only while its request is
in flight. That cache is **lost when the process exits** — an agent-loop restart, a
crash, a deploy re-prefills every session from scratch.

**This spec adds a disk tier below that RAM pool** so sessions survive a full process
restart: append each session's compressed KV to disk crash-safely, and repopulate the
RAM pool from disk at startup. First hit after a restart costs zero re-prefill instead
of a full 50K-token prefill.

The tier model:

```
GPU        = in-flight only            (unchanged)
RAM pinned = between turns             (PrefixCacheStore — prerequisite, planned)
DISK       = across process restart    (THIS SPEC — Subsystem A)
```

## 2. Non-goals

- No change to the RAM `PrefixCacheStore` design, the LCP match logic, `crop`, or the
  engine's prefill/decode hot path. This tier hangs off the existing `store()` /
  startup hooks only.
- No shared-trunk-across-branches (that is A2, deferred; see the A1 spec §8).
- No paged / custom attention.
- No synchronous durability guarantee per turn — a crash may lose the most recent
  turn's append (the file stays coherent at the prior turn). Full per-turn durability
  is available as a config toggle (§7, `kv_persist_sync`) but is not the default.
- Not a general KV serialization format for interchange — internal, versioned,
  invalidated freely on any geometry/quant/schema change.

## 3. Global constraints

- **Ships dark.** `enable_kv_persist` defaults **False** → no disk I/O, behavior
  byte-identical to today. Requires `enable_prefix_cache=True` to have any effect (the
  disk tier sits under the RAM tier; with the RAM tier off there is nothing to
  persist).
- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** No model-number literals. Every
  geometry value in the file header and every record size is DERIVED from `self.config`
  (num_hidden_layers, num_key_value_heads, head_dim) and the live `FP8DynamicCache`
  buffer shapes/dtypes at persist time. The layout makes no assumption about
  GQA vs MLA vs dense — it serializes whatever per-layer buffers the cache holds, at
  their actual sizes. Unit tests parametrized over a Qwen-like, a MiniMax-like, AND an
  MLA-like synthetic config.
- **Budget is hardware/host-derived, not a model literal.** RAM repopulation budget
  reuses the RAM tier's `compute_prefix_cache_budget(free_bytes)`. Disk budget derives
  from free disk (or a host-configured cap), never from the model.
- **Thread-safe.** Parallel agent streams call `store()` concurrently; the disk writer
  is a single-worker executor, so appends serialize with no cross-session contention.
- **Graceful degradation.** Any disk error (full disk, IO failure, corrupt/mismatched
  file at load) → log + skip; the RAM tier and generation are unaffected. Never a
  correctness failure.

## 4. Architecture

Four responsibilities, one new module `vaayu/kv_persist.py` plus small hooks in
`PrefixCacheStore` and the engine startup path.

```
store(session_id, token_ids, kv)          startup
        │                                     │
        ▼                                     ▼
  enqueue (sid, len) ──► DiskWriter    scan persist_dir ──► KvDiskLoader
        (single worker)     │                                   │
                            ▼                                    ▼
                 append delta [disk_nrec,len)          validate header vs config
                 nrec-last crash-safe                  load newest-first ≤ RAM budget
                 to <sha1(sid)>.vaayu_kv                reconstruct FP8DynamicCache
                                                        insert into PrefixCacheStore
```

### Unit 4.1 — file format & size derivation (`kv_persist.py`)

Pure, config-driven helpers, unit-testable with no CUDA:

- `kv_header(config, cache) -> Header` — builds the header struct from config +
  live cache dtypes. Fields: `magic="VAAYUKV1"`, `schema_ver`, `num_layers`,
  `num_kv_heads`, `head_dim`, `kv_dtype_code`, `scale_dtype_code`, `quant_sig`,
  `vocab`, `nrec`.
- `kv_record_bytes(header) -> int` — bytes of one on-disk position record:
  `4 (tok i32) + num_layers * (2 * num_kv_heads * head_dim * kv_dtype_size
  + 2 * num_kv_heads * scale_dtype_size)`. All operands from the header. Matches
  exactly what the writer packs and the loader reads.
- `quant_sig` = stable hash of the FP8 quant scheme constants (dtype
  `float8_e4m3fn`, the `448.0` clamp, per-head-per-token scale layout) so any change
  to how `FP8DynamicCache.update` quantizes invalidates old files.

**On-disk record (per token position):**
```
[tok_id i32]
for L in layers:
    [ key_cache[L][:, :, pos, :]     as fp8   → num_kv_heads*head_dim * kv_dtype_size ]
    [ value_cache[L][:, :, pos, :]   as fp8   → num_kv_heads*head_dim * kv_dtype_size ]
    [ _key_scales[L][:, :, pos, :]            → num_kv_heads * scale_dtype_size ]
    [ _value_scales[L][:, :, pos, :]          → num_kv_heads * scale_dtype_size ]
```

**Not persisted:** `_k_out_cache` / `_v_out_cache`. They are derived
(`fp8.to(scale_dtype) * scale`, per `FP8DynamicCache.update`, `vaayu_base.py:342-343`)
and recomputed at load — halving file size and per-turn write bytes. This mirrors
colibri, which stores only the MLA latent + rope key and reconstructs k_nope/value.

Assumes batch dim B=1 per persisted cache (A1's per-session caches are B=1; the RAM
spec's `to_cpu` compacts to `[..., :used, :]`). The loader restores B=1; the plan
asserts B==1 at persist time and skips persistence (log) for any B>1 cache rather than
guessing a layout.

### Unit 4.2 — `DiskWriter` (single-worker, background)

- Owns a `ThreadPoolExecutor(max_workers=1)` (same rationale as `ram_cache.py`'s clone
  pool: serialize IO, don't fight for bandwidth, don't block the request thread).
- Per session tracks `disk_nrec` (positions already on disk) and a kept-open file
  handle (`fopen` once, append per turn, close at shutdown — colibri optimization #4).
- `enqueue_append(session_id, token_ids, len)` — coalesces per session (only the
  newest `len` matters); worker appends positions `[disk_nrec, len)`:
  1. Grow a reusable contiguous staging buffer to `kv_record_bytes` if needed.
  2. For each new position, pack tok_id + every layer's 4 buffers into the buffer,
     one write per record.
  3. `flush` (+ `fsync` if `kv_persist_sync`) the data.
  4. Overwrite the header `NREC` field, `flush` again. **nrec-last:** a crash between
     step 3 and 4, or mid-3, leaves the old NREC → file valid, missing only this turn.
- `enqueue_truncate(session_id, match_len)` — mirrors a RAM `crop` on prefix
  divergence: rewrite NREC to `match_len`, set `disk_nrec = match_len`; the next append
  writes the divergent tail over the stale records (offset math from `kv_record_bytes`).
- `drain()` — block until the queue is empty; called on clean shutdown.
- `remove(session_id)` — delete the file (explicit session end / disk-budget eviction).

### Unit 4.3 — `KvDiskLoader` (startup)

- `load_all(store, config, ram_budget_bytes) -> int` — scan `persist_dir` for
  `*.vaayu_kv`; sort by mtime descending (most-recent sessions first).
- For each file: read + validate magic and EVERY header geometry field + `quant_sig` +
  `schema_ver` against the current model. Mismatch → log
  `[KV] ignoring <file>: different model/quant/schema` and skip (do NOT delete —
  another build may own it).
- Bounds: `nrec` beyond a sane max context → skip.
- Accumulate into `ram_budget_bytes`; stop loading further files once the next file's
  reconstructed size would exceed budget (remaining files stay on disk; a later
  same-session request still finds nothing in RAM and full-prefills — acceptable, no
  correctness issue).
- Reconstruct an `FP8DynamicCache`: allocate per-layer buffers, read fp8 K/V + scales,
  recompute `_k_out/_v_out = fp8.to(scale_dtype) * scale`, set `_seq_len = nrec` per
  layer. Read the `token_ids` (one i32 per record). Insert into `PrefixCacheStore` as a
  normal entry keyed by the session id recovered from the filename mapping (see §6).
- Returns count of sessions restored; log `[KV] resumed N sessions (M tokens) from disk`.

### Unit 4.4 — integration hooks

- `PrefixCacheStore.__init__` gains an optional `disk_writer: DiskWriter | None`. When
  present:
  - `store(...)` (after RAM insert) calls `disk_writer.enqueue_append(sid, token_ids, len)`.
  - the divergence path that crops an entry calls `disk_writer.enqueue_truncate(sid, match_len)`.
  - the LRU eviction path calls `disk_writer.remove(sid)` ONLY when also over the DISK
    budget (RAM eviction alone must NOT delete the disk copy — that is the whole point).
  - `drop(sid)` calls `disk_writer.remove(sid)`.
- Engine (`vaayu_base.py`) `from_pretrained` / constructor: when
  `enable_kv_persist and enable_prefix_cache`, build the `DiskWriter`, wire it into the
  `PrefixCacheStore`, and run `KvDiskLoader.load_all(store, config, budget)` before
  serving. Register `disk_writer.drain()` on shutdown.

## 5. Data flow (worked example, restart)

- Server up. Turn 1: `store("s1", P+gen, kv1)` → RAM insert + background append writes
  50 020 records to `<hash(s1)>.vaayu_kv`, NREC=50020.
- Turn 2: prompt shares first 50 000 tokens → RAM `lookup` returns `(kv1, 50000)` →
  crop → truncate mirror rewrites NREC=50000 → forward 20 new + generate → append
  writes the divergent tail, NREC grows.
- **Process restarts.** Startup `KvDiskLoader` reads `<hash(s1)>.vaayu_kv`, header
  matches, reconstructs `kv1'` into the RAM pool.
- Turn 3: same session, shared prefix → RAM hit on the restored cache → zero
  re-prefill. Without this tier, Turn 3 full-prefills ~50K tokens.

## 6. Session-id ↔ filename mapping

Filename is `sha1(session_id).vaayu_kv` (avoids unsafe chars / length limits). The raw
`session_id` string is stored in the header region — length-prefixed, immediately
after the fixed header fields and before the records — so the loader recovers the exact
key to re-insert into the pool. The record area therefore starts at
`fixed_header_bytes + 4 + len(session_id)`; the writer computes this offset once per
open and the loader recomputes it after reading the id. The `NREC` field stays within
the fixed header (its offset is constant), so the nrec-last rewrite is unaffected. sha1
collisions are handled by comparing the stored id; on mismatch, skip the file.

## 7. Configuration surface

| Flag | Default | Effect |
|---|---|---|
| `enable_kv_persist` | `False` | Master switch. Off → no disk I/O, today's behavior exactly. |
| `kv_persist_dir` | `<cache_root>/kv` | Directory of per-session files. |
| `kv_persist_sync` | `False` | `True` → `fsync` after each append (full per-turn durability, slower). `False` → `flush` only (crash may lose the last turn; file stays coherent). |
| `kv_persist_budget_bytes` | derived from free disk | LRU-delete oldest session files when the directory exceeds this. Not a model literal. |

`enable_kv_persist=True` with `enable_prefix_cache=False` is a no-op + one WARNING at
startup (nothing to persist).

## 8. Concurrency & correctness

- One in-flight request per session id — the RAM tier's existing documented invariant.
  The disk writer is single-worker, so appends/truncates serialize globally; no two
  sessions' writes interleave.
- **Numerics.** The persisted fp8 bytes + scales are exactly the RAM tier's stored
  representation; reload dequantizes deterministically to the same `_k_out/_v_out`.
  Reloaded-then-continued generation is numerically identical to an uninterrupted
  session. A coherence test guards this (persist → fresh process → load → generate ==
  uninterrupted generate).
- **Crash safety.** nrec-last append: NREC is the commit point, written and flushed
  after the record bytes. A crash at any point leaves NREC pointing at a fully-written
  prefix of records → the file loads to the last committed turn.
- **Write failure isolation.** Disk full / IO error in the background worker → log,
  mark that session non-persistable for the run, continue. RAM tier and generation
  never see an error.

## 9. Model-agnostic derivation (§5 compliance)

Every size and header field traces to `config` + live cache shapes:

- `num_layers` ← `config.num_hidden_layers`; `num_kv_heads` ← `config.num_key_value_heads`
  (or the attribute the loader already reads at `vaayu_base.py:629-659,982`);
  `head_dim` ← `config.head_dim` (or derived `hidden_size / num_attention_heads` where
  head_dim is absent — same fallback the model uses).
- `kv_dtype_code` / `scale_dtype_code` ← the live `FP8DynamicCache` buffer dtypes.
- `kv_record_bytes` computed from those — no literal record size anywhere.
- No architecture string is matched; the serializer walks the cache's per-layer buffer
  lists. An MLA cache with different per-layer shapes serializes and reloads by the same
  code path (its shapes flow through the header). The MLA-like synthetic-config test
  proves this.

## 10. Testing strategy

**Unit (CPU, no CUDA; synthetic configs Qwen-like + MiniMax-like + MLA-like):**
- `kv_record_bytes` / `kv_header`: correct sizes for all three geometries; no literals.
- Record round-trip: write a synthetic cache → read back → fp8 bytes + scales
  byte-identical; recomputed `_k_out/_v_out` equal the originals.
- **nrec-last crash-safety:** write N turns, corrupt the file by truncating mid-record
  in the last turn, load → recovers the prior committed NREC, no exception.
- Header geometry/quant/schema mismatch → loader skips the file (each field flipped in
  turn).
- Truncate mirror: append to len L, truncate to M<L, append new tail → file loads to
  the post-truncate token sequence.
- `quant_sig` changes when the FP8 scheme constants change.

**Integration (may gate on CUDA, marked accordingly):**
- Coherence: persist a session → new engine instance → `load_all` → generate; assert
  numerically equal to an uninterrupted run on the same context.
- Ships-dark: `enable_kv_persist=False` → zero files created, output byte-identical to
  a run with the flag absent.
- Disk-budget LRU: exceed `kv_persist_budget_bytes` → oldest session files deleted,
  newest retained.

## 11. Deferred / future

- **io_uring append.** Vaayu has a Rust io_uring path; a batched-append via io_uring
  could replace the Python single-worker writer if disk write shows up as a
  turn-latency cost. NOTE: py-spy/ptrace profilers corrupt vaayu_rust's io_uring state
  (see memory `io_uring_cqe_overflow_bug`) — never profile the live writer with a
  ptrace tool. Deferred until profiling (non-ptrace) shows the write matters.
- **Compaction.** Long sessions accumulate truncated-over stale tail bytes only if
  truncate+regrow churns; a periodic rewrite-compaction is deferred until observed.
- **Shared-trunk (A2) on disk.** When A2 lands range-addressable trunks, the disk
  format's per-position records already support sub-range sharing; revisit then.
