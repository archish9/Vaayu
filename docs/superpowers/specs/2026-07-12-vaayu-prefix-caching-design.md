# Vaayu Cross-Turn KV Prefix Caching — Design Spec (2026-07-12)

**Status:** Approved design, ready for `superpowers:writing-plans`.
**Scope of this spec:** Feature A1 (cross-turn KV reuse per session). Feature A2
(shared-trunk-across-N-branches) is sketched as forward-looking phase 2 only — NOT
specified for implementation here.

---

## 1. Purpose & context

Vaayu is the inference engine for an autonomous coding agent doing loop-engineering
(repeated agentic iterations; the agent will be built on LangChain's deepagents
framework, not yet finalized). Vaayu runs as a **persistent server process** the
agent calls repeatedly. Successive loop iterations re-send **largely overlapping
context** (same system prompt + mostly-unchanged files, differing only in a small
tail).

Today Vaayu re-prefills the entire prompt every request. **A1 reuses the KV cache of
a previous turn**: on a new request, reuse the longest matching token prefix already
cached and prefill only the divergent tail. For a loop re-sending 50K tokens with a
small change, this collapses prefill to the changed portion — the single biggest TTFT
lever for this workload. (The already-planned B1 GPU-align staging makes the
*uncached* prefill fast; A1 makes most prefill *disappear*.)

**Why A1 first (small) and not the general shared-trunk build (A2, large):** Vaayu
uses **stock HuggingFace attention** (flash_attention_2 → sdpa → eager;
`vaayu_base.py:1106-1150`) over a **contiguous `FP8DynamicCache`** (`vaayu_base.py:218`,
subclass of HF `DynamicCache`; stores FP8, dequantizes to BF16 before attention). A1
reuses this stack unchanged — the trunk KV simply stays in the contiguous cache and
attention reads it normally. A2 (one trunk KV referenced by N parallel branches)
requires segmented/paged attention that stock flash/SDPA cannot do, so it is a large
custom-attention build deferred to phase 2.

## 2. Non-goals (A1)

- No shared-trunk-across-branches (each session keeps its own cache — that is A2).
- No paged attention / custom attention kernel.
- No offload of idle caches to RAM (phase 1.5 optimization — see §8).
- No change to routing, MoE expert streaming, or the decode hot path.
- No change to RoPE / position handling (the existing `past_length` logic covers it).

## 3. Global constraints

- **Ships dark.** Config flag `enable_prefix_cache` defaults **False** → behavior
  byte-identical to today. Feature is opt-in.
- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** No model-number literals. KV
  geometry (num_layers, num_key_value_heads, head_dim) comes from `self.config`
  (already read at `vaayu_base.py:629-659,982`). The cache byte budget is derived from
  hardware free memory, not the model. Unit tests parametrized over both a Qwen-like
  and a MiniMax-like config.
- **Thread-safe.** Parallel agent streams call the store concurrently; all shared
  state under a lock.
- **Graceful degradation.** Any miss / eviction / disabled flag falls back to full
  prefill (current behavior) — never a correctness failure.

## 4. Architecture

A persistent, byte-budgeted, thread-safe pool of per-session KV caches, consulted at
the start of each request and updated at the end. Four units:

```
request(session_id?, input_ids)
        │
        ▼
  PrefixCacheStore.lookup ──► (kv_or_None, match_len)
        │                         uses longest_common_prefix
        ▼
  engine: if match_len>0 → FP8DynamicCache.crop(match_len),
          past_key_values=kv, forward only input_ids[match_len:]
        │
        ▼
  decode … generate …
        │
        ▼
  PrefixCacheStore.store(session_id, full_token_ids, final_kv)
```

### Unit 4.1 — `longest_common_prefix(a, b) -> int`
Pure function. Returns the number of leading elements equal in two token-id lists.
No dependencies. Fully unit-testable in isolation. New file `vaayu/prefix_cache.py`.

### Unit 4.2 — `FP8DynamicCache.crop(length: int) -> None`
Method added to `FP8DynamicCache` (`vaayu_base.py:218`). Crops every layer's stored
FP8 key/value tensors AND their per-head/per-token scales to the first `length`
positions along the sequence dimension, in place. Mirrors HF `DynamicCache.crop`
semantics but for the FP8 + scales representation. After crop, the cache reports
`get_seq_length() == length` and a subsequent forward appends after it.

**Dependency:** the exact FP8 + scales tensor layout inside `FP8DynamicCache`
(`update`, `vaayu_base.py:277`). The plan must read that layout and crop each stored
tensor consistently (data + scales + any `_seen_tokens`/length counter HF tracks).

### Unit 4.3 — `PrefixCacheStore` (new `vaayu/prefix_cache.py`)
Thread-safe pool of cache entries. One responsibility: given a request, return the
reusable KV + match length; and persist a completed generation's KV.

**Entry:** `{session_id: str, token_ids: list[int], kv: FP8DynamicCache, kv_bytes: int}`.

**State:** `OrderedDict[session_id, Entry]` (LRU order), `current_bytes`, `budget_bytes`,
`auto_prefix_match: bool`, a single `threading.Lock`.

**Interface:**
- `__init__(budget_bytes: int, auto_prefix_match: bool = True)`
- `lookup(session_id: "str | None", input_ids: list[int]) -> tuple["FP8DynamicCache | None", int]`
  1. If `session_id` given and present in pool: `m = longest_common_prefix(entry.token_ids, input_ids)`; move entry to MRU; return `(entry.kv, m)`.
  2. Else if `auto_prefix_match`: scan all entries, pick the one with the greatest
     `longest_common_prefix(entry.token_ids, input_ids)`; if that max > 0 return that
     `(entry.kv, m)`; else `(None, 0)`.
  3. Else `(None, 0)`.
  - **Important:** `lookup` returns the entry's *live* kv object. The engine will
    `crop` it to `match_len` and continue using it as this request's cache, so the
    entry is effectively consumed/owned by the in-flight request (see §6 concurrency).
- `store(session_id: str, token_ids: list[int], kv: "FP8DynamicCache") -> None`
  - Compute `kv_bytes` from the cache's resident tensors. Insert/replace entry as MRU.
  - While `current_bytes > budget_bytes` and pool non-empty and the entry-to-evict is
    not the one just inserted: pop LRU, subtract its bytes, drop its kv (GPU refs
    released). If a single entry exceeds the whole budget, keep it (it is the active
    context) but the next `store` will evict others first.
- `drop(session_id: str) -> None` — explicit removal (e.g. agent ends a session).

**Budget:** `budget_bytes` derived by the caller from hardware free memory (a
`compute_prefix_cache_budget(free_bytes)` helper alongside the other `compute_*`
budget helpers in `vaayu_base.py`, following the pattern of
`compute_expert_budget` / `compute_vram_expert_budget`). Never a model literal.

### Unit 4.4 — Engine integration (`vaayu_base.py`)
- Constructor / `from_pretrained` accepts `enable_prefix_cache: bool = False`,
  `auto_prefix_match: bool = True`. When enabled, build `self._prefix_cache =
  PrefixCacheStore(compute_prefix_cache_budget(free_mem), auto_prefix_match)`.
- The generate entry point accepts an optional `session_id` (threaded through from the
  agent request; when absent, auto-match may still apply).
- **Before prefill** (in the prepare-inputs path around `vaayu_base.py:1804`):
  - `kv, match_len = self._prefix_cache.lookup(session_id, input_ids_list)` (guarded by
    `enable_prefix_cache`).
  - If `match_len == len(input_ids)`: set `match_len -= 1` (always leave ≥1 position to
    run so the forward has a token to process).
  - If `match_len > 0`: `kv.crop(match_len)`; set `past_key_values = kv`; run the
    forward over `input_ids[match_len:]` only; positions continue via the existing
    `past_length` logic (`vaayu_base.py:1810,1826-1830`).
  - Else: normal full prefill.
- **After generation completes:** `self._prefix_cache.store(session_id, full_token_ids,
  final_kv)` where `full_token_ids` = prompt + generated tokens and `final_kv` is the
  cache after the last decode step.

## 5. Data flow (worked example, loop iteration)

- Turn 1: request(session="s1", ids=[P(50000 tokens)]). Miss → full prefill →
  generate → `store("s1", P+gen, kv1)`.
- Turn 2: request(session="s1", ids=[P' where P'[:49900]==P[:49900], differs after]).
  `lookup` → `(kv1, 49900)`. `crop(49900)`, forward only the 100 changed + new tokens.
  TTFT ≈ prefill of ~100 tokens instead of 50000.

## 6. Concurrency & correctness

- All `PrefixCacheStore` state mutated under one lock. `lookup`/`store`/`drop` are the
  only entry points.
- **One in-flight request per session id.** A session's kv object is handed to the
  in-flight request and mutated (cropped, appended) by it. Two concurrent requests
  with the *same* session id would corrupt each other — the agent must not issue
  concurrent same-session requests (document this; parallel streams use *distinct*
  ids, which is the fan-out model). If defensive handling is wanted, `lookup` can mark
  an entry "checked out" and treat a second concurrent same-id lookup as a miss (full
  prefill). Decide in the plan; default: document the invariant.
- **FP8 numerics:** reusing FP8-stored trunk KV is numerically equivalent to
  re-prefilling it (same tokens, same weights, same deterministic FP8 quant). Risk is
  low, but a coherence test (reused vs fresh output on identical context) guards it.
- **Position correctness:** cropping to `match_len` makes `past_length == match_len`;
  the existing position-id logic offsets new tokens correctly. No RoPE change.

## 7. Configuration surface

| Flag | Default | Effect |
|---|---|---|
| `enable_prefix_cache` | `False` | Master switch. Off → today's behavior exactly. |
| `auto_prefix_match` | `True` | When on, misses with no/unknown session id fall back to longest-prefix scan of the pool. Off → only explicit session-id reuse. |
| `session_id` (per request) | `None` | Primary reuse key. |

Budget is not a user flag — derived from hardware free memory.

## 8. Deferred (phase 1.5 / phase 2)

- **Phase 1.5 — offload idle caches to RAM.** v1 keeps caches on GPU and LRU-*drops*
  whole entries over budget (dropped session re-prefills). Phase 1.5: move idle
  entries to pinned RAM instead of dropping, copy back on resume (H2D ≪ re-prefill).
  Raises effective capacity without the A2 build.
- **Phase 2 — A2 shared-trunk across N branches.** Requires segmented/paged attention
  (leaves stock HF). A1's `longest_common_prefix` + `PrefixCacheStore` pool + `crop`
  are the substrate. A2 adds: block/segment KV so a trunk is referenced (not copied)
  by N branches; a 2-segment (or fully paged) attention path reading `[trunk]+[suffix]`;
  reference counting + block-LRU. A1's auto-prefix-match is the seed of A2's radix
  trunk-sharing. **Design note for A1:** keep entries range-addressable (token_ids +
  KV indexable by position range) so A2 can later share sub-ranges without reworking
  the store.

## 9. Testing strategy

**Unit (CPU, no CUDA; both geometries via synthetic config):**
- `longest_common_prefix`: empty, identical, partial, one-empty, no-overlap.
- `FP8DynamicCache.crop`: after crop, `get_seq_length()==length`; retained
  positions' data/scales byte-identical to pre-crop; a subsequent `update` appends
  correctly. Parametrize a Qwen-like and a MiniMax-like head/layer config.
- `PrefixCacheStore`: session-id hit returns correct `match_len`; auto-match picks the
  longest-prefix entry; miss returns `(None, 0)`; `store` inserts and LRU-evicts over
  budget; `drop` removes; thread-safety smoke test (concurrent lookups/stores don't
  corrupt `current_bytes`).
- `compute_prefix_cache_budget`: derives from free-mem input, both geometries.

**On-device (Qwen only — no MiniMax checkpoint on this box):**
- Turn 1 full prefill; Turn 2 with a context sharing a long prefix (small tail change)
  → measure TTFT turn1 vs turn2; turn2 should drop toward tail-only prefill cost.
- **Coherence guard:** generated output with `enable_prefix_cache=True` (reused KV)
  must be identical to `enable_prefix_cache=False` (fresh prefill) on the same
  prompt+seed. Divergence = a crop/position/FP8 bug — block ship.
- Feature-off regression: `enable_prefix_cache=False` output byte-identical to current
  HEAD.

## 10. Success criteria

1. Second-iteration TTFT on an overlapping-context request drops materially toward the
   cost of prefilling only the divergent tail (e.g. re-send with a small tail change →
   large TTFT reduction). Record the numbers.
2. Reused-KV output is identical to fresh-prefill output (coherence guard passes).
3. Feature-off path is byte-identical to today.
4. No OOM: pool respects the derived budget; over-budget → LRU-drop, graceful.
5. Model-agnostic: unit tests pass for both Qwen-like and MiniMax-like geometries;
   no model literals introduced.

## 11. Open risks to validate early in the plan

- **`FP8DynamicCache` internal layout** — crop must handle the exact FP8 data + scales
  + length-counter representation. Read `update` (`vaayu_base.py:277`) first; this is
  the one place a subtle bug corrupts all reuse.
- **HF generate + partial-prefill** — confirm the generate/prepare-inputs path accepts
  a pre-populated `past_key_values` with `input_ids` sliced to the tail (standard HF
  continuation, but verify Vaayu's custom forward at `vaayu_base.py:1804` threads it
  correctly).
- **Concurrency invariant** — one in-flight request per session id (§6). Document, and
  decide whether to add the defensive checked-out guard.
