# Chunked Prefill Design

**Date:** 2026-05-31  
**Plan:** schedule_plan_g_chunked_prefill.md  
**Status:** Design approved, pending implementation

---

## Problem

When a user pastes a long code file (4,000+ tokens), Vaayu's `forward()` passes all prompt tokens through all 80 layers in one pass. The temporary workspace for each transformer layer's attention computation scales with `O(seq_len²)`. On a 16 GB GPU with ~12 GB already occupied by pinned layers (Plan F), the remaining headroom cannot absorb the spike → OOM before the first token is generated.

---

## Goal

Process prompts in 512-token chunks during the prefill phase. Each chunk passes through all layers once, building the KV cache incrementally. VRAM workspace stays proportional to `chunk_size²` instead of `full_prompt²`.

---

## Scope

- Eliminate OOM crashes for prompts > 512 tokens on 16 GB VRAM with asymmetric pinning active
- Fix an existing latent bug: position IDs are currently computed only up to `max_seq_len` (default 512), which will silently corrupt hidden states for any prompt longer than `max_seq_len` even without chunking
- Output correctness: chunked prefill is mathematically identical to non-chunked prefill (attention is causal — each token only attends to prior tokens, so chunk boundaries introduce no approximation)
- ~10-15% slower prefill vs non-chunked (multiple full-layer passes instead of one)

---

## Design

### Parameter: `prefill_chunk_size`

Added to `VaayuBaseModel.__init__()` signature with default `512`.

```python
def __init__(self, ..., prefill_chunk_size=512, ...):
    self.prefill_chunk_size = prefill_chunk_size
```

`0` disables chunking. `AutoModel.from_pretrained()` requires no changes — `**kwargs` already flows through.

### Fix: Dynamic position ID range in `forward()`

Current code hardcodes position IDs to `max_seq_len` (default 512):

```python
position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]
```

For chunk 2+ (past_len=512, seq_len=512), `get_position_ids_args` slices `[512:1024]` from a 512-element tensor — out of bounds. Same bug affects any single-forward pass with a prompt longer than `max_seq_len`.

Fix: compute range dynamically from actual needs:

```python
past_len  = self.get_past_key_values_cache_seq_len(past_key_values) if past_key_values else 0
cur_len   = input_ids.shape[1] if input_ids is not None else 0
pos_range = max(self.max_seq_len, past_len + cur_len)
position_ids = torch.arange(pos_range, dtype=torch.long, device=self.running_device)[None, :]
```

RoPE recomputes from `pos_range` positions each call — still correct because it's derived from config, not the meta-initialized module.

### Early-exit in `forward()`

At the top of `forward()`, before the layer loop and before KV cache creation:

```python
if (self.prefill_chunk_size > 0
        and input_ids is not None
        and input_ids.shape[1] > self.prefill_chunk_size):
    past_kv_len = self.get_past_key_values_cache_seq_len(past_key_values) if past_key_values else 0
    if past_kv_len == 0:
        return self._chunked_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
```

Guard `past_kv_len == 0` ensures chunking only fires on pure prefill, not during decode (where each step already processes 1 token).

### New method: `_chunked_prefill()`

```python
def _chunked_prefill(self, input_ids, attention_mask, use_cache,
                     output_attentions, output_hidden_states, return_dict):
    chunk_size = self.prefill_chunk_size
    total_len  = input_ids.shape[1]
    n_chunks   = (total_len + chunk_size - 1) // chunk_size
    print(f"[Vaayu] Chunked prefill: {total_len} tokens → {n_chunks} chunks of {chunk_size}")

    past_key_values = None
    output = None

    for start in range(0, total_len, chunk_size):
        end        = min(start + chunk_size, total_len)
        chunk_ids  = input_ids[:, start:end]
        chunk_mask = attention_mask[:, :end] if attention_mask is not None else None

        output = self.forward(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        past_key_values = output.past_key_values

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[Vaayu] Chunk [{start}:{end}] done — peak VRAM: {peak:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

    if not return_dict:
        return tuple(v for v in [
            output.logits,
            output.past_key_values if use_cache else None,
            output.hidden_states,
            output.attentions,
        ] if v is not None)
    return output

# Note: output_hidden_states and output_attentions are accumulated only from
# the final chunk (not the full prompt). Both flags are unused in normal
# inference, so this is acceptable.
```

`return_dict=True` is forced for internal calls so we can reliably extract `past_key_values`. The final `return_dict` path respects the original caller's preference.

### test_quick.py

New config variable:

```python
PREFILL_CHUNK_SIZE = 512   # 0 = disabled; tokens per prefill chunk
```

Passed at model load:

```python
model = AutoModel.from_pretrained(
    MODEL_PATH,
    device=DEVICE,
    kv_cache_mode=KV_CACHE_MODE,
    max_pinned_layers=MAX_PINNED_LAYERS,
    prefill_chunk_size=PREFILL_CHUNK_SIZE,
)
```

---

## Data Flow

```
generate(input_ids[1, 4096])
  └─ forward(input_ids[1, 4096], past_kv=None)
       └─ _chunked_prefill(input_ids[1, 4096])
            ├─ forward(input_ids[1, 512],  past_kv=None)     → kv_512
            ├─ forward(input_ids[1, 512],  past_kv=kv_512)   → kv_1024
            ├─ forward(input_ids[1, 512],  past_kv=kv_1024)  → kv_1536
            │   ...
            └─ forward(input_ids[1, r],    past_kv=kv_3584)  → output (returned)
  └─ generate loop: decode autoregressively, 1 token at a time
```

---

## Files Changed

| File | Change |
|------|--------|
| `vaayu/vaayu_base.py` | Add `prefill_chunk_size` param; fix position ID range; add early-exit guard; add `_chunked_prefill()` |
| `vaayu/auto_model.py` | No change required (`**kwargs` passthrough already exists) |
| `test_quick.py` | Add `PREFILL_CHUNK_SIZE` config var; pass to `from_pretrained()` |

---

## Verification

1. **OOM test:** 8,192-token prompt, `prefill_chunk_size=0` → OOM; `prefill_chunk_size=512` → succeeds
2. **VRAM flatness:** Peak VRAM per chunk stays constant (printed by `_chunked_prefill`)
3. **Output correctness:** Same prompt, chunked vs non-chunked → identical generated text
4. **Decode unaffected:** Single-token decode steps (past_len > 0, seq_len=1) skip chunking path entirely
