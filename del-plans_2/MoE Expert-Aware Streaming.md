Implementing MoE Expert-Aware Streaming.


If you want to use moonshotai/Kimi-K2.6 with the Vaayu engine, you face a unique set of challenges due to its scale. Kimi-K2.6 is a 1-trillion parameter MoE (Mixture-of-Experts) model with 32B active parameters per forward pass.


The MoE Streaming Problem: In vaayu_base.py#L589
, the MoE detection logic logs: "Expert-aware streaming (Phase 3): pending — currently streams full MoE layer".

Because expert-aware streaming is not yet implemented, Vaayu will stream the entire 594 GB model from disk to GPU for every single token generated. Even with a hyper-optimized io_uring + O_DIRECT pipeline fetching at $6.5$ GB/s, reading 594 GB will take $\sim 91$ seconds per token.

No Compatible Drafter: Kimi-K2.6 uses a custom tokenizer with a 160,000 (160K) token vocabulary. Moonshot AI has not released a small (e.g., 1.5B or 0.5B) model that shares this vocabulary. Therefore, you cannot use standard speculative decoding (e.g. drafter_model_path) because there is no matching small model.
What to do:
You must run Kimi-K2.6 without speculation (drafter_model_path = None and eagle_head_path = None).
Or, you must implement Prompt Lookup Decoding (PLD), which does not require any small model weights and works on any vocabulary by matching n-grams directly in your prompt text.


BUT WHAT IF Expert-Aware Streaming is implemented correctly?

Viewed vaayu_base.py:601-627

Implementing **Expert-Aware Streaming** (Phase 3) is the holy grail of running huge MoE models like `moonshotai/Kimi-K2.6` or `mistralai/Mixtral-8x22B` on consumer hardware. 

If you implement it, here is exactly what will happen to your performance, your architecture, and your pipeline:

---

## 📈 The Good: Massive Throughput Surge
You will achieve a **$\sim 95\%$ reduction in SSD-to-GPU data transfer**. 
* **The Math**: Instead of streaming the entire 594 GB model (all experts) per token, you only stream the routing gates and the active experts. For a 32B active parameter MoE, this is only **$\sim 16$ GB** of active weights per token. 
* **The Latency Impact**: At a realistic $6.5$ GB/s NVMe read speed:
  * *Without Expert-Streaming*: $594\text{ GB} \div 6.5\text{ GB/s} = \mathbf{91.3\text{ seconds/token}}$.
  * *With Expert-Streaming*: $16\text{ GB} \div 6.5\text{ GB/s} = \mathbf{2.46\text{ seconds/token}}$.
  * You drop from a completely unusable freeze to a somewhat interactive tier.

---

## 🛑 The Bad: Prefetch Pipeline Breakage
Currently, Vaayu uses a beautiful **producer-consumer overlapping model** ([vaayu_base.py#L1133-L1156](file:///c:/laragon/www/Vaayu/vaayu/vaayu_base.py#L1133-L1156)): while Layer $N$ runs on the GPU, a background thread pre-fetches Layer $N+1$ from the SSD.

**Expert-Aware Streaming completely breaks this overlap:**
1. You cannot know which experts are needed for Layer $N+1$ until you run the routing gate for Layer $N+1$.
2. The input to the routing gate of Layer $N+1$ is the output hidden state of Layer $N$.
3. Therefore, **the SSD cannot start reading Layer $N+1$'s weights until the GPU has finished executing Layer $N$.**

You lose the ability to overlap disk I/O and GPU compute across layers. Every layer boundary becomes a hard synchronization point: the GPU must wait for the SSD to read the dynamically selected experts.

---

## 🛠️ The Technical Implementation Challenges

To make this work, you must redesign three parts of the engine:

### 1. Router Layer Pinning
The "routing gates" (the small linear layers that decide which experts get the token) are tiny (typically $< 1$ MB per layer). 
* You must modify `_init_pinned_layers()` to permanently pin all router layers of the model in VRAM. This allows the GPU to instantly decide which experts are active without waiting for a disk read of the router weights.

### 2. Sparse Byte-Offset File Reading
Instead of loading the entire `layer.safetensors` file, you must perform sparse, non-contiguous block reads:
* At startup, you must parse the safetensors header to map the exact starting and ending byte offsets of each individual expert (e.g., `expert_0` is at bytes `100M–250M`, `expert_1` is at `250M–400M`).
* During execution, after the router selects experts (e.g., indices `3` and `7`), you must issue separate, parallel `io_uring` SQEs to fetch only the byte ranges for those specific experts.

### 3. Dynamic GPU Tensor Reconstruction
In [loader_strategy.py](file:///c:/laragon/www/Vaayu/vaayu/persist/loader_strategy.py), your `inject_to_gpu` method must dynamically map the loaded raw bytes of the sparse experts into the GPU. You will need to allocate placeholder MoE modules on the GPU and only populate the weights of the active expert slots, leaving inactive slots empty/dummy, or call sparse MoE kernels that accept a list of active expert tensors directly.



This is about  "Prefetch Pipeline Breakage", While looking further found that,

The "Prefetch Pipeline Breakage" was overstated.

I said the prefetch pipeline is "completely broken." That's not entirely accurate. Here's why:

Each MoE transformer layer has two distinct sub-components:

Shared attention block (self-attention + layer norms) — these are the same regardless of which experts fire. They are deterministic and can be prefetched.
Expert FFN block (the MoE feed-forward network) — only the active experts are needed, and you can't know which ones until the router runs.
So the pipeline doesn't break completely. It splits into a two-phase pipeline per

Layer N execution on GPU:
  ├─ Phase A: Run shared attention (prefetched) + run router gate (pinned)
  │           → now we know which experts Layer N needs
  │           → issue io_uring reads for Layer N's active experts
  │           → SIMULTANEOUSLY prefetch Layer N+1's shared attention weights
  ├─ Phase B: Run active expert FFN (just loaded)
  │           → SIMULTANEOUSLY, Layer N+1's shared attention is being prefetched
  └─ Done → move to Layer N+1

So the overlap isn't lost — it's reduced. You still overlap the shared attention prefetch of Layer $N+1$ with the expert FFN compute of Layer $N$. The stall only happens when you need to wait for the expert-specific weights after the router decides.

The actual stall per layer is small. For Kimi-K2.6 with 256 experts and top-2 routing, each expert is roughly $\frac{1\text{T}}{256} \approx 4\text{B}$ params. At 4-bit, 2 active experts ≈ $4$ GB per layer. At $6.5$ GB/s, that's only ~0.6 seconds of expert-loading stall per layer — not the full 91 seconds I implied.

Prefetch pipeline is partially broken (expert phase only, shared attention still overlaps).

---

**Now how to fix Prefetch pipeline?**


Great question. There are three practical approaches to restore the prefetch overlap, ranging from simple to aggressive. Each one attacks the stall from a different angle.

---

## The Core Problem (Recap)
When the forward loop reaches an MoE layer, it must:
1. Run shared attention (prefetchable ✅)
2. Run the router gate to select active experts (instant if pinned ✅)
3. **Read active expert weights from SSD** ← STALL happens here
4. Run expert FFN on GPU

Step 3 is the stall: the SSD can't start reading until the router has decided which experts are needed, and the router can't run until the attention output is ready.

---

## Fix 1: Expert RAM Cache (Simplest, Highest Impact)

**Observation:** MoE routing exhibits strong **temporal locality** during autoregressive generation. When generating code, the model is continuing the same thought — token after token tends to activate the *same experts* in each layer. Research on Mixtral and DeepSeek-MoE shows that across consecutive tokens, expert reuse rates are often **70–90%**.

**The fix:** Keep recently-used expert weights in a CPU RAM LRU cache, keyed by `(layer_index, expert_index)`.

```
Token 1: Layer 42 routes to experts [3, 7]
         → Read experts 3 and 7 from SSD → cache them in RAM

Token 2: Layer 42 routes to experts [3, 11]
         → Expert 3: RAM cache HIT (instant, no SSD read)
         → Expert 11: RAM cache MISS → read from SSD (stall)
         → Cache expert 11, evict oldest expert if cache full
```

**Why it works for Vaayu:**
- With 64 GB of system RAM, you could cache **hundreds** of individual experts (each is ~2 GB at 4-bit for a 1T MoE). That's enough to cover the hot working set.
- This stacks perfectly with `O_DIRECT`: cache misses still go through the zero-copy NVMe path, cache hits bypass the SSD entirely.
- The cache is at expert granularity, not layer granularity — far more memory-efficient than caching entire layers.

**Stall elimination:** ~70–90% of expert reads become instant RAM hits. Only cold/novel expert activations stall on SSD.

---

## Fix 2: Speculative Expert Prefetching (Medium Complexity, Near-Zero Stall)

**Observation:** You can *predict* which experts Layer $N+1$ will need while Layer $N$ is still computing, by looking at what Layer $N+1$ used on the **previous token**.

**The fix:** Maintain a per-layer routing history table:

```python
# Updated after each token's forward pass
self._expert_history[layer_idx] = set(active_expert_indices)
```

While Layer $N$ is executing on the GPU, the prefetch thread does:
1. Look up `self._expert_history[N+1]` (which experts Layer $N+1$ used last token)
2. Speculatively prefetch those experts from SSD into pinned RAM
3. When Layer $N+1$'s router actually runs:
   - **Prediction correct** (common case): expert data is already in RAM → zero stall
   - **Prediction wrong** (rare): discard prefetched data, issue on-demand SSD read → same stall as today

```
Time →
GPU:     [Layer N: attn + expert FFN]    [Layer N+1: attn → router → expert FFN]
SSD:     [SPECULATIVE: prefetch N+1 experts based on history]
                                          ↑ if prediction was right, data is ready
```

**Accuracy:** In autoregressive generation, the routing prediction accuracy from the previous token is typically **80–95%** because the hidden states evolve smoothly token-to-token. You're almost always guessing right.

**Downside:** On the first token (prefill), there's no history. You either fall back to full-layer loading or use a cold-start heuristic (e.g., prefetch the statistically most popular experts).

---

## Fix 3: Intra-Layer Split Pipeline (Most Aggressive, Architectural Change)

**The idea:** Stop treating each transformer layer as a monolithic unit. Split it into two sub-layers in Vaayu's layer list:

```
Current layer list:
  [embed, layer.0, layer.1, ..., layer.79, norm, lm_head]

Split layer list:
  [embed, layer.0.attn, layer.0.experts, layer.1.attn, layer.1.experts, ..., norm, lm_head]
```

Now the existing prefetch pipeline works naturally:
- While `layer.0.experts` runs on GPU → prefetch `layer.1.attn` from SSD (shared, deterministic)
- While `layer.1.attn` runs on GPU → router runs instantly (pinned) → now you know which experts to read
- SSD starts reading `layer.1.experts` → while computing... → etc.

The overlap is restored because the attention phase gives the SSD time to fetch the next layer's shared weights, and the router decision comes early enough that expert loading can begin before the attention phase finishes.

**Downside:** This requires restructuring `forward()` to treat each MoE layer as two separate pipeline stages rather than one, and splitting the per-layer safetensors files into `layer.N.attn.safetensors` and `layer.N.experts.{0..255}.safetensors`.

---

## Recommended Strategy

| Fix | Complexity | Stall Reduction | Best For |
|-----|-----------|-----------------|----------|
| **Expert RAM Cache** | Low | ~70–90% | First implementation. Works immediately with existing pipeline. |
| **Speculative Prefetch** | Medium | ~80–95% | Second optimization. Stacks on top of RAM cache for near-zero stall. |
| **Intra-Layer Split** | High | ~100% | Ultimate architecture. Only worth it after the first two are proven. |

**My recommendation:** Implement **Fix 1 (Expert RAM Cache)** first — it's the smallest code change, delivers the biggest single improvement, and naturally evolves your existing [Application-Level RAM Layer Cache plan](file:///c:/laragon/www/Vaayu/plans/Application-Level%20RAM%20Layer%20Cache-implementation_plan.md) from layer-granularity to expert-granularity. Then layer **Fix 2 (Speculative Prefetch)** on top to cover the remaining cache misses.
