# Handoff: context compaction is 19 % of a Rudra run's wall clock

**From:** Vaayu side (engine)
**To:** Rudra side (agent)
**Date:** 2026-11-24
**Status:** Vaayu's half is DONE and live. Rudra's half is unknown — that is what this document asks.
**Reply:** write it in section 9 of this same file and send it back.

This document is self-contained. Every number in it was measured on the Vaayu box; nothing
here requires reading Vaayu's source, its logs or its ledger.

---

## 1. What was measured, and what the headline number actually was

**Evidence:** one 78-minute Rudra stress run, `logs/run-2026-11-24T03-22-54Z`, 42 requests,
MiniMax-M2.7 EXL3 (144 GB on disk, 62 layers, 256 experts, top-8, 4.40 GB working set per
token) on an RTX 5060 Ti (16.62 GB VRAM) with 33.4 GB host RAM under a 28 GB cgroup cap.
Rudra ran off-box, so only Vaayu's own telemetry was available — **no prompt or completion
text is ever written to Vaayu's logs, by design.** Everything below about Rudra's behaviour
is therefore inferred from request *shape* (token counts, prefix match lengths, finish
reasons, timing), never from content. Where that matters, it is flagged.

**First, the reported figure was wrong, and in a reassuring direction.** The run was
described as "130–200 s/Token". Vaayu's decode on this model is **0.504 s/token, flat**.
The largest per-token figure in any run folder on the box is 18.7 s, and that was a
3-token reply where one request's fixed cost was divided by 3. The 130–200 s band matches
**per-request latency**: median 72 s, p75 104 s, mean 110 s, max 485 s, with three requests
at 368–385 s. If Rudra displays a "s/token" figure, it is almost certainly dividing a whole
request's wall time by something other than the tokens Vaayu generated.

**Second, the stress test passed.** 42/42 requests completed. Zero errors, zero cancels, no
crash, 78 minutes continuous, the server 97 % busy, strictly serial (N=1, zero queueing,
150 s total idle). SSE heartbeats fired correctly (max wire silence 10.0–10.6 s against a
10 s keepalive). Nothing about this run says Vaayu is unstable under load.

---

## 2. The two cost laws — use these to price your own ideas without asking us

Both fitted on the 42-request run, residual rms 2.1 s across 41 requests:

```
TTFT  =  12.8 s                     per-request expert disk read (unavoidable, per request)
       +  3.6 s                     first prefill chunk's 256-expert-per-layer visit
       +  6.70 ms x new_positions   positions NOT covered by the prefix cache

decode = 0.504 s/token              flat (expert disk 0.270 + compute 0.200)
```

So: **a cached position is free, a new position costs 6.70 ms, and a generated token costs
0.504 s.** A generated token is 75x a re-prefilled position. That single ratio decides every
trade-off below.

The decode figure is at the hardware floor — it is expert weights streaming off NVMe, 2.62 GB
moved per token against a 4.40 GB working set, with the expert cache at 25 % RAM / 29 % VRAM
hit rate. It cannot be improved without changing the model, which is off the table by design
(the point of the test is that MiniMax-M2.7 runs *at all* on this box).

---

## 3. Where the 78 minutes went

Non-overlapping, summing to the 4690 s span:

| block | seconds | share | can it be argued with? |
|---|---|---|---|
| decode, normal turns | 2220 | 47 % | **No.** Hardware floor, model pinned by design |
| **decode of 3 compaction summaries** | **885** | **19 %** | **Yes — entirely Rudra's** |
| prefill, new positions (6.70 ms each) | 763 | 16 % | Only by sending fewer new positions |
| prefill, fixed cost (16.4 s x 41 requests) | 672 | 14 % | Only by making fewer requests |
| idle (Rudra thinking / running tools) | 150 | 3 % | Rudra's |

**42 % of every prompt position Rudra sent was a re-prefill** — 113,911 of 269,083 positions
were text Vaayu had already seen and had to pay for again.

---

## 4. The compaction block, in detail

Three requests carry the signature:

| # | prompt tokens | prefix match | finish | output tokens | wall |
|---|---|---|---|---|---|
| 1 | 11,750 | **7** | stop | 555 | 372 s |
| 2 | 10,123 | **7** | stop | 595 | 385 s |
| 3 | 7,465 | **7** | stop | 595 | 368 s |
| | | | | **1,745** | **1,125 s** |

Each one is immediately preceded by a session whose chain had grown to **14,340 / 12,818 /
10,172** tokens against a cap of `max_seq_len - prompt_tokens`, and immediately followed by a
much shorter chain that *also* matches only 7 tokens. That is the summarize-then-restart
shape. (Inferred from shape, as stated in section 1 — Vaayu cannot see that these were
summarization prompts, only that they were near-total prefix misses producing ~600 tokens.)

**The 1,125 s decomposes into three very differently-priced parts:**

| part | seconds | why |
|---|---|---|
| decoding the summary itself | 885 | 1,745 output tokens x 0.504 s/token |
| re-prefilling the history | 197 | 29,338 prompt positions x 6.70 ms — because `prefix_match = 7` |
| fixed per-request cost | 43 | 3 x ~14 s |

That first line is the whole ballgame. **79 % of the compaction cost is Rudra deciding how
many tokens the summary may be.** The second line is pure waste of a different kind: the
history being summarized was *already in Vaayu's prefix cache* and the match was 7 tokens,
which means the summarize request arrived with a different prompt **head**, not a different
tail.

With the 7 other full-prefix-miss requests in the run, the full-miss block is **39 % of the
server's busy time.**

---

## 5. What Vaayu already changed (done, live, no action needed from Rudra)

Vaayu owned exactly one thing in this picture: **the ceiling.** A prompt that exceeds
`max_seq_len` becomes an HTTP 400, so Rudra has no way to opt out of compacting. The run
was pinned at `MAX_SEQ_LEN = 16384`.

Measured on device 2026-11-24, five boots, both attention kernels, GPU free 12.71–12.72 GiB
in every one:

| kernel | ceiling it can serve | reserve at its ceiling | pinning budget | expert VRAM tier |
|---|---|---|---|---|
| sdpa (was) | 26,111–26,620, and it **drifts** | 8.77 GiB @16384 | 4.15 GiB | 3.21 GB, 45/62 layers |
| flash_attention_2 (now) | **32,768**, stable | 8.56 GiB @32768 | 4.12 GiB | 3.17 GB, 44/62 layers |

The reason is that `sdpa` materializes an attention score buffer of 48 heads x 512-token chunk
x S keys x 10.9 B/element — 4.09 GiB at S=16384 — and `flash_attention_2` materializes none.
Flash spends that 4.09 GiB on KV cache instead.

**So flash at 32,768 costs almost exactly what sdpa cost at 16,384.** Same reserve, same
pinning budget, same expert cache. Double the ceiling for one price: flash decodes **2.2 %
slower** per token (504.26 vs 493.42 ms/token, measured through the server 2026-11-17 with
the context pinned equal in both arms).

**This is now live.** `logs/run-2026-11-24T10-45-20Z`:

```
✅ Attention implementation: flash_attention_2
[Vaayu-VRAM] attention cost: 0.00 B/score-element
[Vaayu-VRAM] max_seq_len=32768 (caller-supplied, validated against S_max=32768);
             reserve=8.56 GiB of which KV 7.75 GiB; pinning budget=4.12 GiB
GET /v1/models  ->  "max_seq_len": 32768
```

**Flash is not faster. It is 2.2 % slower and bought on purpose**, because it doubles the
ceiling and the ceiling is what forces compaction. The arithmetic on the measured run:

```
-1125 s  compaction removed (chains peaked at 14,340 — under 32,768)
+  68 s  flash decode tax (6,159 tokens x 2.2 % x 0.504 s)
---------
-1057 s of 4690 s  =  -22 % wall clock
```

**This entire -22 % is conditional on one thing on Rudra's side**, which is question 1 below.

---

## 6. What Rudra should change, ranked by payoff

Prices are from the measured run. Items 1–3 are worth more than everything else combined.

### 1. Follow the ceiling Vaayu advertises — worth the whole 1,057 s

Vaayu publishes the live window on **both** `GET /v1/models` and `GET /health` as
`max_seq_len` (now 32768). The run shows Rudra called `/v1/models` three times, so the
value was in Rudra's hands. **If Rudra's compaction trigger is a hardcoded 16 k, or a
`context_tokens` setting that nobody updated, the flash flip buys zero and costs 2.2 %.**

Please make the trigger a function of the advertised number, not a constant.

### 2. Cap the compaction summary's `max_tokens` — worth ~507 s

The three summaries were 555 / 595 / 595 tokens. At 0.504 s/token that is 885 s. A 250-token
cap would make it ~380 s. Nothing about Vaayu's side can touch this: it is output length,
and output length is the most expensive thing on this box.

### 3. Append the summarize instruction to the EXISTING chain, don't start a new one — worth ~197 s and it is free

`prefix_match = 7` on all three means the summarize prompt's **head** differed from the
chain's head. Vaayu's prefix cache matches a *prefix*: change the first token and the whole
history is re-prefilled. The 29,338 positions re-sent cost 197 s.

If instead the summarize instruction is appended to the end of the existing conversation
(same system prompt, same history, instruction as the final user message), the match would
be ~100 % and the prefill would be nearly free. Same tokens, same result, 197 s cheaper.

### 4. Send `max_tokens` on every request — protects decode, bounds the tail

Rudra currently sends none, so Vaayu grants the whole remaining window. Observed on the live
run 8 minutes after the flip:

```
[Vaayu-VRAM] admission: evicted 0.05 GiB of the expert VRAM tier
             to admit 32768 tokens (predicted peak 8.56 GiB)
             and capped its budget 3.17 -> 3.12 GB
```

Admission must reserve for the worst case it was told is possible. On the 16 k run this
ratcheted the expert VRAM cache from 3.21 GB down to 2.81 GB over 42 requests (cached layers
45 -> 39 of 62). At 32 k the worst case is twice as large. A `max_tokens` of 1–2 k costs
Rudra nothing it was going to use, and it also bounds a runaway: an uncapped request at this
ceiling is legally allowed to decode for about four hours.

### 5. Keep mutable blocks out of the system-prompt head — worth ~239 s on that run

A separate, smaller block of partial prefix misses: matches landed at 196 / 426 / 428 / 973 /
2,284 / 2,363 tokens. That pattern means a static preamble matched, then something that
changes every turn (a file tree, a memory block, a timestamp, a token count) invalidated
everything behind it. Mutable content belongs in the **last** message, not the first.

**This may already be fixed in your newer code** — the live run's matches are clean
(3,249 / 4,631 then 4,627 / 4,839 then 4,835 / 4,993, i.e. the full previous prompt each
time). Worth confirming rather than re-fixing.

### 6. Close the last non-streaming call path — 1 request, 88 s of silence

One request in the run had `stream: false` (352 prompt tokens, 140 output, 87.9 s). A
non-streaming request is wire-silent for its entire duration by construction; heartbeats
cannot help it, and it looks exactly like a hang to anything watching the socket. This is
the same shape as the "Vaayu keeps dying" reports from 2026-11-18, which were never crashes.

### 7. Trim the shared preamble, evict large tool results sooner — no direct saving

Chains grew ~1,806 tokens per turn in the run. Nothing here saves time directly; it delays
the trigger, which compounds with items 1 and 2.

### Do NOT do this

**Do not split long chains into fresh per-task sessions.** It was priced and it loses:
continuing a 14 k chain with 1.8 k new positions costs ~28 s of prefill; a fresh 7 k session
costs ~63 s. Shorter chains are *more* expensive here, because a cached position is free and
a new one is not.

---

## 7. What Vaayu will build next

Listed so Rudra can see what is coming and say whether any of it changes its own plan.

### V1. Verify the raised ceiling on a real Rudra run — in progress, no code

Watch the run now in flight for the single discriminator: **the prompt size at which the next
near-total prefix miss with a long output appears.**

- near **13.9 k** -> Rudra's trigger is still on 16 k; the ceiling raise bought nothing, and
  item 1 above is the only thing that matters.
- near **27.8 k**, or never -> the trigger followed; the -22 % is real and gets confirmed.

Acceptance: zero requests with `prefix_match <= 10` and `finish = stop` and `tokens > 400`;
decode within +3 % of 0.504 s/token; no prefix-pool-over-capacity warning.

Early signal with n=4 is good: decode 0.493 / 0.501 / 0.510 / 0.518 s/token (mean 0.505,
against sdpa's 0.504 — the 2.2 % tax is not visible yet), and TTFT on prefix hits is running
2–3 s *under* the cost law (15.8 s measured where the law predicts 17.8 s at 212 new
positions). Not yet a measurement, but not a regression either.

### V2. Make the elastic expert tier work on MiniMax — the engine-side answer to item 4

Vaayu already has a mechanism (`vaayu/elastic_tier.py`) that sizes the expert VRAM cache to
the request actually running instead of to the worst case, and gives the idle reserve back
to the expert cache. It is **refused on MiniMax today**:

```
[Vaayu-VRAM] elastic expert tier OFF: decode graphs are installed, and their memory pool
             is not something this tier may return to the driver. 5.27 GiB of the reserve
             stays unlent.
```

On the one model where it does run it was worth a lot (decode 56.7 -> 39.6 ms/token at 2 k
context, -24 % over an 8-turn session). **Unmeasured on MiniMax**, and it needs either the
decode CUDA graphs out of the way or an allocator that can return that pool — so this is an
investigation with a real chance of being refuted, not a promise. If it lands, an uncapped
request stops costing the expert cache, which makes item 4 a nicety instead of a requirement.

### V3. Fix `ttft_s: null` on non-streaming requests — small, certain

A `stream: false` request currently records no TTFT, so it silently drops out of every
prefill measurement we make. That is how item 6's request nearly went unnoticed. This matters
to Rudra only because it is the shared evidence base we both argue from.

### V4. Fix graceful shutdown — found while doing this work

An idle Vaayu server ignored both SIGINT and SIGTERM (pid unchanged after 180 s, no shutdown
log, main thread spinning in native code); SIGKILL was the only thing that worked. So
`systemctl stop` waits its full 120 s timeout and then hard-kills, meaning the graceful
teardown never runs. Hypothesis, not yet proved: a native call holding the GIL defers the
Python-level signal handler indefinitely. Relevant to Rudra only in that restarts between
test rounds are currently abrupt.

### V5. Conditional: prefix-cache budget at the new ceiling — only if it fires

At 32,768 one full-length prefix entry is 32,768 x 248 KiB = **7.75 GiB**, so the ~12.9 GB
prefix budget holds about 1.7 of them, where at 16,384 it held about 3.2. With three sessions
configured, the pool can now go over capacity and start evicting — something the 03-22 run
provably was *not* doing. If `[Vaayu-Prefix] WARNING: prefix pool over capacity` appears in
the next run, this gets built; if not, it stays unbuilt. **Watch for it.**

### Explicitly NOT building, and why — so these do not get requested

| idea | why not |
|---|---|
| Grow the expert VRAM cache | Settled 2026-10-16: it cannot clear the bar on MiniMax; H2D is only 31.5 % of a token and the ceiling is ~12 %. Flash's own 2.4x bigger tier moved bytes-per-token 0.08 % |
| Batch concurrent requests (N>1) | Measured a net loss on streamed MoE: a batch streams the per-chunk routing union off disk, ~N x expert bytes. `/v1/models` advertises `batching.helps: false` for this model |
| Smaller / different model | Off the table by design — the point of the test is MiniMax-M2.7 running at all |
| Tune `prefill_chunk_size` | Checked and discarded: 23 chunks cost 81.6 s where a per-chunk model predicts 144 s, because chunk retention already pays the per-chunk floor once. Not a lever |

---

## 8. Questions for Rudra

1. **Does Rudra read `max_seq_len` from `/v1/models` or `/health`, or is the window a
   constant?** If a constant, what is it, and where does it live? This is the single question
   that decides whether the -22 % happens.
2. **What triggers compaction** — a fraction of the window, a token count, a turn count? What
   is the fraction today?
3. **Can the summary's output be capped** at ~250 tokens? If not, what sets its length?
4. **Can the summarize request be appended to the existing chain** (same system prompt and
   history, instruction as the final user message) instead of starting a fresh prompt? This is
   item 3 and it is free.
5. **Can Rudra send `max_tokens`?** If yes, what value would it pick per request class?
6. **Are mutable blocks** (file tree, memory, timestamps, counters) still in the system-prompt
   head, or did that already move?
7. **Are there any remaining `stream: false` call paths?**
8. **What does Rudra need from Vaayu that does not exist yet?** Concretely: would a token-count
   endpoint help, so Rudra can decide when to compact against Vaayu's own tokenizer instead of
   estimating? Anything else on `/v1/models` that would make the decision easier? Vaayu is
   the side that can add it.

---

## 9. Rudra's reply — write here and send back

> **Date:**
> **Answered by:**

**Q1 — ceiling source:**

**Q2 — compaction trigger:**

**Q3 — summary output cap:**

**Q4 — append-to-chain instead of new prompt:**

**Q5 — `max_tokens`:**

**Q6 — mutable blocks:**

**Q7 — non-streaming paths:**

**Q8 — what Rudra needs from Vaayu:**

**Anything Rudra found that this document gets wrong:**

**Rudra's own plan, in the order it will do it:**
