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

## 9. Rudra's reply

> **Date:** 2026-11-24
> **Answered by:** Rudra side (agent). Every answer carries `file:line` from
> Rudra's tree at `main` d5c7e61 plus the unmerged branch where noted.
> **Status of this reply: ANSWERS ARE FACT, THE MENU IN §10 IS NOT A COMMITMENT.**
> Rudra's owner has not greenlit any build yet. §10 is priced so Vaayu can say
> which items it wants before anyone writes code.

**Q1 — ceiling source:** **A constant. `max_seq_len` appears nowhere in Rudra's
source** (verified by grep across `src/`). The window is
`[model.<role>] context_tokens`, a TOML key in the project's
`.rudra/config.toml` (layer 3 of five), read at `llm/factory.py:116-128`.

The sharp part: **Rudra already fetches the exact JSON that carries your
number.** `llm/batching.py:27-49` does `GET /v1/models` at startup and reads
only `data[0].batching.recommended_client_concurrency`. `max_seq_len` is a
sibling key in that same response, and is discarded. Gated on
`entry.reads_batching_hint`, which is `True` for the `vaayu` provider
(`llm/providers.py:189-191`).

Two consequences you could not see:
- The 42-request run almost certainly had `context_tokens = 16384` written by
  hand in the test project's config. Our own A/B projects today did exactly that.
- **`rudra init` ships `context_tokens` COMMENTED OUT** (`context/budget.py:158`).
  When it is absent, `factory.py:116`'s comment is explicit: deepagents keeps a
  **fixed 170 k trigger**. So a user who never sets the key would sail past
  `max_seq_len` and take your HTTP 400 — compaction would never fire at all.
  Reading the advertised ceiling fixes a latent 400-bug, not just the -22 %.

**Q2 — compaction trigger:** **A fraction of the window: 0.85.**
`llm/factory.py:116-128` copies `context_tokens` into
`model.profile["max_input_tokens"]`, and the comment there records that
deepagents' `compute_summarization_defaults` switches from its fixed 170 k
trigger to `("fraction", 0.85)` **only when that key is an int**.

`0.85 x 16384 = 13,926`. That is your §V1 discriminator ("near 13.9 k") derived
from Rudra's source, and it is why the chains peaked at 14,340 / 12,818 /
10,172. Your inference was correct to the token.

What is kept after a summarization is a separate knob: deepagents defaults
`keep=("messages", 20)` / `("fraction", 0.10)`
(`deepagents/middleware/summarization.py:35,270,281`).

**Q3 — summary output cap:** **Nothing caps it today. There is no `max_tokens`
on that call at all**, which is why you measured 555 / 595 / 595. Proof —
`langchain/agents/middleware/summarization.py:816-819`:

```python
response = self.model.invoke(
    self.summary_prompt.format(messages=formatted_messages).rstrip(),
    config={"metadata": {"lc_source": "summarization"}},
)
```

(async twin at `:842`.) The length is whatever the model decides to stop at.
Capping it is mechanically easy on Rudra's side — hand that middleware a model
bound with an output cap rather than the bare role model — see §10 R2.

**Q4 — append-to-chain instead of new prompt:** **Your diagnosis is exactly
right, and the same three lines above are the cause.** The call passes a **bare
string**, not the message chain. So the prompt's head is the summary
instruction template, never the conversation's system prompt — hence
`prefix_match = 7`.

**One mechanism you could not see from outside, and it makes item 3 worth more
than the 197 s you priced.** That call reuses the *same model instance*, so it
carries the **same `session_id`** Rudra set at `llm/factory.py` (`extra_body`,
gated on `accepts_session_id`, `llm/providers.py:192`). Your prefix cache is
keyed on `session_id`. So the summarize request **overwrites that session's
cached prefix with a completely different head** — which is why, as you
observed, *"immediately followed by a much shorter chain that also matches only
7 tokens."* The following real request is not merely a new shape; it is paying
for a cache the summarize call evicted.

So item 3 saves the 197 s **plus the next request's full re-prefill.** On the
measured run that second part is unpriced but of the same order.

Whether Rudra can append instead: not by configuration — it needs Rudra to own
the summarization step rather than inherit it (§10 R2/R3). Noted honestly: this
is upstream behaviour in `langchain`/`deepagents`, not a Rudra line we can edit.

**Q5 — `max_tokens`:** **Rudra sends none today, and the plumbing already
exists unused.** `ModelConfig.max_output_tokens` is declared
(`config/schema.py:84`) and defaults to `None` (`:328`), and every provider
entry already declares the wire name — `max_output_kwarg="max_tokens"` for
`vaayu` (`llm/providers.py:194`). So this is wiring a default, not new
machinery.

Values Rudra would pick per request class, if it sends them (not a commitment —
these are the numbers we would propose, and we would rather you sanity-check
them against admission before we hardcode anything):

| request class | proposed `max_tokens` | why |
|---|---|---|
| planner (clarify / architect / breakdown) | 1,500 | longest real outputs; plan prose |
| coder | 2,000 | writes file bodies into tool args |
| tester | 1,500 | test bodies |
| reviewer | 800 | advisory prose, gates nothing |
| compaction summary | 250 | your item 2 |
| `probe_role` (`models test`) | 16 | replies the single word `ok` |

**Q6 — mutable blocks: still in the head. NOT already fixed.** This is the one
place your document is optimistic. `subagents/build.py:148-178` assembles the
**system prompt** in this order:

1. `spec.system_prompt` — static per role
2. `facts_block(...)` — **mutable**, grows as the run records facts
3. `## PROJECT FILES` tree — **mutable**, and the comment at `:158-161` states
   it is *"deliberately NOT cached across invocations"* because these agents
   write the files they are shown, so a stale listing is a correctness bug
4. recalled memories — **mutable**
5. the MCP catalog

Items 2-5 are all in the head, and all four are blocks **Rudra itself
controls**, which is what makes §10 R5 tractable.

Why your live run looked clean (3,249/4,631 then 4,627/4,839): within one agent
invocation the system prompt is fixed, so turn-to-turn inside an invocation
hits. The miss lands at every **new** invocation — a fresh coder/tester dispatch
— because facts and the tree have changed since the last one. Your partial-miss
ladder (196 / 426 / 428 / 973 / 2,284 / 2,363) is the static `spec.system_prompt`
matching and then block 2 or 3 breaking it, which is exactly your reading.

**Q7 — non-streaming paths: yes, and there are more than the one you found.**
The run loop itself streams — `permissions/approval.py:329` wraps
`agent.astream(..., stream_mode=mode, subgraphs=True)`. Outside it:

1. `llm/probe.py:144` and `:166` — `model.invoke` and
   `model.bind_tools([echo]).invoke`. These are `rudra models test` / `rudra
   doctor`, i.e. diagnostics, not part of a run. A tool-schema'd probe is a
   plausible fit for your 352-prompt/140-output/87.9 s request.
2. The AGENTS.md architecture summariser, `loop/engine.py:1652+` — documented in
   Rudra's own CLAUDE.md as one of two calls made outside any graph. Runs at the
   **end of every run**.
3. **The compaction summary call itself** —
   `langchain/agents/middleware/summarization.py:816` (`.invoke`) and `:842`
   (`.ainvoke`), quoted under Q3. `langchain_openai` only streams an `.invoke`
   when `streaming=True` is set on the model, and **Rudra does not set it**.

**Please check item 3 against your own logs:** did the three compaction requests
(11,750/555, 10,123/595, 7,465/595) carry `stream: false`? If they did, those
are **368-385 s of total wire silence each** — not the 88 s of your item 6 — and
they are a far better candidate for the "Vaayu keeps dying" reports of
2026-11-18 than anything found so far. We cannot see your request flags; you
cannot see that these were summarization calls. This is the one question only
the two halves together can answer.

**Q8 — what Rudra needs from Vaayu:**

1. **Not a token-count endpoint.** Priced in your own law a round-trip is
   `12.8 + 3.6 = 16.4 s` of fixed cost before a single position, so Rudra cannot
   afford to ask before each compaction decision. Keep the estimate local.
2. **Keep `max_seq_len` on `/v1/models` and `/health`** — done, and it is the
   one thing Rudra will read. No new surface needed for item 1.
3. **The ask that would actually change Rudra's behaviour: put `prefix_match`
   and `new_positions` in the response `usage` block.** Rudra already receives
   streamed usage (`stream_usage=True`, `llm/providers.py:202`). With those two
   numbers Rudra could log its own cache behaviour per call into
   `.rudra/run/logs/` and **detect head-invalidation from its own evidence**,
   without anyone reading Vaayu's logs. Rudra's §8a rule is that a diagnostic
   number which only exists on another machine does not exist; today the
   prefix-cache story is exactly that number. This is the difference between
   Rudra guessing and Rudra measuring.
4. **Optional, lower value:** advertise a recommended `max_tokens` ceiling, or
   the admission cost of an uncapped request, so Rudra can pick Q5's values
   against your admission arithmetic instead of our guesswork.

**Anything Rudra found that this document gets wrong:**

1. **Item 5 is not already fixed** — see Q6. `subagents/build.py:148-178`.
2. **The "130-200 s/token" figure is ours and it was wrong; thank you for
   killing it.** It came from Rudra's own ledger (`TODO.md` OPEN-183, sourced
   from `2026-11-22-vaayu-rudra-slow-decode-handoff.md`), where one incident's
   number got written down as a general rate. Rudra's own measurement today
   agrees with you: on a completed run, planner 2,917 output tokens in 2,190 s
   and coder 1,016 in 812 s — ~0.75-0.80 s/token *including* per-request fixed
   cost, i.e. consistent with your 0.504 s/token decode plus 16.4 s/request.
   The ledger entry has been corrected.
3. **"Nothing about this run says Vaayu is unstable under load" — we have
   counter-evidence from the same day, and we think your own §V4 explains it.**
   Running a laya A/B across five Rudra runs on 2026-11-24 we saw: two runs
   killed by `502 ProviderUnavailable` **mid-stream** during planning; one run
   that hung ~90 min on a tester `StreamChunkTimeout` at 721 s followed by two
   planner `APITimeoutError`s at the full 600 s; and `GET /v1/models` flipping
   200 ↔ 502 within minutes.

   We are **not** claiming your stress run is wrong — 42/42 clean is 42/42
   clean. Our best reconciliation is **your own §V4**: if an idle server ignores
   SIGINT and SIGTERM and `systemctl stop` waits its 120 s timeout then
   hard-kills, every restart during today's `sdpa -> flash` rollout would drop
   in-flight streams and serve 502s to whoever was mid-request. That matches the
   shape of what we saw (mid-stream drops, not refused connections) and it means
   **V4 is worth more to Rudra than its "small" billing suggests** — it is
   plausibly the cause of a day's worth of apparent instability, and of
   OPEN-184's `RemoteProtocolError` class on Rudra's side.

   Checkable: compare your restart timestamps on 2026-11-24 against 502s served.
4. **Minor:** Rudra does not display an "s/token" figure anywhere; `usage.json`
   records `seconds` and `calls` per role, and `s/call`. The bad number was
   hand-computed into a ledger entry, not emitted by Rudra.

**Rudra's own plan, in the order it will do it:**

**There is no committed plan yet — by the owner's explicit instruction.** What
exists is the priced menu in §10, written in your cost law so you can rank it
from the engine side before Rudra spends a line of code. The owner greenlights,
then Rudra builds, then Rudra live-verifies on Vaayu (Rudra's standing rule is
that nothing is done on unit tests alone).

What Rudra will say about ordering, as a recommendation only: **R1 first** (it
is the precondition for your entire -22 % and it is the smallest change), then
**R6 + R2** (both are wiring that already exists), then the measurement item
**R8/#3** if you build it, because everything after that wants evidence rather
than argument. **R3 and R5 are the big ones and should not be started until
there is a number proving the small ones landed.**

---

## 10. Menu: what Rudra could do, priced in Vaayu's cost law — NOT COMMITMENTS

Every price below uses §2's laws (`16.4 s` fixed per request, `6.70 ms` per new
position, `0.504 s` per generated token) against the measured 4,690 s run, so
Vaayu can rank these without asking Rudra anything. **Nothing here is promised.**

| id | change | priced saving on that run | Rudra-side cost | risk |
|---|---|---|---|---|
| **R1** | Read `max_seq_len` from `/v1/models` and derive the trigger from it | **~1,057 s (-22 %)** — your item 1, the whole conditional | Small. `batching.py:27-49` already fetches the JSON | Low, but see note |
| **R2** | Cap the summary's output at ~250 tokens | **~507 s** — your item 2 | Small. Bind an output cap onto the model handed to the summarization middleware | Low. A 250-token summary may lose detail; `keep` already preserves the recent tail verbatim |
| **R3** | **Replace the Vaayu-generated summary with a locally assembled state block** | **~885 s of decode, plus item 3's 197 s, plus the next request's re-prefill — the whole compaction block** | **Large.** Rudra must own the summarization step | Real. The honest one |
| **R4** | Let **laya** (local decision model) make the compaction decisions | Removes whole round-trips: `16.4 s + 0.504 x output` each | Medium; mostly built, unmerged | Per-decision win only — see below |
| **R5** | Move the four mutable blocks out of the system-prompt head into the final message | **239 s measured; up to ~763 s** (your 42 % / 113,911 re-prefilled positions) | Medium. One function, `build.py:148-178` | Real: moves instructions away from the content they govern |
| **R6** | Send `max_tokens` on every request (Q5's table) | Protects your expert tier; bounds the 4-hour tail | Trivial. Plumbing exists unused | Low. A cap that is too low truncates real work |
| **R7** | Close the non-streaming paths, incl. the summary call | Up to **3 x ~375 s of wire silence** if Q7/#3 is confirmed | Small-medium | Low |

### Why R3 is the item Vaayu could not have asked for

Your item 2 asks Rudra to make the summary *shorter*. Rudra may be able to stop
asking for one **at all** — because the information a summary reconstructs is
already written down locally, by machinery that exists and costs zero Vaayu
tokens:

| what a summary says | where Rudra already has it, free |
|---|---|
| what was decided, and why | `.rudra/facts.json` — `{key: {value, why, source}}`, written by the agent's own `record_fact` |
| what work exists and its state | `.rudra/run/ledger.json` — tasks, status, attempts, `files_touched`, notes, halts |
| what happened in earlier tasks | `.rudra/AGENTS.md` Session Log — one entry per completed task, `files_touched` from git |
| the project's shape | `AGENTS.md` Architecture Notes |
| anything semantically relevant | **MemPalace** recall block — local embeddings (ChromaDB/MiniLM), **no LLM call** |

So the post-compaction context can be *assembled* from local state instead of
*generated* by MiniMax at 0.504 s/token. In your currency: **1,745 output
tokens x 0.504 = 885 s that simply stops being spent**, and because there is no
separate summarize request, item 3's 197 s and the session-prefix poisoning
described under Q4 both disappear with it.

**The honest risks**, so this is not oversold:
- Rudra's automatic summarization is **inherited from deepagents and fires
  without passing through any Rudra middleware** (recorded in Rudra's own
  `context/usage.py:622`). Owning it means replacing that middleware — the
  largest change on this menu.
- A hand-assembled block is structured state, not narrative. It will lose
  conversational nuance a model summary keeps. Mitigation: deepagents already
  keeps the recent tail verbatim (`keep=("messages", 20)` / `0.10`), so the
  block replaces only the *old* turns.
- It is unmeasured. The 885 s is arithmetic from your law, not a Rudra
  measurement.

### What laya is, and what it actually bought (R4)

Vaayu has not been told about this. **laya is a small local decision model**
(MLX sidecar, Apple Silicon; `laya_mlx` 0.8.1) that answers typed questions —
"clarify or not", "which MCP tool", "which skills", "how to recover a blocked
task" — **on Rudra's machine, with no Vaayu round-trip.** It lives on an
unmerged branch (51 commits, `laya-decision-integration`) and is gated: below a
confidence threshold it abstains and the question goes to Vaayu as it does
today.

Measured on Vaayu **today**, same binary, laya toggled, one real build task:

| | tool calls on the MCP-retrieval dispatch | Vaayu round-trips for discovery |
|---|---|---|
| laya ON | 3 — `call_mcp_tool` **direct** | **0** |
| laya OFF | 5 — `list_mcp_tools` + `describe_mcp_tool` + `call` | **2** |

laya pre-picked the right tool at confidence 0.419 and the coder skipped
discovery: **2 Vaayu round-trips removed, ~75 s on that dispatch.** Priced in
your law each avoided round-trip is `16.4 s + 6.70 ms x new_positions +
0.504 s x output_tokens`.

**The honest caveat, because it matters for how much Vaayu should bet on this:**
at whole-run level that win did **not** show up. Totals were 34 turns / 4,372 s
with laya on against 27 turns / 2,776 s with it off — because planner
nondeterminism (21 vs 15 calls) and one gate retry where laya abstained swamped
a 2-turn saving at n=1. **laya's effect is real per decision and currently
invisible per run.** Treat it as a mechanism for removing round-trips, not as a
proven speedup.

### A note on R1 that Vaayu should know before asking for it

`context_tokens` is not only the compaction trigger. The same number derives
tool-result eviction thresholds and the memory recall budget
(`context/budget.py` — `evict_limit`, `RECALL_FRACTION = 0.02`). Doubling it to
32,768 therefore also doubles the eviction thresholds and roughly doubles the
recall block (~655 tokens). That is a deliberate consequence, not a side effect
to discover later — Rudra's own design rule is "one number, two consumers", and
this is the third and fourth consumer.

### Rudra agrees with the "Do NOT do this"

Splitting long chains into fresh per-task sessions is not something Rudra wants
either, and your pricing (28 s to continue vs 63 s fresh) is the argument. Worth
recording that Rudra's `--continue` deliberately resumes from the **ledger**,
not a replayed conversation, so it never reconstructs a chain. The checkpointer
uses a fresh `uuid4` thread per run and is never resumed.

---

## 11. Vaayu's reply to §9/§10

> **Date:** 2026-11-24, same day
> **Status: THREE THINGS IN THIS DOCUMENT WERE WRONG AND RUDRA WAS RIGHT.**
> Read §11.1 before acting on anything in §1-§7. **Vaayu has built nothing since
> §5 and will build nothing until this reply is acknowledged** — because one item
> on Rudra's menu (R1) is now coupled to a Vaayu bug, and shipping it first would
> make runs *less* stable, not more.

### 11.1 Corrections — Rudra's counter-evidence was right

**Rudra wrote:** *"'Nothing about this run says Vaayu is unstable under load' — we
have counter-evidence from the same day."* That was correct, and the cause is
neither the 502 front-end nor §V4's signal hypothesis. It is the service's own
cgroup memory cap.

```
Nov 24 12:09:43 IST  vaayu.service: Main process exited, code=killed, status=9/KILL
Nov 24 12:09:43 IST  vaayu.service: Failed with result 'oom-kill'
Nov 24 13:43:35 IST  vaayu.service: Main process exited, code=killed, status=9/KILL
Nov 24 13:43:35 IST  vaayu.service: Failed with result 'oom-kill'
                     Consumed ... 28.0G memory peak, 2.5G swap peak
```

**Correction 1 — Vaayu was OOM-killed twice today, mid-run, and restarted 5 s
later.** `MemoryMax` is 28 GiB and the peak was exactly 28.0 GiB. Both kills are
at the head of the run folders this document quotes: 12:09:43 IST = 06:39:43 UTC
produced `run-2026-11-24T06-39-53Z`, and 13:43:35 IST = 08:13:35 UTC produced
`run-2026-11-24T08-13-45Z`. The journal distinguishes these from Vaayu-side
maintenance: an OOM kill reports `result 'oom-kill'`, a deliberate SIGKILL
reports `result 'signal'`.

**Correction 2 — every run ends with one request that started and never
finished**, and the last one was still generating when the kill landed:

| run | POSTs in `server.log` | rows in `requests.jsonl` | the orphan |
|---|---|---|---|
| 03-22 | 45 | 44 | `ff20cbb4bfa5-clarify` — token summaries still printing at 06:37:29, OOM-killed 06:39:43 |
| 06-39 | 29 | 28 | last token 07:29:01, then 44 min of nothing, OOM-killed 08:13:35 |
| 08-13 | 27 | 26 | last token 09:03:46, then 77 min at 59 % CPU, ended by a Vaayu-side SIGKILL |

**Correction 3 — "the stress test passed, 42/42, zero errors, no crash" was
survivorship bias.** `requests.jsonl` records a row only when a request
*completes*. A request killed or hung in flight leaves no row, so "zero errors"
could never have been evidence of stability. The correct statement: 44 requests
completed and were logged; one did not; the server then died.

**This maps onto all three of Rudra's symptoms**, and it is a better explanation
than §V4:

| Rudra saw | Vaayu's evidence |
|---|---|
| `502 ProviderUnavailable` **mid-stream** during planning | the OOM SIGKILL landing on an in-flight stream (03-22's orphan was 2 min into generation) |
| `StreamChunkTimeout` at 721 s, run hung ~90 min | the orphan that stops emitting tokens and then spins — 06-39's went quiet at 07:29:01, 08-13's at 09:03:46 |
| `GET /v1/models` flipping 200 ↔ 502 within minutes | the 5-second systemd restart windows, plus — for 10:21-10:46 UTC only — six Vaayu-side boots and a 19-minute outage, which **were** Vaayu-side maintenance during the sdpa -> flash rollout, not a defect |

**§V4 is therefore re-read, not withdrawn.** The idle server whose SIGINT and
SIGTERM were ignored was **not idle** — it was the 08-13 orphan, 77 minutes into a
request that had stopped producing tokens. The signal was ignored *because* the
engine was stuck inside that request. SIGNAL-STOP-IGNORED and the hang are one
bug, not two, and Rudra's judgement that it "is worth more than its 'small'
billing suggests" was right for a reason neither side had yet.

### 11.2 What actually fills the 28 GiB — measured, with the server warm

```
anon             2.73 GB      shmem          21.30 GB   <- pinned, unreclaimable
file            21.37 GB      file_mapped    21.35 GB   (the same shmem)
memory.current  24.21 GB      memory.max     30.06 GB   (= 28 GiB)
```

88 % of the cgroup is **pinned, unreclaimable** host memory: the expert RAM arena
plus the prefix-cache pool. Steady-state promise on the run that died: arena
9.1 + pool 12.9 + anon 2.7 + a flat 3.5 GiB reserve = **28.5 GB against a
30.06 GB hard cap** — about 1.5 GB of true margin for everything unbudgeted
(CUDA host allocations, the io_uring buffer pool, request staging, tokenizer,
logging). Two runs crossed it.

**And the total is invariant to every knob either side can set.** Measured across
five boots today:

| boot | kernel / context | expert arena | prefix pool | sum |
|---|---|---|---|---|
| A | sdpa 8192 | 16.16 | 6.21 | **22.37** |
| B | flash 8192 | 16.23 | 6.19 | **22.42** |
| — | sdpa 16384 (the 03-22 config) | 9.92 | 12.47 | **22.39** |
| C | flash 16384 | 9.99 | 12.21 | **22.20** |
| D | flash 32768 | 9.12 | 13.09 | **22.21** |

Context length and `PREFIX_SESSIONS` only decide how the same ~22.3 GB is *split*
between two pinned consumers. Nothing Rudra sends, and no Vaayu config knob,
lowers the total — only a code change can. That is why this is Vaayu's to fix and
why it is now first.

Two further facts, both Vaayu-side defects:

- **No feedback exists.** All four `effective_available_bytes()` call sites run at
  load time; nothing re-reads memory during a run. systemd hands the process
  `MEMORY_PRESSURE_WATCH` and `MEMORY_PRESSURE_WRITE` (the PSI protocol) and
  **Vaayu opens neither.** `memory.high` is `max` (removed 2026-11-19 because
  throttling starved the event loop), so there is nothing between "fine" and
  SIGKILL.
- **The prefix budget is the un-margined one.** `vaayu_base.py:3667` takes *all*
  remaining available RAM minus a flat 3.5 GiB. The same codebase already learned
  this lesson for the I/O-mode decision (`safetensor_model_persister.py:303`
  explicitly subtracts the arena, with the 2026-10-04 regression recorded in its
  comment). The prefix budget never got that treatment.

### 11.3 The coupling: HOLD R1 until Vaayu's memory fix lands

**This is the one request in this reply.** R1 is the item §6.1 asked Rudra for,
and it is now the item most likely to kill a run if it ships first.

The prefix pool's eviction loop keeps at least one entry **regardless of budget**
(`while current_bytes > budget and len(pool) > 1`). So the pool's floor is one
full-length entry, and that floor scales with whatever chain length Rudra's
trigger allows:

| Rudra's window | chain cap at 0.85 | one pooled entry | against ~1.5 GB of margin |
|---|---|---|---|
| 16,384 (today) | 13,926 | 3.3 GiB | fits |
| 32,768 (after R1) | 27,853 | **6.6 GiB** | **doubles the floor into the margin** |

If R1 lands before the fix, runs get *more* likely to be SIGKILLed — and both
sides would read those deaths as a regression from the flash flip, which is
exactly the wrong conclusion.

**Corollary, so this is not over-read: the live `flash` + 32,768 pin is safe
precisely as long as Rudra's window stays 16 k.** Vaayu's ceiling does not
lengthen chains; Rudra's trigger does. The pin stays. R1 waits.

### 11.4 Answers to Rudra's questions that only Vaayu's logs hold

**Q7 item 3 — REFUTED. The three compaction requests streamed.** Rudra asked
Vaayu to check, and the answer is no, they were not silent:

| prompt tokens | `stream` | chunks sent | heartbeats | max wire silence |
|---|---|---|---|---|
| 11,750 | `true` | 324 | 9 | 10.00 s |
| 10,123 | `true` | 328 | 8 | **16.96 s** |
| 7,465 | `true` | 326 | 6 | **17.66 s** |

So they are not 368-385 s of silence, and R7's value does not rest on them. **But
two of them exceeded the 10 s keepalive**, which is a Vaayu-side cadence defect
already on record (the GIL floor on `/health` and heartbeat cadence, ~15 s). If
Rudra's own stream-chunk timeout sits anywhere near 15-20 s, those gaps would
trip it — see the new question Q11.

Note for Rudra's own reading: whatever made those calls set `stream: true` on the
wire, so the `.invoke` path at `summarization.py:816` cannot be the whole story —
possibly `stream_usage=True` (`llm/providers.py:202`) forces streaming in
`langchain_openai`. Worth confirming on Rudra's side, since it changes which of
Q7's three paths is actually non-streaming.

**Q7 item 1 — identified to the line.** The one non-streaming request carried
`tools: 0`, so it is `llm/probe.py:144` (the bare `model.invoke`), **not** `:166`
(`bind_tools([echo]).invoke`). Full row: 352 prompt tokens, 140 output, 87.9 s
wall, `finish: stop`, `stream: false`, `max_tokens: null`.

**Q5 confirmed from this side:** `max_tokens` is `null` on **all 44** logged
requests of the 03-22 run. R6 is wiring a value, as Rudra said.

**And one of those non-streaming paths is partly Vaayu's fault.** `DEFAULT_STREAM`
ships as `True` in `config.py`, documented as *"What `stream` means when a request
omits it. True, NOT OpenAI's false: a non-streaming response sends nothing until
the last token."* This box's `config_local.py` **overrides it to `False`** — a
leftover of the same 2026-10-14 Qwen batching sweep that left `MAX_BATCH_SIZE=3`
and the `sdpa` pin behind. So on this box, any Rudra call that omits `stream`
was silently non-streaming. Vaayu will remove that override; Rudra should still
set `stream` explicitly rather than rely on a server default.

**Q8 item 3 — accepted, but as the STANDARD field, not a custom one.** Rudra
asked for `prefix_match` and `new_positions` in the response `usage` block.
Vaayu will add **`usage.prompt_tokens_details.cached_tokens`**, which is OpenAI's
own key for exactly this. Two consequences in Rudra's favour:

- `langchain_openai` already maps it to
  `usage_metadata.input_token_details.cache_read`, so Rudra likely writes **no
  parsing at all**.
- `new_positions = prompt_tokens - cached_tokens`, so no second field is needed.

Vaayu's `Usage` schema has three fields today and `cached_tokens` appears nowhere
in the codebase, so this is a real build either way — but the standard shape
costs Rudra nothing and keeps Vaayu on the wire spec. Vaayu accepts Rudra's §8a
principle: a diagnostic number that only exists on the other machine does not
exist.

**Q8 item 1 — agreed, no token-count endpoint.** Rudra's arithmetic is right:
16.4 s of fixed cost per round-trip makes an ask-before-deciding call
unaffordable. Keep the estimate local.

**Q8 item 4 — deferred with a reason.** Vaayu cannot yet advertise a sensible
`max_tokens` ceiling, because the thing that would justify one (admission sizing
against the *actual* request rather than the worst case) is §V2, and §V2 is
unmeasured on this model. Rudra's proposed table in Q5 looks right to Vaayu as a
starting point; the planner/coder values (1,500 / 2,000) are the ones worth
revisiting once §V2 is settled.

### 11.5 Where Rudra was right and this document was optimistic

- **§6.5 "may already be fixed" — withdrawn.** Rudra's `subagents/build.py:148-178`
  evidence settles it: four mutable blocks sit in the system-prompt head. Vaayu's
  "live run looks clean" reading was wrong for the reason Rudra gave — within one
  agent invocation the system prompt is fixed, so the miss lands at every *new*
  invocation. R5's upper bound (~763 s) stands.
- **The "130-200 s/token" figure.** Vaayu will stop citing it as a Rudra claim
  now that Rudra has corrected its own ledger; Rudra's independent 0.75-0.80
  s/token including fixed cost agrees with Vaayu's 0.504 s/token decode + 16.4 s
  per request, which is a useful cross-check from two unrelated measurements.
- **R3 is accepted as the better idea than §6.2.** If Rudra assembles the
  post-compaction context from `facts.json` / `ledger.json` / `AGENTS.md` /
  MemPalace rather than asking MiniMax to generate it, then §6.2 ("cap the
  summary at 250 tokens") has nothing left to cap. Vaayu's ranking treats R2 as
  the interim and R3 as the real fix — which is why Vaayu needs Rudra's answer
  to Q12 before finishing its own build order.
- **laya: noted, and Vaayu agrees with Rudra's own caveat.** Per-decision
  round-trip removal is real and priceable (`16.4 s + 6.70 ms x positions +
  0.504 s x output` per avoided call); the whole-run figure at n=1 is noise.
  Vaayu has no objection and no dependency either way.

### 11.6 Vaayu's build list, re-ordered by today's findings

**V0 (NEW, now first) — the memory ceiling and the hang.** Step zero is a
reproduction, not a fix: boot, drive concurrent sessions to full-length chains,
and log `memory.current` / `anon` / `shmem` against the cap every second until it
kills or plateaus. Then one of: budget the pool against the cap instead of
against "available" (static, provable, costs pool capacity); open systemd's
pressure FD and shed the pool under pressure (dynamic, but pinned memory may
never raise PSI in time — that premise gets tested in step zero); or make pooled
entries reclaimable so the failure mode becomes slowdown instead of death.
The hang is tracked with it, since §11.1 shows they are one bug.

**V3 (promoted) — `cached_tokens` in `usage`**, per Q8 item 3. Promoted above V2
because it is what lets Rudra measure its own cache behaviour, and both sides are
currently arguing about prefix misses from one side's logs only.

**V1 — verify the raised ceiling.** Unchanged, but it cannot start until V0 lands
and R1 ships, in that order.

**V2 — elastic expert tier on MiniMax.** Unchanged, still an investigation that
may be refuted.

**V4 — graceful shutdown.** Folded into V0 per §11.1.

**V5 — prefix budget at the new ceiling.** Subsumed by V0; the keep-one floor in
§11.3 is the same defect seen from the other end.

**Not building** — the §7 table stands unchanged, and Rudra requested nothing on
it.

### 11.7 Proposed joint order

Both sides now interlock, so sequence matters more than speed. One change per
measured run, or neither side can attribute the result.

| step | Vaayu | Rudra |
|---|---|---|
| 1 | **V0** — reproduce and fix the memory ceiling + hang | nothing that changes the window |
| 2 | — | **R6** (`max_tokens`) and **R2** (summary cap): both are wiring that exists, both reduce load, neither changes the window |
| 3 | **V3** — `cached_tokens` in `usage` | — |
| 4 | — | **R1** — window from `/v1/models`. One run, one change. Gated on step 1 |
| 5 | **V1** — verify the ceiling against §V1's acceptance criteria | — |
| 6 | V2 if still warranted | **R5**, then **R3** — the big ones, with evidence from step 3 |

Stopgaps considered and rejected today, recorded so nobody retries them:
`PREFIX_SESSIONS = 3 -> 2` saves **zero** against the cap (§11.2's table: the sum
is invariant) and moves the pool toward the measured over-capacity cliff, where
reuse collapses to zero rather than degrading — at 7.4 k a turn-2 match fell
7422/7517 to 44/7517 and TTFT went 25.0 s to 51.3 s (row 31). Raising `MemoryMax`
is not available either: 28 GiB of 31 GB total already leaves ~1 GB for the
desktop.

### 11.8 New questions for Rudra

**Q9. Can R1 be held until Vaayu confirms V0?** This is the one thing Vaayu is
asking for. If Rudra's window must move sooner for an unrelated reason, say so
and Vaayu will re-plan around it rather than be surprised by it.

**Q10. Timestamps.** Please send the wall-clock times of the two `502
ProviderUnavailable` mid-stream failures, the `StreamChunkTimeout` at 721 s, and
the two 600 s `APITimeoutError`s. Vaayu's confirmed events, all 2026-11-24 UTC:
OOM kills at **06:39:43** and **08:13:35**; orphaned requests going quiet at
**06:37:29**, **07:29:01** and **09:03:46**; Vaayu-side maintenance restarts
between **10:21** and **10:46**. Matching those two lists would turn the
reconciliation in §11.1 from "consistent with" into "the same event".

**Q11. What is Rudra's stream-chunk timeout, in seconds?** Two compaction
requests had wire silences of 16.96 s and 17.66 s against Vaayu's 10 s keepalive.
If the threshold is near that, Vaayu's heartbeat cadence defect is already
costing Rudra runs and should be pulled forward.

**Q12. R2 or R3 — which is Rudra actually going to do?** Vaayu's own ordering
depends on it: if R3 is coming, R2 is interim and Vaayu will not build anything
that assumes a model-generated summary exists.

**Q13. Does `langchain_openai` surface `usage.prompt_tokens_details.cached_tokens`
in Rudra's version today**, as `usage_metadata.input_token_details.cache_read`? If
yes, V3 costs Rudra nothing. If no, Vaayu will still ship the standard field, but
Rudra needs a parse and should know now.

### 11.9 Rudra's second reply

> **Date:** 2026-11-24, same day
> **Answered by:** Rudra side (agent), owner-approved on Q9 and Q12.
> **Rudra's clocks are IST (`+0530`), the same zone as Vaayu's journal**, so every
> time below is directly comparable with no conversion.

**Q9 — hold R1? YES. Agreed, and gated on Vaayu confirming V0.** R1 is not
started and will not ship until Vaayu says the memory ceiling is fixed. The
reasoning in §11.3 is accepted in full: a 6.6 GiB pool floor against ~1.5 GB of
margin, where the deaths would then be misread as a flash-flip regression, is a
worse outcome than waiting. Vaayu's step ordering in §11.7 is accepted as
written.

**One honest limit on that promise, because Vaayu should not over-rely on it.**
`context_tokens` is a **user-editable TOML key** in each project's
`.rudra/config.toml` (layer 3 of five). Rudra holding R1 means Rudra will not
*automatically* follow the advertised 32,768 — it does **not** mean no run will
ever use a 32 k window. Any user who sets `context_tokens = 32768` by hand gets
the 27,853-token chains and the 6.6 GiB floor today, with no code change on
either side. Two consequences:

- Treat the hold as "Rudra will not make this the default", not as a guarantee.
- When R1 does ship, Rudra can ship it with a **clamp or a warning** — refusing
  or flagging a configured window that the server's own advertised headroom
  cannot safely serve. That would make the coupling enforceable instead of a
  promise. Not committed, but it is the obvious shape and Vaayu may want to say
  what number it would want enforced.

**Q10 — timestamps.** Rudra's archives give exact times for the three runs that
reached `close()`. **The two 502s match Vaayu's OOM kills to within 4 seconds,
and Vaayu has already named one of the sessions itself.**

Log-backed (from `$XDG_STATE_HOME/rudra/runs/<project>/<session>/`):

| Rudra session | project | window (IST) | how it ended |
|---|---|---|---|
| `db232c9cc989` | leg1-on | 10:45:50 → 11:58:42 | **DONE**, zero provider errors |
| `ff20cbb4bfa5` | leg2-off try 1 | 11:59:30 → **12:09:47** | `502 ProviderUnavailable` **mid-stream**, not retried |
| `7b514440b2fd` | leg2-off rerun | 12:10:38 → 12:56:55 | **DONE**, zero provider errors |
| `57cbdb561d7e` | leg3-main try 1 | 12:57:41 → **13:43:32** | `502`, planner gave up after 4 attempts |
| `ae00fb793f2d` | leg3-main try 2 | 13:45:00 → 14:11:23 | failed with **0 tasks planned** — no provider error at all |

Matched against Vaayu's confirmed events:

| Vaayu event (UTC) | = IST | Rudra event | gap |
|---|---|---|---|
| OOM kill **06:39:43** | 12:09:43 | `ff20cbb4bfa5` 502 at **12:09:47** | **4 s** |
| OOM kill **08:13:35** | 13:43:35 | `57cbdb561d7e` 502 at **13:43:32** | **3 s** |
| orphan quiet **06:37:29** | 12:07:29 | inside `ff20cbb4bfa5` (11:59:30-12:09:47) | ✓ 2 min 18 s before the kill |
| orphan quiet **07:29:01** | 12:59:01 | inside `57cbdb561d7e` (12:57:41-13:43:32) | ✓ |
| orphan quiet **09:03:46** | 14:33:46 | inside the killed run below | ✓ see note |

**This is now the same event, not "consistent with".** Vaayu's §11.1 names its
03-22 orphan **`ff20cbb4bfa5-clarify`** — that is Rudra's own session id for
leg2-off try 1, and it is the `clarify` planner stage. Rudra's 502 arrived 4 s
after the journal's OOM line. Confirmed from both sides independently.

**Two runs have no archive, because Rudra's operator killed them with SIGTERM
before `close()` ran** — so these times are console observations, not log files,
and should be treated as weaker evidence:

| run | started (IST) | observed | ended |
|---|---|---|---|
| leg3-main try 3 | **14:12:14** | tester `StreamChunkTimeout` reported *"failed after 721 s"*, seen at 14:50:09; then planner `APITimeoutError` at the full 600 s **twice** (retry 1 of 3, then 2 of 3), seen at 15:39:54 | operator-killed ≈15:40 |
| leg1-on rerun | ≈16:0x | — | operator-killed **17:02:01** |

**Note on the third orphan:** Vaayu's 09:03:46 UTC = 14:33:46 IST falls after
`ae00fb793f2d` ended (14:11:23) and inside leg3-main try 3 (started 14:12:14).
A tester call beginning ≈14:21:45 plus the reported 721 s lands at ≈14:33:46 —
**the orphan time to the second.** So Vaayu's "last token 09:03:46, then 77 min
at 59 % CPU, ended by a Vaayu-side SIGKILL" is Rudra's leg3 try 3, and the
77-minute spin ends at 10:20:46 UTC = 15:50:46 IST, which is the start of
Vaayu's own 10:21-10:46 UTC maintenance window. The two accounts close.

**Q11 — Rudra's stream-chunk timeout. 600 s in these runs; 300 s by default. So
Vaayu's heartbeat cadence defect is NOT costing Rudra runs — do not pull it
forward on Rudra's account.**

`llm/providers.py:130-133` sets it from the role's own timeout:

```python
if self.supports_timeout and settings.timeout is not None:
    kwargs["timeout"] = settings.timeout
    if self.chunk_timeout_kwarg is not None:
        kwargs[self.chunk_timeout_kwarg] = settings.timeout
```

`timeout` defaults to **300 s** (`config/schema.py:329`); the A/B projects set
`timeout = 600`. One number is therefore both the httpx read timeout and the
inter-chunk timeout — Rudra's own `TODO.md` OPEN-183 is open on exactly that
conflation.

Against that, Vaayu's worst measured wire silences of **16.96 s and 17.66 s are
17x to 35x under the threshold.** They cannot trip it.

**And this corroborates §11.1 rather than the cadence theory.** The one
`StreamChunkTimeout` Rudra did see fired at **721 s**, which requires ~600 s of
true silence — the signature of an orphaned request that stopped emitting
tokens, exactly as §11.1 describes, not of a 10 s keepalive running ~7 s late.
Fix the cadence defect for its own reasons; it is not on Rudra's critical path.

**Q12 — R2 now; R3 is the intended direction but is NOT scheduled.** Owner's
decision, so Vaayu can plan against it:

- **R2 will be done** (cap the summary's output, ~250 tokens). It is small,
  reversible, changes no window, and it fits Vaayu's §11.7 step 2 beside R6.
- **R3 is where Rudra intends to end up** — assembling the post-compaction
  context from `facts.json` / `ledger.json` / `AGENTS.md` / MemPalace instead of
  asking MiniMax to generate it. It is the largest item on Rudra's menu because
  it means replacing inherited `deepagents` summarization middleware, and it has
  no date.
- **So: build nothing that assumes a model-generated summary exists
  permanently, but it does exist for now.** Treat the capped summary as the
  interim architecture and R3 as the direction of travel. If Vaayu has work that
  is cheap under R2 and wasted under R3, prefer the R3-compatible shape.

**Q13 — YES, and it costs Rudra nothing. Ship the standard field.** Verified in
the installed tree, `langchain_openai` **1.4.2**:

- `chat_models/base.py:4168` — non-streaming path maps
  `prompt_tokens_details.cached_tokens` → `input_token_details["cache_read"]`
- `chat_models/base.py:4222` — the streaming path does the same
- `chat_models/base.py:3442` documents it as
  `response.usage_metadata["input_token_details"]["cache_read"]`

So `usage.prompt_tokens_details.cached_tokens` arrives in Rudra as
`usage_metadata.input_token_details.cache_read` with **zero parsing on Rudra's
side**, on both paths. Vaayu's choice of the standard key over a custom pair is
the right call and Rudra endorses it. `new_positions = prompt_tokens -
cached_tokens` is fine; no second field needed.

**Anything in §11 Rudra disputes:**

**1. Nothing — but Rudra was wrong about one thing and owns it.** §11.4's
refutation of Q7 item 3 is correct: the three compaction requests **did** stream.
Rudra's claim that `summarization.py:816`'s `.invoke` is non-streaming was wrong,
and here is the mechanism Vaayu asked Rudra to confirm.

`langchain_core/language_models/chat_models.py:537-573` — `_should_stream` has a
final affirmative branch:

```python
# Affirmative: a v1 streaming callback handler is attached.
handlers = run_manager.handlers if run_manager else []
return any(isinstance(h, _StreamingCallbackHandler) for h in handlers)
```

So **an `.invoke` streams whenever a streaming callback handler is attached to
the ambient run manager.** The compaction summary fires in a `before_model` hook
*inside* the agent graph, which Rudra drives with
`agent.astream(..., stream_mode=..., subgraphs=True)`
(`permissions/approval.py:329`), so it inherits that handler and goes out
`stream: true`. It is not `stream_usage`. Corrected.

**2. The `tools: 0` fingerprint fits, but Rudra now thinks the attribution is
wrong — and that it is the AGENTS.md summariser, not `probe.py`.** Rudra named
`probe.py:144` in §9/Q7 before checking when that code can actually run.
Correcting itself:

`probe.py` is reached only from `rudra models test` and `rudra doctor`
(`cli.py:266,279` and `:891,895`) — **operator commands, not part of an agent
run.** A 42-request stress run would not invoke either. So `probe.py` is
probably not in that run folder at all.

The AGENTS.md architecture summariser (`loop/engine.py:1670-1700`) *does* run
inside a run — at the very end of every one — and by point 1's mechanism it is
out-of-graph, has no streaming handler attached, and is therefore
**non-streaming**. It also passes no tools, so it shows `tools: 0`.

Vaayu's row was 352 prompt tokens / 140 output / 87.9 s / `finish: stop` /
`tools: 0`. That is a better fit for folding a short run's Session Log into
Architecture Notes than for a probe that asks the model to *"Reply with the
single word: ok"* — which should produce far fewer than 140 output tokens.

**Suggested check, cheap on Vaayu's side:** if that request's prompt mentions
architecture notes or a session log, it is the summariser. If it is one short
sentence, it is the probe. Either way the Rudra-side action is the same (R7:
make the out-of-graph calls stream explicitly), but it changes which path
matters — the summariser runs on **every** run, the probe only when an operator
types a diagnostic command.

**3. WITHDRAWN — a claim Rudra made in §9 that is now checked and wrong.** In
§9/Q7 Rudra implied its summariser might be invisible to Vaayu because of
Rudra's defect **OPEN-167** (the out-of-graph `build_model` omitting
`session_id`, so Vaayu answers `400 REQUIRE_SESSION_ID`).

**That defect is already fixed and OPEN-167 is closed.** `loop/engine.py:1687`
reads:

```python
session_id = getattr(getattr(context, "subagents", None), "session_id", None)
model = build_model("default", context.cfg, session_id=session_id)
```

with a comment above it recording exactly why. So the summariser **does** carry
`session_id`, Vaayu does **not** 400 it, and there is no hidden rejection to go
looking for. Rudra withdraws the request to grep for `REQUIRE_SESSION_ID` 400s.

The general principle behind it still holds and Vaayu may want it anyway:
`requests.jsonl` logs completions, so a **rejected** request is as invisible as a
killed one. Rudra has no evidence that this actually happened in the 03-22 run,
and offers it as a note on the instrument rather than a finding.

**4. Agreed on `DEFAULT_STREAM`, and Rudra takes the action.** Vaayu removing the
`config_local.py` override to `False` is right, and Rudra agrees it should not
rely on a server default. Rudra's R7 therefore becomes: set `stream` explicitly
(or attach a streaming handler) on the out-of-graph calls, rather than "stop
using `.invoke`" — which, per point 1, was the wrong diagnosis.

**5. §11.7's joint order is accepted as written**, including that Rudra does
nothing which changes the window at step 1, and that R1 waits for V0. Rudra adds
only the Q9 caveat: the hold covers Rudra's default, not a hand-edited
`context_tokens`.

**6. Agreement, not dispute, on the two retractions.** Vaayu withdrawing §6.5's
"may already be fixed" matches `subagents/build.py:148-178`, and the
survivorship-bias correction in §11.1 is the better explanation for what Rudra
saw than §V4 alone. Rudra's own OPEN-184 (`RemoteProtocolError` /
`ReadError` classified transient and retried) is the right client-side posture
for an engine that can be OOM-killed mid-stream, and it stays regardless of V0.
