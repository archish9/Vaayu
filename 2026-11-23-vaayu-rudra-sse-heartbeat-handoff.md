# Vaayu → Rudra handoff — SSE heartbeat & request pacing (2026-11-23)

Follow-up to the 2026-11-22 slow-decode handoff (already shared). This one is standalone and
covers only what changed since then and what Rudra needs to do. You do **not** need the earlier
doc to act on this.

## TL;DR

On the 2026-11-23 retest Vaayu did **not** crash and did **not** run out of memory. The requests
were sitting in Vaayu's queue, and the client gave up because the old keep-alive was an SSE
*comment* (which clients ignore), not a data chunk. Vaayu has now fixed its side. Rudra needs two
pacing changes: **send one request at a time**, and **raise the read/stream timeout**.

## What we saw (run 2026-11-23T05-23-12Z)

- Vaayu was up the whole time (~6 h), healthy, no crash, no OOM.
- `/health` reported `queue_depth 8, queued 5` — several Rudra requests were **queued at once**.
- Every one of those requests ended `finish=cancelled` with `cancel_reason=abandoned_queued`:
  the client disconnected **while the request was still waiting in the queue**, before Vaayu ever
  started generating it.
- On the wire: `heartbeats_sent=0`, `comments_sent=36`, longest silence ~611 s.

## Why it happened

- Vaayu serves **one request at a time** (by design on this box). The model (MiniMax-M2.7 on
  16 GB VRAM / 28 GB RAM) decodes at **130–200 seconds per token**, and prefill on a large prompt
  takes minutes. So a request can wait **many minutes** in the queue plus prefill before its first
  token.
- During that wait Vaayu used to send a `: keepalive` **SSE comment**. Comments are valid SSE but
  clients that time out on "no **data chunk** for N seconds" do not count them — so to Rudra the
  socket looked silent and it closed the connection.

## What changed in Vaayu (shipped 2026-11-23)

1. **Keep-alive is now an empty-content data chunk, not a comment.** During the queue/prefill wait
   Vaayu now sends a normal streaming chunk with `delta: {"content": ""}` and no `finish_reason`,
   on the keep-alive interval. It carries no text, cannot end the stream, and never changes the
   answer — but it **does** reset a "no data chunk for N seconds" client timer. (The same chunk
   already covered the buffered tool-call window; both silent windows now behave the same.)
2. **Keep-alive interval default lowered to 10 s** (`SSE_KEEPALIVE_S`).

So after Vaayu restarts with this build, a streaming client that counts data chunks will see a
chunk at least every ~10–15 s even while a request waits or prefills.

### Limit to know about (not fixed)

Under heavy decode, Vaayu's event loop is serviced only about **every ~15 s** (a GIL effect from
the model's compute). Consequences for Rudra:

- `/health` can take up to ~15 s to answer while a generation is running (9.2 s observed). **Do not
  use a health check with a timeout shorter than ~20 s** to decide Vaayu is dead — it isn't.
- The heartbeat cadence under load is effectively **~15 s regardless of the 10 s setting**. Size
  client timeouts against ~15 s, not 10 s.

## What Rudra needs to do

1. **Send one request at a time (concurrency = 1).** Vaayu advertises
   `recommended_client_concurrency: 1` on `/v1/models` and `/health`. Batching/fan-out is a net
   **loss** on this streamed-MoE model, and firing several at once just fills the queue and causes
   the `abandoned_queued` cancellations above. Wait for one request to finish before sending the
   next.
2. **Raise the read/stream (chunk) timeout.** Make it minutes-safe — well above the per-token time.
   This box is a deliberate stress target; a single token can take 130–200 s, and prefill longer.
   A 30–60 s chunk timeout **will** give up mid-answer. Treat an empty-content chunk as liveness
   (keep waiting), not as the end of the stream.
3. **Keep `stream: true`.** A non-streamed request returns nothing until the final token and will
   outlive any client.
4. **Health checks:** timeout ≥ ~20 s, and treat a slow `/health` as "busy," not "down."

## Not Rudra's to fix / not fixable here

The raw **130–200 s/token** speed is the model on this hardware, not a bug and not something the
heartbeat changes. It only goes away with a smaller/fitting model, which is deliberately not being
used on this box right now. The heartbeat + pacing changes stop the **false "it's dead" timeouts**;
they do not make the model faster.
