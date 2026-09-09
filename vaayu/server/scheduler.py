"""The dispatcher: the ONLY thing in this server that runs a generation.

Routes build a Job, submit it, and read its queue. One dispatch task drains
the wait queue, forms a batch, holds `engine._lock` for the whole run, and
repeats -- so "one generation at a time" is unchanged; the unit is now a batch
of one OR N.

WHY A DISPATCHER RATHER THAN A COALESCING GATE. `batched_generate`'s docstring
treats an abandoned generator as real: it leaves the model in batched-decode
mode until GC, and a client disconnect is exactly that abandonment. Here the
generator lives inside run_batch and clients hold queues, so no client CAN
abandon it. Every other shape puts that contract in two places.

N=1 NEVER GOES THROUGH batched_generate. Qwen3-235B's admission grants N=1,
and driving that one request through the batched path measured 573.88 s
against 177.14 s serial -- a 3.24x regression for every Qwen user. The rule is
one branch in `_run`, and tests/test_server_scheduler.py pins it.
"""
import asyncio

from .batch_policy import Candidate, select_width
from .engine import QueueFull
from .logs import clear_current_session, set_current_sessions


class Job:
    """One request, as the dispatcher sees it.

    `out` receives ("delta", text)*, then one ("final", result) or
    ("error", exc), then None as the end sentinel. Unbounded on purpose: a
    slow client must not stall the other N-1 streams of its batch, and each
    stream's output is bounded by its own cap.

    ("dispatch_error", exc) IS THE FOURTH KIND, AND IT MEANS THE ENGINE
    NEVER RAN. `engine.stream()`, `engine.generate()` and `run_batch` each
    write this request's requests.jsonl row from their own `finally`, so a
    route that sees a real item hands row-writing to them ("the engine owns
    the row"). The dispatch loop can also terminate a job the engine never
    entered -- stranded in `_waiting` at loop exit, or fanned out from its
    `except BaseException` -- and nothing wrote a row for those. The routes
    read the kind and keep ownership, which is what keeps the "a request
    never ends without a row" contract (documentation/server.md) true on
    those paths. Otherwise the row would be lost: it looked like an ordinary
    error, so both routes assumed a row already existed.

    `prepared` and `stream` are set by the route, not passed to `__init__`
    (both default so existing call sites -- and the brief's own test file,
    which never touches either -- keep working):
      * `prepared`: `engine.prepare(req)`'s result (prompt_ids, cap), already
        computed by the route before this Job existed. `_run_serial` passes it
        through so a serial run never re-renders the chat template and
        re-tokenizes a prompt that is already tokenized.
      * `stream`: whether this request came in as `stream=true`. A batch
        (N>=2) always uses `run_batch` regardless of it; a lone job honours it
        by calling `engine.generate()` (non-streaming, own speculative
        dispatch, `stream=false` in requests.jsonl) or `engine.stream()`.
    """

    def __init__(self, req, session_key, prompt_ids, cap, cancel):
        self.req = req
        self.session_key = session_key
        self.prompt_ids = prompt_ids
        self.cap = cap
        self.cancel = cancel
        self.out: asyncio.Queue = asyncio.Queue()
        self.peek_len = 0
        # Task 8 sets these on the Job it builds; a caller that never sets
        # them (this file's own test suite) gets `_run_serial`'s streaming
        # path with a fresh prepare() -- today's default behaviour.
        self.prepared = None
        self.stream = True


class BatchScheduler:
    # Formation attempts with two or more waiting requests before the "never
    # admitted" line prints. Enough that a couple of unlucky warm-prefix
    # pairs do not trigger it; small enough that a misconfigured server says
    # so in the first minutes rather than in a probe months later.
    NEVER_ADMITTED_AFTER = 20

    def __init__(self, engine, queue_depth: int = 8, max_batch_size: int = 1,
                 prefill_chunk_size: int = 512, batch_wait_s: float = 0.05):
        self.engine = engine
        self.queue_depth = queue_depth
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        # How long a formation waits for company before running what it has.
        # Zero restores the pre-2026-09-09 behaviour, which never batched two
        # concurrent clients at all -- see `_coalesce`.
        self.batch_wait_s = batch_wait_s
        self._waiting: list = []
        # The batch currently running, if any -- tracked separately from
        # `_waiting` because `_dispatch_loop` removes members from `_waiting`
        # BEFORE running them. `depth` (below) must still count them: today's
        # `slot()` increments its own counter before awaiting the lock so a
        # client's queue-position line and the 503 threshold both count the
        # request that is already running, and this dispatcher keeps that
        # promise the same way.
        self._running: list = []
        self._arrived = asyncio.Event()
        self._task = None
        self._stopping = False
        # Counted so the server can say it out loud when batching never
        # admits -- the CLAUDE.md section 8 failure this row exists to end is
        # a feature that ships and quietly does nothing. Split by WHICH gate
        # refused (B1, final whole-branch review): gate 1 is this scheduler's
        # own pre-check (`_admit`'s plan_batch pricing) narrowing a would-be
        # batch to one before `batched_generate` is ever called; gate 2 is
        # `batched_generate` itself refusing at the real width and falling
        # back to serial in `_run`. The on-device measurement needs to tell
        # a gate 1 that is still too conservative apart from a gate 2 that
        # hit a real OOM ceiling (MiniMax N=3, P10a) -- one counter conflated
        # them.
        self.gate1_refused = 0
        self.gate2_refused = 0
        self.batches_run = 0
        # CLAUDE.md section 8's guard: attempts counted (2+ waiting jobs,
        # batching enabled) so the on-device probe -- and the latched notice
        # below -- can tell "never formed a batch" from "formed and did not
        # help".
        self._formation_attempts = 0
        self._never_admitted_warned = False
        if self.max_batch_size > 1 and not self.batching_enabled:
            # ATTN_IMPLEMENTATION = None looks like sdpa in a stale config.py
            # comment but actually prefers flash_attention_2 whenever
            # flash_attn imports -- so an operator going by that comment gets
            # this exact silent-inert knob. Name it here so they don't have
            # to find that out from a probe.
            print(f"[Vaayu-Server] WARNING: MAX_BATCH_SIZE is "
                  f"{self.max_batch_size} but this model loaded on "
                  f"{getattr(engine.model, '_active_attn_impl', None)!r}. "
                  f"Batched decode requires sdpa, so requests will be served "
                  f"ONE AT A TIME and the knob is inert. Set "
                  f"ATTN_IMPLEMENTATION = \"sdpa\" in vaayu/server/config.py "
                  f"to enable it (the kernel itself costs 2-4 % end-to-end).")

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def depth(self) -> int:
        return len(self._waiting) + len(self._running)

    @property
    def batching_enabled(self) -> bool:
        """batched_generate requires sdpa. An operator who set
        flash_attention_2 gets the serialized path -- decided ONCE, here,
        never as a live assert on the request path."""
        return (self.max_batch_size > 1
                and getattr(self.engine.model, "_active_attn_impl", None) == "sdpa")

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._dispatch_loop())

    async def stop(self) -> None:
        self._stopping = True
        self._arrived.set()
        if self._task is not None:
            task, self._task = self._task, None
            try:
                await task
            except asyncio.CancelledError:
                # Two different cancellations arrive at this one `await`,
                # and they must NOT be treated alike:
                #   * the DISPATCH TASK was cancelled (Task 8's app-lifespan
                #     path, not this cooperative one) -- awaiting it here
                #     re-raises that same CancelledError. `stop()` must
                #     still return cleanly: `self._task` is cleared
                #     unconditionally, above, so a caller sees the
                #     dispatcher as stopped either way, and letting this
                #     escape would surface in the lifespan's own shutdown
                #     code as an exception, easily misread as the lifespan
                #     task's OWN cancellation.
                #   * the CALLER of stop() was itself cancelled while
                #     suspended here. Swallowing THAT breaks cancellation
                #     for whoever asked -- the shutdown coroutine would
                #     sail on as though it had never been cancelled.
                # `task.cancelled()` ALONE DOES NOT SEPARATE THEM, and this
                # is the whole subtlety: cancelling a task that is suspended
                # on `await other_task` propagates the cancel INTO
                # `other_task`, so in the second case the dispatch task ends
                # up cancelled too and `task.cancelled()` is True either way
                # (verified directly against this asyncio, both cases). What
                # does separate them is the CALLER's own cancellation count:
                # `Task.cancel()` bumps `_num_cancels_requested` on the task
                # it was called on before propagating inward, so
                # `current_task().cancelling()` is non-zero exactly when it
                # was US who were cancelled. `not task.cancelled()` is kept
                # alongside it for the remaining shape -- the dispatch task
                # absorbing a cancellation and ending some other way.
                caller = asyncio.current_task()
                if (caller is not None and caller.cancelling()) or not task.cancelled():
                    raise

    async def submit(self, job: Job) -> None:
        """Enqueue `job`. Raises QueueFull past `queue_depth`, and also when
        there is no live dispatcher to drain it.

        The count includes the running batch's members, so a client's queue
        position is `depth` at entry -- the same promise slot() made.

        THE INVARIANT THIS GUARD ENFORCES: a job is only ever in `_waiting`
        while a dispatcher exists to drain it. Every job that enters here
        must end with a sentinel on `job.out`, and the only thing that ever
        writes one is the dispatch loop -- so enqueuing when that loop is
        gone (stopped, or dead from an exception out of `_form`/`_admit`,
        which `start()` deliberately does not resurrect) hands the caller a
        queue nobody owns and an `await job.out.get()` that never returns.
        A refusal is the honest answer: the routes already turn QueueFull
        into a 503, which is exactly "this server cannot currently accept
        work". The message says WHICH refusal it is, so an operator reading
        the log can tell a dead dispatcher from a genuinely full queue.
        """
        if self._stopping or self._task is None or self._task.done():
            raise QueueFull(
                "the dispatcher is not running (stopped, or its task died); "
                "this server cannot currently accept work. This is NOT a "
                "full queue -- the request was refused before it was "
                f"queued (depth {self.depth} of {self.queue_depth}).")
        if self.depth >= self.queue_depth:
            raise QueueFull(
                f"{self.depth} requests are already queued (depth "
                f"{self.queue_depth}).")
        job.peek_len = self._peek(job)
        self._waiting.append(job)
        self._arrived.set()

    def _peek(self, job) -> int:
        """What this prompt would get from the prefix pool on its own -- the
        reuse batching would spend. Non-destructive (peek_match never pops)."""
        pc = getattr(self.engine.model, "_prefix_cache", None)
        if pc is None:
            return 0
        try:
            return pc.peek_match(job.session_key, job.prompt_ids)
        except Exception:                        # noqa: BLE001 -- never blocks
            # A formation heuristic must never fail a request. Zero means
            # "assume no warm prefix", which biases towards batching and is
            # what the acceptance A/B measures anyway.
            return 0

    # ── the loop ─────────────────────────────────────────────────────────────

    async def _dispatch_loop(self) -> None:
        try:
            while not self._stopping:
                if not self._waiting:
                    await self._arrived.wait()
                    self._arrived.clear()
                    continue
                await self._coalesce()
                members = await self._form()
                for j in members:
                    self._waiting.remove(j)
                if not members:
                    # Everything waiting was a client that vanished before
                    # its turn (see `_form`) -- nothing to run this
                    # iteration.
                    continue
                self._running = members
                try:
                    await self._run(members)
                except BaseException as ex:      # noqa: BLE001 -- see below
                    # run_batch's own `pending_base_exc` (a CancelledError /
                    # KeyboardInterrupt surfacing while finalising one job) is
                    # a BaseException `_run`'s `except Exception` does not
                    # catch. Left unguarded this unwinds straight out of the
                    # loop and kills the dispatcher task -- every later
                    # request then waits forever on a queue nobody will ever
                    # write to again. Deliver what this batch's own clients
                    # can still be told, on every such BaseException.
                    # (Harmless if some of `members` already got a real
                    # sentinel from `_run` before this fired -- their client
                    # already returned at the first one and never reads a
                    # second. A job's contract is therefore "at least one
                    # sentinel, stop at the first" -- never exactly one.)
                    #
                    # "dispatch_error", not "error": for a member the engine
                    # never ran, nothing wrote a requests.jsonl row, and the
                    # route only knows that from the KIND (see Job's
                    # docstring). The gate-2 fallback's `members[1..N-1]`
                    # arrive here that way -- `_run_serial(members[0])`
                    # re-raising a BaseException skips them entirely.
                    for j in members:
                        await j.out.put(("dispatch_error", ex))
                        await j.out.put(None)
                    if isinstance(ex, asyncio.CancelledError):
                        # A dead dispatcher must not be a possible end state
                        # for a CRASH -- but cancellation is not a crash, it
                        # is a request to stop (Task 8 wires this into the
                        # app lifespan), and swallowing it here would mean a
                        # cancelled dispatch task never actually ends:
                        # shutdown would hang waiting for a task that keeps
                        # looping. Notified above; now let the cancellation
                        # through.
                        raise
                    print(f"[Vaayu-Server] WARNING: the dispatcher caught "
                          f"{type(ex).__name__}: {ex} running a batch of "
                          f"{len(members)}; its members were notified and "
                          f"the dispatcher keeps running.")
                finally:
                    self._running = []
        finally:
            # Every job that ever entered submit() must end with a sentinel,
            # on every path the dispatcher can exit -- and this is the one
            # the two guards above don't cover: jobs still sitting in
            # `_waiting`, never picked up. That happens on a graceful
            # stop() with requests still queued (pre-existing), and now
            # ALSO on the CancelledError re-raise just above, which can
            # leave `_waiting` non-empty the moment it unwinds out of the
            # loop. Without this, `job.out.get()` hangs forever -- finding
            # C's exact failure mode (a queued job dropped with nothing
            # written to its queue), arriving via loop exit instead of via
            # `_form`.
            stranded, self._waiting = self._waiting, []
            for j in stranded:
                await j.out.put(("dispatch_error", RuntimeError(
                    "the dispatcher stopped before this request could run")))
                await j.out.put(None)

    async def _coalesce(self) -> None:
        """Wait, briefly and boundedly, for company before forming a batch.

        WITHOUT this the server never batches two concurrent clients, which is
        the whole feature. `submit()` sets `_arrived` on the FIRST job, so the
        dispatcher -- parked on that event -- wakes and forms while the second
        client's request is still being rendered and tokenized in its own route.
        Measured through the real server on Llama-3.2-1B, 2 clients, warm:
        `batch=None` on both requests, 3 runs of 3. It is not a race that
        sometimes goes the other way; the wake-up ordering is deterministic.
        The unit tests above missed it because they submit every job before the
        dispatcher's first iteration, which is not a shape HTTP can produce.

        It also closes the ANTI-PHASE LOCK, which is a different failure with
        the same cure: two steady clients alternate one-waiting-at-a-time (A
        waits while B runs, B finishes, A runs alone, B re-posts during A's
        run) so neither ever meets the other in `_waiting`. Whenever the
        partner's turnaround is shorter than this window they meet again. When
        it is longer -- a real agent that thinks or runs a tool for seconds --
        nothing here helps and nothing here pretends to: admitting a latecomer
        into a RUNNING batch is continuous batching, which this engine does not
        do.

        The wait is a CEILING, not a fixed delay. It ends the moment the batch
        is full, so an already-concurrent load pays nothing; a genuinely lone
        request at an idle server pays the whole window once, which is why the
        knob exists and why it is inert at MAX_BATCH_SIZE = 1.
        """
        if (self.batch_wait_s <= 0 or self.max_batch_size < 2
                or not self.batching_enabled):
            return
        if len(self._waiting) >= self.max_batch_size:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.batch_wait_s
        while not self._stopping and len(self._waiting) < self.max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            # Cleared before the wait, not after: an arrival that lands between
            # the check and the wait must still wake it. The dispatch loop
            # clears the same event for the same reason.
            self._arrived.clear()
            try:
                await asyncio.wait_for(self._arrived.wait(), remaining)
            except asyncio.TimeoutError:
                return

    async def _form(self) -> list:
        """Which waiting jobs run together. The head is always a member, so
        head-of-line always progresses and nothing starves.

        A job whose client already cancelled while queued is dropped here,
        before it can ever become a batch member -- it would waste a real
        batch slot (and GPU work) on a stream nobody is listening to any
        more. It still gets its sentinel: a route awaiting `job.out.get()`
        after its own disconnect watcher fires must not hang forever on a
        queue nobody will ever write to.
        """
        dropped = [j for j in self._waiting if j.cancel.cancelled]
        self._waiting = [j for j in self._waiting if not j.cancel.cancelled]
        for j in dropped:
            await j.out.put(None)
        if not self._waiting:
            return []
        if not self.batching_enabled or len(self._waiting) < 2:
            return self._waiting[:1]
        self._formation_attempts += 1
        cands = [Candidate(prompt_ids=j.prompt_ids, own_match=j.peek_len)
                 for j in self._waiting]
        width = select_width(cands, max_batch_size=self.max_batch_size,
                             chunk=self.prefill_chunk_size)
        if width < 2:
            self._note_if_never_admitting()
            return self._waiting[:1]
        admitted = self._admit(self._waiting[:width])
        if len(admitted) < 2:
            self.gate1_refused += 1
            self._note_if_never_admitting()
        return admitted

    def _note_if_never_admitting(self) -> None:
        """Say it ONCE when batching is configured and never forms a batch.

        This is Qwen3-235B today: its admission grants N=1, so an operator
        who set MAX_BATCH_SIZE > 1 is paying sdpa's 2-4 % for nothing and has
        no way to know. CLAUDE.md section 8 -- a feature that ships and
        quietly does nothing -- is exactly what this row exists to end.
        Latched: a flood of warnings would itself be the failure it reports.
        """
        if (self._never_admitted_warned or self.batches_run
                or self._formation_attempts < self.NEVER_ADMITTED_AFTER):
            return
        self._never_admitted_warned = True
        print(f"[Vaayu-Server] NOTE: batching has never admitted in "
              f"{self._formation_attempts} attempts with MAX_BATCH_SIZE="
              f"{self.max_batch_size}. Every request has been served alone, "
              f"so the sdpa kernel is being paid for (2-4 % end-to-end) with "
              f"no batching in return. Either this model's admission refuses "
              f"N>1 at this context, or warm prefix reuse is worth more than "
              f"batching for this workload -- both are legitimate, and both "
              f"mean MAX_BATCH_SIZE = 1 plus flash is the faster "
              f"configuration here.")

    def _admit(self, members: list) -> list:
        """KV pricing (plan_batch). Whatever it defers keeps its queue place.

        `giveback=True`: gate 1 must not be a STRICTER filter than gate 2.
        `batched_generate` (gate 2, vaayu_base.py:5302) always counts
        `expert_giveback_bytes` -- the expert VRAM cache's ENTITLEMENT above
        one decode step's working set, 2.88 GB on the MiniMax arm that
        measured P10's +1.606x -- at N>1, regardless of `reclaim`. Pricing
        gate 1 against free VRAM alone (B1, final whole-branch review) could
        collapse every batch to width 1 and make an operator conclude the
        model cannot batch, even though gate 2 would have admitted it. See
        `plan_batch`'s own docstring for why this is a distinct knob from
        `reclaim` (entitlement vs. currently-resident usage).

        On a pricing FAILURE (no CUDA -- a dense CPU-compute load, per CLAUDE.md
        section 5's model-agnostic mandate, has no VRAM to price at all -- or any
        other exception out of geo_from_model/free_vram_budget) this admits
        everyone rather than reducing to the head: plan_batch is this
        scheduler's OWN pre-check (gate 1), and `batched_generate` still runs
        its own admission (gate 2) inside run_batch regardless, with the
        refusal-to-serial fallback in `_run` below as the real safety
        net. Reducing to the head here would silently disable batching
        wholesale under exactly the condition (no VRAM introspection) where
        gate 2's own check still runs and can be trusted.
        """
        from ..batch import BatchRequest, plan_batch

        reqs = [BatchRequest(prompt_ids=j.prompt_ids, max_new_tokens=j.cap)
                for j in members]
        try:
            admitted, _deferred = plan_batch(self.engine.model, reqs, giveback=True)
        except Exception:                        # noqa: BLE001 -- never blocks
            return members
        # plan_batch sorts by length, so map back by the IDENTITY OF THE
        # REQUEST OBJECT (each freshly built above, one per member, via the
        # same `reqs`/`members` pairing) -- not `id(r.prompt_ids)`, which two
        # jobs sharing one prompt_ids list object would collide on.
        keep = {id(r) for r in admitted}
        chosen = [j for r, j in zip(reqs, members) if id(r) in keep]
        return chosen if len(chosen) >= 2 else members[:1]

    async def _run(self, members: list) -> None:
        if not members:
            return
        async with self.engine._lock:
            set_current_sessions([j.session_key for j in members])
            try:
                # THE RULE. A batch of one must NEVER reach batched_generate --
                # 3.24x regression on Qwen (573.88s vs 177.14s). This is the
                # one branch that enforces it.
                if len(members) == 1:
                    await self._run_serial(members[0])
                else:
                    try:
                        # run_batch (vaayu/server/engine.py) raises exactly
                        # when NOTHING reached a real job.out yet: a gate-2
                        # admission refusal inside batched_generate before it
                        # ever yielded a token, or thread.start() itself
                        # failing. Either way this retry is always safe -- a
                        # fallback to the serialized path, never a 500, and
                        # never a reason to delete gate 2 or lower the
                        # giveback floor (it was right once: MiniMax N=3
                        # genuinely OOM'd). A real cancellation racing with
                        # that raise is NOT read as a refusal here -- see the
                        # `sys.exc_info()` guard in run_batch's own finally.
                        await self.engine.run_batch(members)
                        self.batches_run += 1
                    except Exception as ex:      # noqa: BLE001 -- fallback
                        self.gate2_refused += 1
                        print(f"[Vaayu-Server] batch of {len(members)} refused "
                              f"({type(ex).__name__}: {ex}) -- running them "
                              f"serially.")
                        for j in members:
                            # B2 (final whole-branch review): `_current_sessions`
                            # still holds the WHOLE batch from the call above --
                            # narrowing straight to one member's session here
                            # reads as a second, overlapping generation to
                            # `logs.set_current_sessions`, which latches
                            # `_attribution_lost` and prints a false "real
                            # invariant violation" warning, and every marker
                            # read for the rest of this designed fallback comes
                            # back `('*', ())`. Clear first so each per-job
                            # call is seen as a fresh single-session marker,
                            # not an overlap.
                            clear_current_session()
                            set_current_sessions([j.session_key])
                            await self._run_serial(j)
            finally:
                clear_current_session()

    async def _run_serial(self, job) -> None:
        """Today's path, unchanged: engine.stream()/engine.generate(), row-
        writing finally and all. batched_generate is not imported on this
        branch.

        `prepared` is passed through so this never re-renders the chat
        template and re-tokenizes a prompt the route already tokenized.
        `stream` picks the same method the non-batched routes call today:
        `generate()` for stream=false (its own per-request speculative
        dispatch, and a `stream=false` requests.jsonl row) or `stream()`
        otherwise.

        A plain `Exception` is reported to the client and swallowed here, as
        before. A `BaseException` that is NOT an `Exception` (CancelledError,
        KeyboardInterrupt) is reported too, but then RE-RAISED once the
        client has been told -- mirroring `run_batch`'s own
        `pending_base_exc` pattern. Without this, a cancellation delivered
        while a serial job is running (every N=1 job, and every gate-2
        fallback retry) would be silently absorbed right here and never
        reach `_dispatch_loop`, which is what actually stops the dispatcher
        task on a real cancellation (see its `except BaseException` there).
        """
        if not job.stream:
            base_exc = None
            try:
                result = await self.engine.generate(job.req, cancel=job.cancel,
                                                     prepared=job.prepared)
            except Exception as ex:              # noqa: BLE001 -- reported, not re-raised
                await job.out.put(("error", ex))
            except BaseException as ex:          # noqa: BLE001 -- reported, then re-raised
                await job.out.put(("error", ex))
                base_exc = ex
            else:
                await job.out.put(("final", result))
            finally:
                await job.out.put(None)
            if base_exc is not None:
                raise base_exc
            return
        gen = self.engine.stream(job.req, cancel=job.cancel, prepared=job.prepared)
        base_exc = None
        try:
            async for item in gen:
                await job.out.put(item)
        except Exception as ex:                  # noqa: BLE001 -- reported, not re-raised
            await job.out.put(("error", ex))
        except BaseException as ex:              # noqa: BLE001 -- reported, then re-raised
            await job.out.put(("error", ex))
            base_exc = ex
        finally:
            await gen.aclose()
            await job.out.put(None)
        if base_exc is not None:
            raise base_exc
