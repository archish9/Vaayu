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


class _Shadow:
    """A Job look-alike run_batch can write to that is NOT a real client's
    queue -- see `BatchScheduler._run_batched` for why this exists.

    Duck-types exactly what `run_batch` reads (`.req`, `.prompt_ids`, `.cap`,
    `.cancel`, `.out`, `.session_key`); everything but `.out` is the real
    job's own object, so run_batch's per-request kwargs, prompt, cap, cancel
    token and logging all match the real request precisely.
    """

    __slots__ = ("req", "prompt_ids", "cap", "cancel", "session_key", "out")

    def __init__(self, job):
        self.req = job.req
        self.prompt_ids = job.prompt_ids
        self.cap = job.cap
        self.cancel = job.cancel
        self.session_key = job.session_key
        self.out: asyncio.Queue = asyncio.Queue()


class BatchScheduler:
    def __init__(self, engine, queue_depth: int = 8, max_batch_size: int = 1,
                 prefill_chunk_size: int = 512):
        self.engine = engine
        self.queue_depth = queue_depth
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
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
        # a feature that ships and quietly does nothing.
        self.formations_refused = 0
        self.batches_run = 0

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
            await self._task
            self._task = None

    async def submit(self, job: Job) -> None:
        """Enqueue `job`. Raises QueueFull past `queue_depth`.

        The count includes the running batch's members, so a client's queue
        position is `depth` at entry -- the same promise slot() made.
        """
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
        while not self._stopping:
            if not self._waiting:
                await self._arrived.wait()
                self._arrived.clear()
                continue
            members = self._form()
            for j in members:
                self._waiting.remove(j)
            self._running = members
            try:
                await self._run(members)
            finally:
                self._running = []

    def _form(self) -> list:
        """Which waiting jobs run together. The head is always a member, so
        head-of-line always progresses and nothing starves."""
        if not self.batching_enabled or len(self._waiting) < 2:
            return self._waiting[:1]
        cands = [Candidate(prompt_ids=j.prompt_ids, own_match=j.peek_len)
                 for j in self._waiting]
        width = select_width(cands, max_batch_size=self.max_batch_size,
                             chunk=self.prefill_chunk_size)
        if width < 2:
            return self._waiting[:1]
        return self._admit(self._waiting[:width])

    def _admit(self, members: list) -> list:
        """KV pricing (plan_batch). Whatever it defers keeps its queue place.

        On a pricing FAILURE (no CUDA -- a dense CPU-compute load, per CLAUDE.md
        section 5's model-agnostic mandate, has no VRAM to price at all -- or any
        other exception out of geo_from_model/free_vram_budget) this admits
        everyone rather than reducing to the head: plan_batch is this
        scheduler's OWN pre-check (gate 1), and `batched_generate` still runs
        its own admission (gate 2) inside run_batch regardless, with the
        refusal-to-serial fallback in `_run_batched` below as the real safety
        net. Reducing to the head here would silently disable batching
        wholesale under exactly the condition (no VRAM introspection) where
        gate 2's own check still runs and can be trusted.
        """
        from ..batch import BatchRequest, plan_batch

        reqs = [BatchRequest(prompt_ids=j.prompt_ids, max_new_tokens=j.cap)
                for j in members]
        try:
            admitted, _deferred = plan_batch(self.engine.model, reqs)
        except Exception:                        # noqa: BLE001 -- never blocks
            return members
        # plan_batch sorts by length, so map back by identity of the prompt
        # list rather than by position.
        keep = {id(r.prompt_ids) for r in admitted}
        chosen = [j for j in members if id(j.prompt_ids) in keep]
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
                    ran, reason = True, None
                    try:
                        ran = await self._run_batched(members)
                    except Exception as ex:      # noqa: BLE001 -- fallback
                        # run_batch's own docstring names its one gap:
                        # thread.start() failing leaves no sentinel on ANY
                        # queue at all (not even the shadow ones _run_batched
                        # uses) -- always safe to retry.
                        ran, reason = False, f"{type(ex).__name__}: {ex}"
                    if ran:
                        self.batches_run += 1
                    else:
                        # Gate 2 re-checks admission at the real width inside
                        # batched_generate and may still refuse -- a fallback
                        # to the serialized path, never a 500, and never a
                        # reason to delete gate 2 or lower the giveback floor
                        # (it was right once: MiniMax N=3 genuinely OOM'd).
                        self.formations_refused += 1
                        print(f"[Vaayu-Server] batch of {len(members)} refused"
                              f"{f' ({reason})' if reason else ''} -- running "
                              f"them serially.")
                        for j in members:
                            set_current_sessions([j.session_key])
                            await self._run_serial(j)
            finally:
                clear_current_session()

    async def _run_batched(self, members: list) -> bool:
        """Run `members` through run_batch, forwarding every item to each
        real job.out live as it arrives. Returns True if the batch actually
        ran, False if it was a whole-batch refusal that must be retried
        serially -- in which case NOTHING has reached a real job.out, so the
        retry cannot double-deliver.

        WHY SHADOW QUEUES. `batched_generate` can refuse the whole batch
        (gate 2) after `_admit`'s own plan_batch pricing (gate 1) already
        passed it -- the exact scenario `test_a_gate_two_refusal_falls_back_
        to_serial` pins. But `run_batch` answers that refusal by writing
        ("error", exc) + None straight into EVERY job's queue itself and
        returning normally -- it does not raise to this caller (only
        thread.start() failing does that, per its own docstring). So `_run`'s
        `except Exception` alone cannot see a gate-2 refusal: by the time
        run_batch returns, the "mistake" is already in the client's hands.
        Passing shadow queues instead of the real ones lets this method peek
        each job's FIRST item before anything reaches a real client, and hold
        the whole batch back -- retried serially -- only when EVERY member's
        first item is an error (the shared, whole-batch refusal). Any other
        shape (even one real delta anywhere) forwards live: a per-job failure
        that happens after real generation started is not a formation
        refusal, and re-running it would risk sending a client output twice.
        """
        shadows = [_Shadow(j) for j in members]
        firsts = [None] * len(members)
        error_holder: list = []

        async def _peek(i):
            firsts[i] = await shadows[i].out.get()

        def _on_task_done(t):
            # Cancel any peek still waiting on a shadow queue that
            # thread.start() failing means will never receive anything.
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                error_holder.append(exc)
                for p in peek_tasks:
                    if not p.done():
                        p.cancel()

        task = asyncio.ensure_future(self.engine.run_batch(shadows))
        task.add_done_callback(_on_task_done)
        peek_tasks = [asyncio.ensure_future(_peek(i)) for i in range(len(members))]
        await asyncio.gather(*peek_tasks, return_exceptions=True)
        if error_holder:
            raise error_holder[0]

        if all(f is not None and f[0] == "error" for f in firsts):
            # Whole-batch, before-any-token refusal: nothing was forwarded to
            # a real job.out, so the caller can retry serially untouched.
            await task
            return False

        async def _forward(i):
            item = firsts[i]
            while True:
                await members[i].out.put(item)
                if item is None:
                    break
                item = await shadows[i].out.get()

        await asyncio.gather(task, *(_forward(i) for i in range(len(members))))
        return True

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
        """
        if not job.stream:
            try:
                result = await self.engine.generate(job.req, cancel=job.cancel)
            except BaseException as ex:          # noqa: BLE001 -- reported
                await job.out.put(("error", ex))
            else:
                await job.out.put(("final", result))
            finally:
                await job.out.put(None)
            return
        gen = self.engine.stream(job.req, cancel=job.cancel, prepared=job.prepared)
        try:
            async for item in gen:
                await job.out.put(item)
        except BaseException as ex:              # noqa: BLE001 -- reported
            await job.out.put(("error", ex))
        finally:
            await gen.aclose()
            await job.out.put(None)
