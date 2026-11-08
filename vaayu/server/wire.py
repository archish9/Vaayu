"""`WireMeter` -- what the SOCKET did, for one request.

Every counter this server had instrumented the ENGINE; none instrumented the
socket. Run E (2026-11-08) is what that costs: a stream delivered ~10 chunks,
then nothing for 300 s with TCP alive, and every engine-side number on its
`requests.jsonl` row read healthy -- 629 tokens, `decode_tok_s 1.911`, and
`io_wait=0.0s` on every token summary in the session log. The one quantity
nobody logged was how long the client got nothing, so diagnosing it needed the
reporter's own bug report and a read of `_sse`'s source. Neither is in a
mailed `logs/` folder.

This object is where the socket's facts live. The ROUTE creates it -- the one
frame holding both the `Request` and the `chatcmpl-...` the client is given --
hangs it on the `Job`, and the dispatcher hands it to
`EngineHandle.stream()`/`generate()`, which pass it to the row writer. `_sse`,
which owns the bytes, records into it.

It lives in its own module rather than in `reqlog.py` because that module's
docstring states that nothing in it instruments the generation path, and that
is still true. This does, so it is not that file's business.

Two rules hold for every method here, both because this runs on the
generation path:

  * **Nothing raises.** `row()` is read by `_write_request_row`, which is
    called from a `finally`; an exception here would replace whatever real
    exception was already propagating. A half-filled meter -- a request that
    died before its first byte -- is the ordinary case, not an error.
  * **`meter=None` is legal everywhere.** `EngineHandle` is importable as a
    library with no server, no route and no socket, so every call site is
    guarded instead of requiring a meter.

The clock is injected so tests can drive it, and it is read **once per
event**: `tests/test_server_app.py::_TickingClock` advances one second per
reading, so a method reading it twice would invent a second of silence.
"""
import time


class WireMeter:
    """The socket's side of one request.

    `id` is the `chatcmpl-...` the client was given. It is what joins this
    request's row to its session-log lines, to its `server.log` traceback and
    to the id the reporter's own client logged -- rows otherwise key on
    `session` plus a second-resolution `ts`, which on a fan-out workload joins
    concurrent requests to each other.

    The `note_*` methods record what the socket did; the fields they feed are
    published by `row()` as the measurements that need them land. Only
    `engine_started_s` is published today.
    """

    def __init__(self, request_id: str, *, client: dict | None = None,
                 clock=time.monotonic) -> None:
        self.id = request_id
        # The coarse client shape the route derived (family/version/forwarded),
        # or None. Never the raw user-agent: a mailed folder must carry no
        # address, hostname or raw client string.
        self.client = client
        self.cancel_reason = None
        self._clock = clock
        # Request ARRIVAL. Every second `row()` reports is measured from here,
        # not from the engine's `started`: the queue wait is part of what the
        # client experienced, and the row's own `ttft_s`/`wall_s` already
        # begin after it. `engine_started_s` is the offset between the two.
        self._t0 = clock()
        self._engine_started = None
        # Has anything the SOCKET did been recorded? `_sse` is the only frame
        # that writes bytes, so "nothing recorded" and "no `_sse` frame
        # existed for this request" are the same condition -- which is what
        # makes `wire: null` the honest answer for a non-streaming request and
        # for a route-level refusal, with no second flag to keep in sync.
        #
        # `note_engine_started` deliberately does NOT set it: the engine's
        # start is a time ORIGIN for the fields below, not something the
        # socket did, and the non-streaming path records it on every request.
        # Anything that DOES set it makes `wire` non-null for that request.
        self._recorded = False
        # Recorded from here down, published by `row()` as the tasks that
        # measure them land. Counters, so a half-filled meter reads as zero
        # rather than as a missing attribute.
        self._chunks = 0
        self._comments = 0
        self._heartbeats = 0
        self._bytes = 0
        self._tokens = 0
        self._buffering_opened_at_token = None
        self._buffered_bytes = 0

    # ── what the socket did ─────────────────────────────────────────────────

    def note_chunk(self, nbytes: int) -> None:
        """One SSE data chunk went out. What a chunk-counting client sees."""
        self._chunks += 1
        self._bytes += nbytes
        self._recorded = True

    def note_comment(self, nbytes: int) -> None:
        """One `: ...` comment line went out -- a queue position or keepalive.

        Separate from a chunk because a client's timer does not count
        comments: in run E the keepalive comments were flowing and the client
        died anyway.
        """
        self._comments += 1
        self._bytes += nbytes
        self._recorded = True

    def note_heartbeat(self, nbytes: int) -> None:
        """One empty-content chunk from the buffered-tool-call heartbeat."""
        self._heartbeats += 1
        self._bytes += nbytes
        self._recorded = True

    def note_engine_started(self) -> None:
        """The engine took its own `started` just now.

        Records the time ORIGIN the row's `ttft_s`/`wall_s` are measured from,
        as an offset from request arrival. It does not by itself give this
        request a `wire` block -- see `_recorded` above.
        """
        self._engine_started = self._clock()

    def note_tokens(self, total: int) -> None:
        """The running TOTAL of tokens generated so far -- not a delta.

        A total because `_sse` cannot count tokens: it receives decoded text,
        and one streamer `put` can carry a whole accepted speculative run.
        The engine's loop, which has the ids, is what knows the count.
        """
        self._tokens = total
        self._recorded = True

    def note_buffering_open(self) -> None:
        """A tool-call open marker was seen: text stops reaching the wire."""
        self._buffering_opened_at_token = self._tokens
        self._recorded = True

    def note_buffered_bytes(self, nbytes: int) -> None:
        """Text held back in the tool-call buffer rather than sent."""
        self._buffered_bytes += nbytes
        self._recorded = True

    def note_cancel(self, reason: str) -> None:
        """Why this request ended early, in one word.

        A top-level row field rather than part of `wire`, so it reaches a
        non-streaming row too -- which is why this does not give the request a
        `wire` block of its own.
        """
        self.cancel_reason = reason

    # ── the row ─────────────────────────────────────────────────────────────

    def row(self) -> dict | None:
        """This request's `wire` object, or None if the socket did nothing.

        None means "there was no stream at all" -- a non-streaming request, or
        a refusal answered before `_sse` was entered. A zero-filled block
        would instead claim "the wire sent nothing", which is a different
        statement about a request that did have a socket frame.

        Never raises on a half-filled meter: every field is a counter that
        started at zero or an explicit None.
        """
        if not self._recorded:
            return None
        return {
            # Seconds from request ARRIVAL to the engine's own `started`. The
            # offset that makes the wire's seconds comparable with the row's
            # `ttft_s` and `wall_s`, which start only once this request was
            # admitted -- without it, a reader comparing the two on one row is
            # wrong by the queue wait. None if the engine never started.
            "engine_started_s": (None if self._engine_started is None
                                 else self._engine_started - self._t0),
        }
