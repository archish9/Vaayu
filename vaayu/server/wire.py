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
import re
import time


class WireMeter:
    """The socket's side of one request.

    `id` is the `chatcmpl-...` the client was given. It is what joins this
    request's row to its session-log lines, to its `server.log` traceback and
    to the id the reporter's own client logged -- rows otherwise key on
    `session` plus a second-resolution `ts`, which on a fan-out workload joins
    concurrent requests to each other.

    The `note_*` methods record what the socket did; the fields they feed are
    published by `row()` as the measurements that need them land.
    `tokens_held_at_max_silence` and `buffering` are None until a chunk or a
    tool-call open marker is seen -- every row keeps the same keys.
    """

    def __init__(self, request_id: str, *, client: dict | None = None,
                 clock=time.monotonic) -> None:
        self.id = request_id
        # The coarse client shape the route derived (family/version/forwarded),
        # or None. Never the raw user-agent: a mailed folder must carry no
        # address, hostname or raw client string.
        self.client = client
        self.cancel_reason = None
        # Seconds from arrival to the FIRST note_cancel, and how long the wire
        # had been quiet at that moment -- both frozen there, not at row().
        self.cancel_detected_at_s = None
        self.s_since_last_chunk = None
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
        self._first_byte = None
        self._first_chunk = None
        self._last_chunk = None
        # The OPEN silence span's start. Request arrival, not the first chunk:
        # the wait for the first chunk is client-visible silence too -- a
        # chunk-counting client times out on a cold prefill exactly as it does
        # on a buffered tool call (the run-D class).
        self._silence_from = self._t0
        # None until a data chunk has closed a span. `0` would be a
        # measurement ("silent for nothing"), and a wire that never carried a
        # chunk has nothing to report.
        self._max_silence = None
        self._max_silence_at = None
        self._tokens = 0
        # The running token total when the OPEN span started, and the
        # (opened, closed) totals of the longest span so far: the tokens that
        # accumulated while the wire was quiet are closed - opened, which is
        # not the total produced and not the total at row time.
        self._silence_from_tokens = 0
        self._max_silence_tokens = None
        self._buffering_opened_at_token = None
        self._buffered_bytes = 0

    # ── what the socket did ─────────────────────────────────────────────────

    def note_chunk(self, nbytes: int) -> None:
        """One SSE data chunk went out. What a chunk-counting client sees."""
        self._data_chunk(nbytes)

    def note_comment(self, nbytes: int) -> None:
        """One `: ...` comment line went out -- a queue position or keepalive.

        Separate from a chunk because a client's timer does not count
        comments: in run E the keepalive comments were flowing and the client
        died anyway. So this moves `bytes_sent`, `comments_sent` and
        `first_byte_s`, and deliberately does NOT end the silence span -- the
        single most consequential line in this object. A byte-silence meter
        would have reported ~15 s for run E (`SSE_KEEPALIVE_S` guarantees a
        byte that often) and called it healthy.
        """
        now = self._clock()
        self._comments += 1
        self._bytes += nbytes
        self._recorded = True
        if self._first_byte is None:
            self._first_byte = now

    def note_heartbeat(self, nbytes: int) -> None:
        """One empty-content chunk from the buffered-tool-call heartbeat.

        A data CHUNK, which is why it is one on the wire too (`656e3e2`): it
        counts in `chunks_sent` and it ends the silence span, because a
        chunk-counting client's timer is what it exists to refresh.
        """
        self._heartbeats += 1
        self._data_chunk(nbytes)

    def _data_chunk(self, nbytes: int) -> None:
        """A data chunk left the frame: the event that ENDS a silence span.

        Reads the clock once (see the module docstring) and shares that one
        reading between the first/last markers and the span it closes.
        """
        now = self._clock()
        self._chunks += 1
        self._bytes += nbytes
        self._recorded = True
        if self._first_byte is None:
            self._first_byte = now
        if self._first_chunk is None:
            self._first_chunk = now
        self._last_chunk = now
        span = now - self._silence_from
        if self._max_silence is None or span > self._max_silence:
            self._max_silence = span
            self._max_silence_at = self._silence_from - self._t0
            self._max_silence_tokens = (self._silence_from_tokens, self._tokens)
        self._silence_from = now
        self._silence_from_tokens = self._tokens

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

        First write wins: the first reason observed is the true one, and the
        teardown a disconnect causes must not relabel it `stream_closed`. The
        quiet span is measured NOW -- "it timed out" and "someone pressed
        stop" differ by how long the wire had been silent when the cancel
        landed, and `row()` time is later.
        """
        if self.cancel_reason is not None:
            return
        now = self._clock()
        self.cancel_reason = reason
        self.cancel_detected_at_s = now - self._t0
        # `_silence_from` is the last data chunk, or arrival if none went out.
        self.s_since_last_chunk = now - self._silence_from

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
        max_silence, max_silence_at = self._max_silence, self._max_silence_at
        silence_tokens = self._max_silence_tokens
        if self._chunks:
            # Close the span that is still OPEN. The row is written from the
            # engine's `finally`, so run E's own row was written while its
            # 322 s of quiet was still in flight: reporting only the spans a
            # chunk happened to CLOSE would have read `max_silence_s: 0` on
            # the one request this object exists to describe.
            open_span = self._clock() - self._silence_from
            if open_span > max_silence:
                max_silence = open_span
                max_silence_at = self._silence_from - self._t0
                silence_tokens = (self._silence_from_tokens, self._tokens)
        opened = self._buffering_opened_at_token
        return {
            # As of ROW time, not for the whole response: `_sse` emits the
            # terminating chunk and `data: [DONE]` after the engine's
            # row-writing `finally` has run. Those are milliseconds; the
            # silence this measures happens during generation.
            "chunks_sent": self._chunks,
            "comments_sent": self._comments,
            "heartbeats_sent": self._heartbeats,
            "bytes_sent": self._bytes,
            # Seconds from request ARRIVAL. `first_byte_s` counts a comment
            # (the queue-position line is usually the first byte out);
            # `first_chunk_s` is the first thing a chunk-counting client sees.
            "first_byte_s": (None if self._first_byte is None
                             else self._first_byte - self._t0),
            "first_chunk_s": (None if self._first_chunk is None
                              else self._first_chunk - self._t0),
            "last_chunk_s": (None if self._last_chunk is None
                             else self._last_chunk - self._t0),
            # The headline number: the longest span with no data CHUNK, and
            # when that span started. None -- never `0` -- when no chunk was
            # ever written: there is no span to report, and a zero there would
            # read as a healthy wire on the rows most likely to be broken.
            "max_silence_s": max_silence,
            "max_silence_at_s": max_silence_at,
            # Tokens generated while the longest quiet span was open (run E:
            # 629 - 9 = 620). None when no chunk was ever written, like
            # `max_silence_s`; 0 when that span was not a tool call.
            "tokens_held_at_max_silence": (
                None if silence_tokens is None
                else silence_tokens[1] - silence_tokens[0]),
            # None unless a tool-call open marker was seen.
            "buffering": (None if opened is None else {
                "opened_at_token": opened,
                "tokens_buffered": self._tokens - opened,
                "bytes_buffered": self._buffered_bytes,
            }),
            # Seconds from request ARRIVAL to the engine's own `started`. The
            # offset that makes the wire's seconds comparable with the row's
            # `ttft_s` and `wall_s`, which start only once this request was
            # admitted -- without it, a reader comparing the two on one row is
            # wrong by the queue wait. None if the engine never started.
            "engine_started_s": (None if self._engine_started is None
                                 else self._engine_started - self._t0),
        }


def cancel_fields(meter) -> dict:
    """The three top-level cancel keys of a row; null for a missing meter or a
    request nothing cancelled. One place so both row writers keep one key set."""
    if meter is None:
        return {"cancel_reason": None, "cancel_detected_at_s": None,
                "s_since_last_chunk": None}
    return {"cancel_reason": meter.cancel_reason,
            "cancel_detected_at_s": meter.cancel_detected_at_s,
            "s_since_last_chunk": meter.s_since_last_chunk}


# Most specific first; the first probe that matches wins. The optional group
# is the major version, taken from the digits right after the product name.
_VER = r"(?:[/ ]v?(\d{1,4})(?!\d))?"
_CLIENT_PROBES = [(re.compile(p, re.I), fam) for p, fam in (
    (r"\bopenai/python" + _VER, "openai-python"),
    (r"\bopenai/js" + _VER, "openai-js"),
    (r"\banthropic(?:[/-][A-Za-z]+)*" + _VER, "anthropic-sdk"),
    (r"\bcurl" + _VER, "curl"),
    (r"\bpython-httpx" + _VER, "httpx"),
    (r"\bpython-requests" + _VER, "python-requests"),
    (r"\baiohttp" + _VER, "aiohttp"),
    (r"\bnode-fetch" + _VER, "node-fetch"),
    (r"\bundici" + _VER, "undici"),
    (r"\baxios" + _VER, "axios"),
    (r"\bgo-http-client" + _VER, "go-http-client"),
    (r"^mozilla/", "browser"),
)]
# Headers whose PRESENCE says a proxy hop sat in front. Their values are
# addresses and are never read.
FORWARDING_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip", "via")


def client_shape(user_agent, *, forwarded: bool) -> dict:
    """`{family, version, forwarded}` -- the coarse client, never the raw string.

    A user-agent can carry a machine name or a build, and a mailed `logs/`
    folder must not. Only a family from the fixed list above, a major version
    number and whether a forwarding header was present are returned; the input
    is discarded. Anything unrecognised, empty, non-string or enormous is
    `unknown` -- a normal value, never an error (this runs on the logging path).
    """
    family, version = "unknown", None
    try:
        if isinstance(user_agent, str):
            ua = user_agent[:256]
            for rx, fam in _CLIENT_PROBES:
                m = rx.search(ua)
                if m:
                    family = fam
                    version = m.group(1) if m.groups() else None
                    break
    except Exception:                                   # noqa: BLE001
        family, version = "unknown", None
    return {"family": family, "version": version, "forwarded": bool(forwarded)}


def client_field(meter):
    """The row's `client` key: the meter's derived shape, null if none."""
    return None if meter is None else meter.client
