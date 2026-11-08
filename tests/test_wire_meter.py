"""`WireMeter` -- the socket's side of one request.

The object, the id it carries, and the `None` row that keeps `wire` honest on
a request that never had a stream at all.
"""
from vaayu.server.wire import WireMeter


def test_a_meter_that_recorded_nothing_has_no_row():
    # Non-streaming requests and route-level refusals have no _sse frame;
    # `wire: null` is the honest answer, not a zero-filled block.
    assert WireMeter("chatcmpl-x").row() is None


def test_the_meter_carries_the_id_the_client_was_given():
    assert WireMeter("chatcmpl-abc").id == "chatcmpl-abc"


class _Clock:
    """Readings in order. Pops, so a method that reads twice per event runs
    off the end -- which is the property the injected clock exists for:
    `tests/test_server_app.py::_TickingClock` advances a second per reading,
    so a second read would invent a second of silence.
    """

    def __init__(self, *readings):
        self._readings = list(readings)

    def __call__(self):
        return self._readings.pop(0)


def test_the_engine_start_is_recorded_as_an_offset_from_request_arrival():
    """The wire's seconds start at request ARRIVAL; the row's `ttft_s` and
    `wall_s` start at the engine's own `started`, after admission. This offset
    is what makes the two comparable on one row."""
    # One reading per event: __init__, note_engine_started, note_chunk, row().
    m = WireMeter("chatcmpl-x", clock=_Clock(100.0, 104.5, 105.0, 105.0))
    m.note_engine_started()
    m.note_chunk(12)            # what `_sse` does; `row()` is None without it
    assert m.row()["engine_started_s"] == 4.5


def test_the_engine_start_alone_does_not_invent_a_wire_block():
    """Every non-streaming request records the engine start, and a
    non-streaming request has no socket activity to report. `wire` must stay
    null there: a block whose only content is an offset into nothing reads as
    a stream that sent no bytes."""
    m = WireMeter("chatcmpl-x")
    m.note_engine_started()
    assert m.row() is None


def test_a_meter_whose_request_never_reached_the_engine_still_rows():
    """`row()` is read from a `finally` on the generation path, so it must not
    raise on a half-filled meter -- a request refused or abandoned after its
    first byte but before the engine ran."""
    m = WireMeter("chatcmpl-x")
    m.note_comment(30)
    assert m.row()["engine_started_s"] is None


def test_the_meter_carries_the_client_shape_the_route_derived():
    """A coarse shape, derived in the route where the `Request` is -- never
    the raw user-agent, which must not reach any file."""
    assert WireMeter("chatcmpl-x", client={"family": "curl"}).client == {
        "family": "curl"}
    m = WireMeter("chatcmpl-x")
    assert m.client is None and m.cancel_reason is None


# ── chunks, bytes and the silence between them ──────────────────────────────


def test_a_comment_does_not_break_a_silence_span():
    """THE definition (spec section 5.1): a span runs between DATA CHUNKS.

    In run E the keepalive comments were flowing through the 322 s quiet and
    the client died anyway, because its timer counts chunks. A byte-silence
    meter would have reported ~15 s there -- `SSE_KEEPALIVE_S` guarantees a
    byte that often -- and called the run healthy, so the field could never
    report the number it exists to report.
    """
    # One reading per event: __init__, chunk, comment, comment, chunk, and one
    # per `row()` call below.
    m = WireMeter("chatcmpl-x", clock=_Clock(0, 0, 10, 20, 30, 30, 30, 30))
    m.note_chunk(40)                            # t=0
    m.note_comment(14)                          # t=10  -- must NOT reset
    m.note_comment(14)                          # t=20
    m.note_chunk(40)                            # t=30
    assert m.row()["max_silence_s"] == 30
    assert m.row()["comments_sent"] == 2 and m.row()["chunks_sent"] == 2


def test_row_closes_the_span_that_is_still_open():
    """Run E's row was written while the quiet was still in flight. Reporting
    only spans a chunk happened to CLOSE would have read `max_silence_s: 0`
    on the one request this whole object exists to describe."""
    m = WireMeter("chatcmpl-x", clock=_Clock(0, 0, 100))
    m.note_chunk(40)                            # t=0
    assert m.row()["max_silence_s"] == 100      # clock now at t=100


def test_a_heartbeat_is_a_chunk_and_is_counted_as_one():
    """The buffered-tool-call heartbeat is an empty-content CHUNK, so it both
    counts as one and ends the span -- which is the whole reason it is a chunk
    and not a comment."""
    m = WireMeter("chatcmpl-x", clock=_Clock(0, 1, 5, 5))
    m.note_chunk(40); m.note_heartbeat(60)
    r = m.row()
    assert r["heartbeats_sent"] == 1 and r["chunks_sent"] == 2
    # 1 s of wait for the first chunk, then 4 s the heartbeat closed -- not
    # the 5 s a wire that counted only `note_chunk` would have reported.
    assert r["max_silence_s"] == 4
    assert r["bytes_sent"] == 100


def test_bytes_and_the_first_and_last_markers():
    # One reading per event: __init__, comment, chunk, chunk, row().
    m = WireMeter("chatcmpl-x", clock=_Clock(0, 1, 2, 5, 5))
    m.note_comment(14)          # the queue-position line: a byte, not a chunk
    m.note_chunk(40)            # the role chunk
    m.note_chunk(60)
    r = m.row()
    assert r["bytes_sent"] == 114
    assert r["comments_sent"] == 1 and r["chunks_sent"] == 2
    assert r["first_byte_s"] == 1 and r["first_chunk_s"] == 2
    assert r["last_chunk_s"] == 5


def test_the_silence_is_dated_from_when_it_started():
    """`max_silence_at_s` is the START of the span, so a reader can find the
    moment in the session log -- and tell a cold prefill (quiet before
    `first_chunk_s`, the run-D class) from run E's (quiet after the last
    chunk, with tokens held)."""
    m = WireMeter("chatcmpl-x", clock=_Clock(0, 2, 40, 40))
    m.note_chunk(40)            # t=2: the role chunk, 2 s after arrival
    m.note_chunk(40)            # t=40: 38 s of quiet before it
    r = m.row()
    assert r["max_silence_s"] == 38 and r["max_silence_at_s"] == 2


def test_the_first_span_starts_at_request_arrival():
    """The wait for the first chunk is client-visible silence too: a chunk-
    counting client times out on a cold prefill exactly as it does on a
    buffered tool call. Measuring from the first chunk instead would report
    `0` for the run-D class."""
    m = WireMeter("chatcmpl-x", clock=_Clock(0, 26, 26))
    m.note_chunk(40)            # the role chunk, 26 s after the request landed
    r = m.row()
    assert r["max_silence_s"] == 26 and r["max_silence_at_s"] == 0


def test_a_wire_that_never_sent_a_chunk_reports_no_silence_rather_than_zero():
    """Zero is a measurement -- "silent for nothing". A request whose socket
    never carried a data chunk has no span to report, and `0` there would
    read as a healthy wire on exactly the rows most likely to be broken."""
    m = WireMeter("chatcmpl-x", clock=_Clock(0, 4))
    m.note_comment(30)
    r = m.row()
    assert r["max_silence_s"] is None and r["max_silence_at_s"] is None
    assert r["chunks_sent"] == 0 and r["first_chunk_s"] is None
    assert r["last_chunk_s"] is None and r["first_byte_s"] == 4


def test_the_wire_block_has_every_key_the_row_promises():
    """Every row keeps the same keys, null where unknowable, so a reader never
    has to guess whether a field is missing or zero.
    `tokens_held_at_max_silence` and `buffering` are published here and filled
    by the task that measures the buffer."""
    m = WireMeter("chatcmpl-x")
    m.note_chunk(1)
    assert set(m.row()) == {
        "chunks_sent", "comments_sent", "bytes_sent", "first_byte_s",
        "first_chunk_s", "last_chunk_s", "max_silence_s", "max_silence_at_s",
        "heartbeats_sent", "engine_started_s", "tokens_held_at_max_silence",
        "buffering"}
