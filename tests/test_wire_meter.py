"""`WireMeter` -- the socket's side of one request.

The object, the id it carries, and the `None` row that keeps `wire` honest on
a request that never had a stream at all.
"""
import json

import pytest

from vaayu.server.wire import WireMeter, client_shape


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
    `tokens_held_at_max_silence` and `buffering` are None until a chunk or a
    tool-call open marker is seen."""
    m = WireMeter("chatcmpl-x")
    m.note_chunk(1)
    assert set(m.row()) == {
        "chunks_sent", "comments_sent", "bytes_sent", "first_byte_s",
        "first_chunk_s", "last_chunk_s", "max_silence_s", "max_silence_at_s",
        "heartbeats_sent", "engine_started_s", "tokens_held_at_max_silence",
        "buffering"}


def _ticks(*readings):
    return _Clock(*readings)


def test_tokens_held_is_measured_when_the_span_opens_not_when_it_closes():
    # Readings: __init__, note_chunk, row().
    m = WireMeter("chatcmpl-x", clock=_ticks(0.0, 1.0, 100.0))
    m.note_tokens(9)
    m.note_buffering_open()
    m.note_chunk(40)                       # closes the 1 s startup span (9 tokens)
    m.note_tokens(629)                     # the body generates while the wire is quiet
    r = m.row()                            # the open 99 s span wins
    assert r["max_silence_s"] == 99.0
    assert r["buffering"]["opened_at_token"] == 9
    assert r["tokens_held_at_max_silence"] == 620      # 629 - 9, held, not produced
    assert r["buffering"]["tokens_buffered"] == 620


def test_the_tokens_belong_to_the_winning_span_not_the_latest_one():
    # Span 1 (t 0->50) holds 30 tokens and wins; span 2 (50->51) holds 5.
    m = WireMeter("chatcmpl-x", clock=_ticks(0.0, 50.0, 51.0, 51.5))
    m.note_tokens(30)
    m.note_chunk(1)
    m.note_tokens(35)
    m.note_chunk(1)
    m.note_tokens(36)
    assert m.row()["tokens_held_at_max_silence"] == 30


def test_tokens_held_is_zero_when_the_quiet_was_not_a_tool_call():
    # The longest span (0->10) had no tokens in it; the 5 arrived after.
    m = WireMeter("chatcmpl-x", clock=_ticks(0.0, 10.0, 11.0))
    m.note_chunk(40)
    m.note_tokens(5)
    r = m.row()
    assert r["max_silence_s"] == 10.0
    assert r["tokens_held_at_max_silence"] == 0 and r["buffering"] is None


def test_buffered_bytes_accumulate_and_a_chunkless_wire_holds_nothing():
    m = WireMeter("chatcmpl-x")
    m.note_tokens(3)
    m.note_buffering_open()
    m.note_buffered_bytes(10)
    m.note_buffered_bytes(5)
    r = m.row()
    assert r["buffering"] == {"opened_at_token": 3, "tokens_buffered": 0,
                              "bytes_buffered": 15}
    assert r["tokens_held_at_max_silence"] is None


def test_note_cancel_is_first_write_wins_and_measures_at_cancel_time():
    t = [0.0]
    m = WireMeter("chatcmpl-x", clock=lambda: t[0])
    t[0] = 2.0
    m.note_chunk(1)
    t[0] = 9.0
    m.note_cancel("client_disconnected")
    t[0] = 50.0
    m.note_cancel("stream_closed")
    assert m.cancel_reason == "client_disconnected"
    assert m.cancel_detected_at_s == 9.0
    # 7 s since the chunk AT cancel time -- not 48, which is what reading the
    # clock at row() time (or taking the later cancel) would give.
    assert m.s_since_last_chunk == 7.0


def test_note_cancel_alone_keeps_the_wire_block_null():
    m = WireMeter("chatcmpl-x")
    m.note_cancel("client_disconnected")
    assert m.row() is None


def test_s_since_last_chunk_falls_back_to_time_since_arrival_with_no_chunk():
    # Unlike `max_silence_s` (null on a wire that never sent a chunk), this is
    # non-null there: the quiet span began at arrival.
    t = [0.0]
    m = WireMeter("chatcmpl-x", clock=lambda: t[0])
    t[0] = 4.0
    m.note_cancel("abandoned_queued")
    assert m.s_since_last_chunk == 4.0


@pytest.mark.parametrize("ua,family,version", [
    ("OpenAI/Python 1.54.3", "openai-python", "1"),
    ("curl/8.5.0", "curl", "8"),
    ("python-httpx/0.27.2", "httpx", "0"),
    ("node-fetch/1.0 (+https://github.com/bitinn/node-fetch)", "node-fetch", "1"),
    ("Mozilla/5.0 (X11; Linux x86_64)", "browser", None),
    ("", "unknown", None),
    (None, "unknown", None),
    ("x" * 100000, "unknown", None),
    (b"curl/8", "unknown", None),
    ("OpenAI/JS 4.67.1", "openai-js", "4"),
    ("anthropic-sdk-python/0.1", "anthropic-sdk", "0"),
    ("Anthropic/Python 0.40.0", "anthropic-sdk", "0"),
    ("python-requests/2.31.0", "python-requests", "2"),
    ("Python/3.12 aiohttp/3.9.5", "aiohttp", "3"),
    ("undici", "undici", None),
    ("axios/1.7.2", "axios", "1"),
    ("Go-http-client/2.0", "go-http-client", "2"),
    ("curl/999999999999999999999999", "curl", None),
])
def test_the_client_family_is_derived(ua, family, version):
    assert client_shape(ua, forwarded=False) == {
        "family": family, "version": version, "forwarded": False}


@pytest.mark.parametrize("ua", [
    "curl/8.5.0 host=alice-laptop.local",
    "my-secret-agent/3.1 alice-laptop.local build 9f3c",
    "Mozilla/5.0 (X11; Linux x86_64) alice-laptop.local",
])
def test_the_raw_user_agent_is_never_in_the_shape(ua):
    # Only the private tail is checked, so a family that happened to contain
    # the UA's leading text cannot satisfy this; echoing the string back can't.
    blob = json.dumps(client_shape(ua, forwarded=False))
    assert "alice-laptop" not in blob and "9f3c" not in blob and "secret" not in blob


def test_a_forwarded_request_records_the_fact_and_no_address():
    shape = client_shape("curl/8.5.0", forwarded=True)
    assert shape["forwarded"] is True
    assert set(shape) == {"family", "version", "forwarded"}
