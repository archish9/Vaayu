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
    m = WireMeter("chatcmpl-x", clock=_Clock(100.0, 104.5))
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
