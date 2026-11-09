"""uvicorn's access line must not carry the peer address into `server.log`.

`documentation/server.md` tells a user to zip `logs/run-<timestamp>/` and mail
it, which is only askable if the folder carries nothing personal about them.
uvicorn logs `'%s - "%s %s HTTP/%s" %d'` with `get_client_addr(scope)` as the
first argument -- `"host:port"` -- and `CAPTURE_STDIO` tees fd 1/2 into
`server.log`, so over Tailscale that is the reporter's stable `100.x` tailnet
address. Found by the WIRE-TELEMETRY Task 6 review; `todo.md`'s
`LOG-FOLDER-PEER-IP` row records the three options and the user picked option 2:
keep the access log, scrub the address.

These tests pin the scrub, not the format: the arrival record, the path and the
status are the signal and must survive.
"""
import logging
import logging.config

import pytest

from vaayu.server.logs import ScrubPeerAddr, uvicorn_log_config

# Exactly what uvicorn's h11 and httptools implementations both emit
# (`protocols/http/h11_impl.py`'s `access_logger.info` call).
_MSG = '%s - "%s %s HTTP/%s" %d'


def _access_record(client_addr, msg=_MSG):
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(client_addr, "POST", "/v1/chat/completions", "1.1", 200),
        exc_info=None)


@pytest.mark.parametrize("addr,token", [
    # The case that matters: a tailnet address is a stable identifier for the
    # reporter's machine.
    ("100.94.62.117:54321", "remote"),
    ("192.168.1.42:1234", "remote"),
    ("203.0.113.9:443", "remote"),
    ("127.0.0.1:54321", "local"),
    ("::1:54321", "local"),
    ("", "-"),
])
def test_the_peer_address_is_replaced_by_a_coarse_token(addr, token):
    rec = _access_record(addr)
    assert ScrubPeerAddr().filter(rec) is True
    line = rec.getMessage()
    assert line.startswith(f"{token} - ")
    # No octet of the original address survives anywhere in the line.
    for part in addr.replace(":", ".").split("."):
        if part.isdigit() and len(part) > 1:
            assert part not in line, f"{part!r} from {addr!r} leaked into {line!r}"


def test_the_request_line_and_status_survive_the_scrub():
    # Option 1 (access_log=False) was rejected because the arrival record is
    # the only trace of a request that never reached a route. Scrubbing must
    # not cost that.
    rec = _access_record("100.94.62.117:54321")
    ScrubPeerAddr().filter(rec)
    line = rec.getMessage()
    assert '"POST /v1/chat/completions HTTP/1.1" 200' in line


def test_local_and_remote_stay_distinguishable():
    # The one bit of the address worth keeping: a request through the Tailscale
    # front-end arrives on loopback, a direct tailnet bind does not.
    local, remote = _access_record("127.0.0.1:1"), _access_record("100.64.0.1:1")
    ScrubPeerAddr().filter(local)
    ScrubPeerAddr().filter(remote)
    assert local.getMessage().split(" - ")[0] != remote.getMessage().split(" - ")[0]


@pytest.mark.parametrize("args", [
    None,
    (),
    ("only-one-arg",),
    # logging's dict-style formatting: the mapping must be wrapped in a tuple,
    # because a bare dict is not a constructable `LogRecord.args` (3.12's
    # `LogRecord.__init__` does `args[0]` on it and raises `KeyError: 0`).
    ({"mapping": "style"},),
    ("a", "b", "c", "d", "e", "f"),
])
def test_a_record_that_is_not_an_access_line_is_passed_through_untouched(args):
    # uvicorn.access carries only the one message shape today, but a logging
    # filter that mangles anything else would corrupt a future line.
    rec = logging.LogRecord(name="uvicorn.access", level=logging.INFO,
                            pathname=__file__, lineno=1, msg="plain %s",
                            args=args, exc_info=None)
    before = rec.args
    assert ScrubPeerAddr().filter(rec) is True
    assert rec.args is before


def test_the_filter_never_raises_and_never_drops_a_record():
    # This runs on every served request. A logging error must never become the
    # request's problem, and dropping the record would lose the arrival trace.
    class Hostile:
        def __str__(self):
            raise RuntimeError("boom")

    rec = _access_record(Hostile())
    assert ScrubPeerAddr().filter(rec) is True


def test_the_log_config_wires_the_filter_onto_uvicorn_access():
    cfg = uvicorn_log_config()
    assert "scrub_peer_addr" in cfg["filters"]
    assert cfg["loggers"]["uvicorn.access"]["filters"] == ["scrub_peer_addr"]
    # uvicorn's own handlers/formatters must be left as they are -- we are
    # adding a filter, not taking over its logging.
    assert set(cfg["handlers"]) == {"default", "access"}
    assert cfg["loggers"]["uvicorn.access"]["handlers"] == ["access"]


def test_the_configured_dotted_path_actually_resolves():
    # A typo in the `()` path is invisible until a real server boots, and then
    # it is a dictConfig error at startup rather than a scrubbed line.
    path = uvicorn_log_config()["filters"]["scrub_peer_addr"]["()"]
    resolved = logging.config.DictConfigurator({"version": 1}).resolve(path)
    assert resolved is ScrubPeerAddr


_CAPTURED = []


class Capture(logging.Handler):
    """Module level, not a closure: `dictConfig` resolves handlers by dotted
    path and cannot import a class defined inside a function."""

    def emit(self, record):
        _CAPTURED.append(record.getMessage())


def test_the_scrub_holds_end_to_end_through_dictconfig():
    """The filter, the config and the real logging machinery together.

    The unit tests above call `.filter()` directly; this one proves the wiring,
    which is where a `filters:` key on the wrong logger would hide.
    """
    _CAPTURED.clear()
    captured = _CAPTURED
    cfg = uvicorn_log_config()
    cfg["handlers"]["access"] = {"class": f"{Capture.__module__}.{Capture.__qualname__}"}
    # Keep the test's own loggers alive while this config is installed.
    cfg["disable_existing_loggers"] = False
    try:
        logging.config.dictConfig(cfg)
        logging.getLogger("uvicorn.access").info(
            _MSG, "100.94.62.117:54321", "POST", "/v1/chat/completions", "1.1", 200)
    finally:
        import uvicorn.config
        logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)

    assert captured == ['remote - "POST /v1/chat/completions HTTP/1.1" 200']


def test_the_server_entrypoint_passes_the_scrubbing_log_config(monkeypatch):
    """CLAUDE.md §8: a filter uvicorn is never told about is decoration.

    This is the only thing standing between the scrub and "banked, not
    shipped" -- `uvicorn.run` builds its own logging from
    `uvicorn.config.LOGGING_CONFIG` unless handed a config, and it does so
    inside `run()`, after any filter added by hand would have been installed.
    """
    import uvicorn

    from vaayu.server import __main__ as main_mod

    captured = {}
    monkeypatch.setattr(main_mod.config, "MODEL_PATH", "unused-nonempty")
    monkeypatch.setattr(main_mod, "create_app", lambda cfg: object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(kw))

    assert main_mod.main() == 0
    assert captured["log_config"]["loggers"]["uvicorn.access"]["filters"] \
        == ["scrub_peer_addr"]
    # The access log stays ON -- scrubbing it was chosen over silencing it.
    assert captured["log_level"] == "info"
    assert captured.get("access_log") is not False
