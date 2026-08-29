"""The docs' port must be the port the code ships (B13).

Every curl example in the server docs is copy-pasted by a user against a default
install, so a stale port there is a doc that cannot work. The expected value is
READ FROM `config.py` rather than written here -- a literal in this file would
just be a fourth place to forget.
"""
import re
from pathlib import Path

from vaayu.server import config
from vaayu.server.app import ServerConfig

ROOT = Path(__file__).resolve().parent.parent
DOCS = ("README.md", "documentation/server.md", "vaayu/server/README.md")

_URL_PORT = re.compile(r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d+)")
_TABLE_PORT = re.compile(r"\|\s*`PORT`\s*\|\s*`([^`]+)`")


def test_the_dataclass_default_is_the_shipped_port():
    # __main__ always passes config.PORT, so this default is dead for the server
    # -- but it is live for every library caller and for the tests, and it was
    # the third disagreeing answer to "what port is this?".
    assert ServerConfig(model_path="/x").port == config.PORT


def test_no_doc_shows_a_url_on_a_port_the_server_does_not_use():
    wrong = []
    for name in DOCS:
        text = (ROOT / name).read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            for port in _URL_PORT.findall(line):
                if int(port) != config.PORT:
                    wrong.append(f"{name}:{lineno}: {line.strip()}")
    assert not wrong, (
        f"these doc lines target a port that is not config.PORT ({config.PORT}); "
        f"copy-pasting them fails against a default install:\n" + "\n".join(wrong))


def test_every_doc_tables_the_real_port_default():
    wrong = []
    for name in DOCS:
        text = (ROOT / name).read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            for shown in _TABLE_PORT.findall(line):
                if shown != repr(config.PORT):
                    wrong.append(f"{name}:{lineno}: PORT documented as {shown}")
    assert not wrong, "\n".join(wrong)
