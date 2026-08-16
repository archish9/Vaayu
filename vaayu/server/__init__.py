"""Vaayu's OpenAI-compatible HTTP server.

Design: plans/2026-08-16-vaayu-server-design.md
Plan:   plans/2026-08-16-vaayu-server-plan.md

Import order matters: `app` pulls in FastAPI, so the lazy re-export below keeps
`import vaayu` cheap for every non-server caller (every test script in the repo).
"""

__all__ = ["create_app", "ServerConfig"]


def __getattr__(name):
    if name in __all__:
        from . import app as _app
        return getattr(_app, name)
    raise AttributeError(name)
