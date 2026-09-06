"""Suite-wide fixtures.

One job today: keep the tests out of the repo's own `logs/` root. A test that
builds a bare `ServerConfig` inherits `log_dir="logs"` -- relative, so it
resolves against the CWD, which for a test run is the repo -- and every
`create_app(config=...)` lifespan then creates a real `run-<UTC>/` there. That
is ~11 stray dirs per full suite run, and they dilute the real runs the logging
work exists to make trustworthy: `latest` ends up pointing at whichever test
ran last.
"""
import pytest


@pytest.fixture(autouse=True)
def redirect_default_log_root(tmp_path_factory, monkeypatch):
    """Send the DEFAULT log root under pytest's tmp root; leave explicit ones alone.

    Redirects rather than disables: the run dir is still created, so the
    run-dir path stays under test (a fixture returning None would hide every
    bug in it). A test that passes its own `log_dir=` -- a read-only root, a
    path it then reads back -- is untouched.
    """
    try:
        from vaayu.server import app as app_module
    except ImportError:                 # no fastapi: server tests skip anyway
        yield
        return

    import dataclasses

    default = next(f.default for f in dataclasses.fields(app_module.ServerConfig)
                   if f.name == "log_dir")
    real = app_module.new_run_dir

    def redirected(root):
        # mktemp only on the tests that actually boot a default-config app --
        # this fixture is autouse, and an eager `tmp_path` would make a
        # directory for every test in the suite.
        return real(tmp_path_factory.mktemp("logs") if str(root) == default
                    else root)

    monkeypatch.setattr(app_module, "new_run_dir", redirected)
    yield
