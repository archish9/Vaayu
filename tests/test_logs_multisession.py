"""Session attribution under N>1 (issue L1 T4).

The single-session marker was sound only while one generation ran at a time.
Batched decode breaks that: several sessions share one forward and one output
stream. A shared line must be tagged `session=*` -- never guessed at one
member -- and must still reach every member's own log, or reading one
session's file during a batch shows tokens with none of the engine work that
produced them.
"""
from vaayu.server import logs


def _reset():
    logs.clear_current_session()


def test_one_member_is_todays_behaviour():
    _reset()
    logs.set_current_session("alpha")
    tag, members = logs.current_marker()
    assert tag == "alpha"
    assert members == ("alpha",)
    assert logs.current_session() == "alpha"
    _reset()


def test_two_members_tag_star_and_keep_both():
    _reset()
    logs.set_current_sessions(["alpha", "beta"])
    tag, members = logs.current_marker()
    assert tag == "*", "a shared line must not be attributed to one member"
    assert set(members) == {"alpha", "beta"}
    assert logs.current_session() == "*"
    _reset()


def test_members_are_deduped_and_unnamed_sessions_get_the_placeholder():
    _reset()
    logs.set_current_sessions(["alpha", "alpha", None])
    _tag, members = logs.current_marker()
    assert members == ("alpha", logs.UNNAMED)
    _reset()


def test_clearing_returns_to_idle():
    _reset()
    logs.set_current_sessions(["alpha", "beta"])
    logs.clear_current_session()
    assert logs.current_marker() == (None, ())
    assert logs.current_session() is None


def test_an_overlap_loses_attribution_rather_than_guessing():
    """Two generations in flight is an invariant violation once the scheduler
    exists -- the dispatcher is the only setter and always clears first. It
    must not RAISE (a logging fault must never become a 500), and it must not
    guess: no member gets the line."""
    _reset()
    logs.set_current_session("alpha")
    logs.set_current_session("beta")
    tag, members = logs.current_marker()
    assert tag == "*"
    assert members == (), "an unattributable line belongs to nobody's log"
    _reset()


def test_shared_lines_are_written_to_every_member_log(tmp_path):
    _reset()
    logs.set_current_sessions(["alpha", "beta"])
    tag, members = logs.current_marker()
    for member in members:
        logs._session_write(tmp_path, member,
                            f"2026-09-08T00:00:00Z session={tag} hello\n".encode(),
                            rotate_mb=50)
    alpha = (tmp_path / "sessions" / "alpha.log").read_text()
    beta = (tmp_path / "sessions" / "beta.log").read_text()
    assert "session=*" in alpha and "hello" in alpha
    assert "session=*" in beta and "hello" in beta
    _reset()
