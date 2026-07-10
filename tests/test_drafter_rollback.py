"""Fix #9 Task 3: drafter cache rollback is incremental, never full re-prefill."""
from vaayu.speculative.standard import drafter_rollback_target


def test_rejection_keeps_accepted_prefix():
    # Before draft: drafter cache covers 100 tokens. Drafted 5 (cache grew to
    # 104: the 5th draft is predicted but not yet fed). Verifier accepted 3
    # (2 correct drafts + 1 correction the drafter never saw).
    # Valid drafter prefix = 100 + (3 - 1) = 102.
    assert drafter_rollback_target(seen_before_draft=100, num_drafted=5, num_accepted=3) == 102


def test_full_accept_keeps_everything_seen():
    # All 5 drafts accepted + bonus token (num_accepted = 6). Drafter cache
    # covers 100 + 4 = 104 fed tokens, all still valid. Target = 104; the
    # 2 unseen tokens (5th draft + bonus) arrive via the unseen-suffix feed.
    assert drafter_rollback_target(seen_before_draft=100, num_drafted=5, num_accepted=6) == 104


def test_immediate_rejection():
    # First draft wrong: accepted = 1 (just the correction). Valid prefix = 100.
    assert drafter_rollback_target(seen_before_draft=100, num_drafted=5, num_accepted=1) == 100
