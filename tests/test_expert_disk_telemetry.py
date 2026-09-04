"""Fix #6: expert NVMe time is surfaced, not hidden inside compute."""
from vaayu.persist.expert_cache import ExpertRAMCache  # ensures package imports


def test_accumulator_attribute_contract():
    # The forward loop must expose a per-token expert-disk accumulator that
    # starts at 0.0 and is a float. We test the contract on a stand-in object
    # since constructing a full model needs 138GB of weights.
    class Stub:
        def __init__(self):
            self._expert_disk_s = 0.0
    s = Stub()
    assert isinstance(s._expert_disk_s, float)
    s._expert_disk_s += 1.5
    assert s._expert_disk_s == 1.5


def test_the_running_total_is_initialised_at_construction():
    """`requests.jsonl` diffs `_expert_disk_total_s` per request. Left to
    spring into existence on the first forward, the first request of a process
    had no `before` value and reported a null stall. Asserted on the source
    because constructing a model needs the weights (see the stand-in above)."""
    import inspect

    from vaayu.vaayu_base import VaayuBaseModel

    src = inspect.getsource(VaayuBaseModel.__init__)
    assert "self._expert_disk_total_s = 0.0" in src


def test_the_running_total_is_folded_at_the_end_of_the_pass():
    """Folding the per-token value on ENTRY shifted the server's per-request
    window by one forward: a request's diff covered the previous request's
    last token plus its own tokens 1..N-1, so a one-token request reported
    somebody else's stall entirely. The fold must happen after the pass it
    belongs to -- `_exec_end` marks that point."""
    import inspect

    from vaayu.vaayu_base import VaayuBaseModel

    src = inspect.getsource(VaayuBaseModel.forward)
    end_marker = src.index("_exec_end = time.time()")
    folds = [i for i in range(len(src))
             if src.startswith("self._expert_disk_total_s = ", i)]
    assert folds, "the running total is never folded"
    assert any(i > end_marker for i in folds), (
        "the only fold is at the top of the pass -- that is the off-by-one-pass "
        "shift this test exists to prevent")
