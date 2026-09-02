"""The speculative path is GREEDY and must SAY SO, not pretend.

`speculative_generate` takes `temperature` and swallows every other sampling
control in `**kwargs`. Nothing reads either: the first token is
`prefill_out.logits[:, -1].argmax(...)`, and both speculators
(`StandardDrafterDecoder`, `PromptLookupDecoder`) argmax the drafter AND the
verifier. So a request asking for sampling silently gets greedy text --
`temperature` included, which the server's own warning claimed was honoured.

`stopping_criteria` IS honoured (checked per round at the top of the loop), so
it is never named here; HF's `stop_strings` is not, and is.

The rule this file pins: when a call asks for something this loop drops, warn
with the names. Ships with P7 -- latent only because the drafter is off by
default.
"""
import torch

from vaayu.prefix_cache import PrefixCacheStore
from vaayu.vaayu_base import VaayuBaseModel, speculative_ignored_sampling

from .test_speculative_prefix_reuse import _FakeSpecModel


def test_a_greedy_call_is_honoured_in_full_so_nothing_is_named():
    assert speculative_ignored_sampling(0.0, {}) == []
    assert speculative_ignored_sampling(0.0, {"do_sample": False}) == []


def test_temperature_is_named_because_the_loop_never_reads_it():
    # The bug this file exists for: the parameter is DECLARED and unused, and
    # the server's warning told the caller it was the one thing honoured.
    assert speculative_ignored_sampling(0.7, {}) == ["temperature"]


def test_every_requested_sampling_control_is_named():
    ignored = speculative_ignored_sampling(0.7, {"do_sample": True,
                                                 "top_p": 0.9, "top_k": 50})
    assert ignored == ["temperature", "top_p", "top_k"]


def test_do_sample_alone_is_still_named():
    # do_sample=True at temperature 0 asks for sampling too; nothing else names
    # it, so the warning would be empty on a request that does not get what it
    # asked for.
    assert speculative_ignored_sampling(0.0, {"do_sample": True}) == ["do_sample"]


def test_top_p_without_sampling_is_not_named():
    # HF's own generate() ignores top_p/top_k when do_sample is False, so this
    # call loses nothing by running here -- naming it would be a false alarm on
    # every greedy request that carries the defaults along.
    assert speculative_ignored_sampling(0.0, {"top_p": 0.9, "top_k": 50}) == []


def test_stop_strings_are_named_even_on_a_greedy_call():
    # Dropped independently of sampling: HF's stop_strings machinery never runs
    # in this hand-written loop.
    assert speculative_ignored_sampling(0.0, {"stop_strings": ["```"]}) == \
        ["stop_strings"]


def test_stopping_criteria_is_not_named_because_it_is_honoured():
    # Checked per round at the top of the while loop -- this is what carries
    # request cancellation, and warning about it would send callers away from
    # the one control that works.
    assert speculative_ignored_sampling(0.0, {"stopping_criteria": object()}) == []


def _run(capsys, **kwargs):
    model = _FakeSpecModel(PrefixCacheStore(budget_bytes=10 ** 9))
    VaayuBaseModel.speculative_generate(
        model, torch.tensor([[1, 2, 3, 4, 5]]), max_new_tokens=2, **kwargs)
    return capsys.readouterr().out


def test_the_generation_warns_once_naming_what_it_dropped(capsys):
    out = _run(capsys, temperature=0.7, top_p=0.9, stop_strings=["```"])
    lines = [ln for ln in out.splitlines() if "WARNING" in ln]
    assert len(lines) == 1, f"expected exactly one warning, got: {lines}"
    for name in ("temperature", "top_p", "stop_strings"):
        assert name in lines[0]
    assert "greedy" in lines[0].lower()


def test_a_greedy_generation_prints_no_warning(capsys):
    assert "WARNING" not in _run(capsys)
