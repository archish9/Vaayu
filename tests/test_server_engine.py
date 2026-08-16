"""EngineHandle: admission gate and request-to-kwargs mapping.

Exercised against a fake model. No torch model, no checkpoint -- the parts that
need a real GPU are covered by test_server_smoke.py at the repo root.
"""
import asyncio

import pytest

from vaayu.server.engine import EngineHandle, QueueFull
from vaayu.server.schemas import ChatCompletionRequest

HERMES_TEMPLATE = "tools <tool_call> {% for m in messages %}{{ m.content }}{% endfor %}"
PLAIN_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


class FakeTokenizer:
    def __init__(self, chat_template=PLAIN_TEMPLATE):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, **kw):
        return [1] * 10


class FakeModel:
    def __init__(self, max_seq_len=8192, chat_template=PLAIN_TEMPLATE, speculator=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = FakeTokenizer(chat_template)
        self.speculator = speculator
        self.running_device = "cpu"
        self._vram_plan = object()      # not the CPU fallback path


def _req(**over):
    base = {"model": "vaayu", "messages": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return ChatCompletionRequest(**base)


def test_max_seq_len_is_read_off_the_model_not_stored():
    e = EngineHandle(FakeModel(max_seq_len=71981))
    assert e.max_seq_len == 71981


def test_prepare_returns_the_prompt_ids_and_the_resolved_cap():
    e = EngineHandle(FakeModel(max_seq_len=100))
    ids, cap = e.prepare(_req())
    assert ids == [1] * 10
    assert cap == 90, "an omitted max_tokens takes the whole remaining window"


def test_prepare_clamps_a_requested_cap_to_the_window():
    e = EngineHandle(FakeModel(max_seq_len=100))
    _, cap = e.prepare(_req(max_tokens=10_000))
    assert cap == 90


def test_greedy_requests_do_not_sample():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(temperature=0.0), [1, 2, 3])
    assert kw["do_sample"] is False
    assert "top_p" not in kw and "top_k" not in kw


def test_sampling_parameters_are_passed_only_when_sampling():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(temperature=0.7, top_p=0.9, top_k=40), [1, 2, 3])
    assert kw["do_sample"] is True
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9
    assert kw["top_k"] == 40


def test_stop_strings_require_the_tokenizer_and_are_passed_together():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(stop=["```"]), [1, 2, 3])
    assert kw["stop_strings"] == ["```"]
    assert kw["tokenizer"] is e.model.tokenizer


def test_the_tool_close_marker_is_added_as_a_stop_string():
    # Otherwise decode runs past the call and hallucinates the tool's result.
    e = EngineHandle(FakeModel(chat_template=HERMES_TEMPLATE))
    kw = e.build_generate_kwargs(
        _req(tools=[{"type": "function", "function": {"name": "f"}}]), [1, 2, 3])
    assert "</tool_call>" in kw["stop_strings"]


def test_session_id_is_passed_straight_through():
    e = EngineHandle(FakeModel())
    kw = e.build_generate_kwargs(_req(session_id="agent-7"), [1, 2, 3])
    assert kw["session_id"] == "agent-7"


def test_an_omitted_session_id_is_none_so_auto_prefix_match_runs():
    e = EngineHandle(FakeModel())
    assert e.build_generate_kwargs(_req(), [1, 2, 3])["session_id"] is None


def test_no_warning_when_there_is_no_speculator():
    e = EngineHandle(FakeModel(speculator=None))
    assert e.warn_if_speculative_sampling_ignored(_req(top_p=0.5)) is None


def test_a_speculator_plus_top_p_warns_because_the_path_ignores_it():
    # speculative_generate honours temperature only; top_p/top_k/stop fall into
    # **kwargs and vanish. Warn, never pretend (CLAUDE.md §5).
    e = EngineHandle(FakeModel(speculator=object()))
    msg = e.warn_if_speculative_sampling_ignored(_req(temperature=0.7, top_p=0.5))
    assert msg is not None and "top_p" in msg


def test_a_speculator_with_stop_strings_warns_too():
    e = EngineHandle(FakeModel(speculator=object()))
    assert "stop" in e.warn_if_speculative_sampling_ignored(_req(stop=["```"]))


def test_a_speculator_with_defaults_only_does_not_warn():
    e = EngineHandle(FakeModel(speculator=object()))
    assert e.warn_if_speculative_sampling_ignored(_req(temperature=0.0)) is None


def test_the_cpu_max_seq_len_fallback_is_reported_at_construction(capsys):
    # vaayu_base falls back to max_seq_len=512 when there is no VramPlan (CPU /
    # no CUDA). Defaulting the cap to "the remaining window" would silently
    # inherit a 512-token window, which is the very bug this server removes.
    m = FakeModel(max_seq_len=512)
    m._vram_plan = None
    EngineHandle(m)
    assert "512" in capsys.readouterr().out


def test_prefix_stats_reports_disabled_when_there_is_no_pool():
    assert EngineHandle(FakeModel()).prefix_stats() == "disabled"


def test_prefix_stats_delegates_to_the_pool():
    m = FakeModel()
    m._prefix_cache = type("P", (), {"stats": lambda self: "hits=3 misses=1"})()
    assert EngineHandle(m).prefix_stats() == "hits=3 misses=1"


def test_requests_beyond_the_queue_depth_are_refused():
    async def main():
        e = EngineHandle(FakeModel(), queue_depth=2)
        held = []

        async def hold():
            async with e.slot():
                held.append(1)
                await asyncio.sleep(0.2)

        a = asyncio.create_task(hold())
        b = asyncio.create_task(hold())
        await asyncio.sleep(0.05)
        with pytest.raises(QueueFull):
            async with e.slot():
                pass
        await asyncio.gather(a, b)
        assert len(held) == 2

    asyncio.run(main())


def test_the_gate_serialises_so_the_model_is_never_touched_concurrently():
    async def main():
        e = EngineHandle(FakeModel(), queue_depth=8)
        order, live = [], 0

        async def work(tag):
            nonlocal live
            async with e.slot():
                live += 1
                assert live == 1, "two requests were inside the gate at once"
                order.append(tag)
                await asyncio.sleep(0.02)
                live -= 1

        await asyncio.gather(*(work(i) for i in range(4)))
        assert sorted(order) == [0, 1, 2, 3]

    asyncio.run(main())


def test_a_slot_is_released_even_when_the_body_raises():
    async def main():
        e = EngineHandle(FakeModel(), queue_depth=1)
        with pytest.raises(ValueError):
            async with e.slot():
                raise ValueError("boom")
        assert e.depth == 0
        async with e.slot():
            pass

    asyncio.run(main())
