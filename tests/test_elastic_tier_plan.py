"""S35-E: how much of the load-time reserve ONE request needs (vaayu.vram_plan).

Pure. The plan reserves VRAM for a request at max_seq_len; request_reserve_bytes
says what a shorter one needs, and the difference is what the expert VRAM tier may
borrow. Geometries are read off checkpoint configs (CLAUDE.md §5); the 235B-like one
is synthetic (its checkpoint is retired) and the last one is deliberately odd.
"""
import pytest

from vaayu.vram_plan import (VramShape, peak_envelope_bytes, plan_vram,
                             predict_peak_vram, request_reserve_bytes,
                             reserve_floor_bytes)

GB = 1024 ** 3
MB = 1024 ** 2
FLOOR = int(3.5 * GB)            # vaayu_base._VRAM_RESERVE_FLOOR_BYTES: a hardware headroom
FREE = 12 * GB                   # free VRAM on a 16 GB card after pinning: hardware

SHAPES = {
    # eager attention, per-chunk prefill, windowed-KV constant (gpt-oss-20b's load log)
    "gptoss_like": VramShape(
        kv_per_token=24576, N=1, chunk=512, hidden=2880, n_heads=64, vocab=201088,
        elem_bytes=2, score_elem_bytes=6.03125, alloc_factor=1.0, head_bytes=1158266976,
        score_bytes_per_key=16384.0, kv_constant_bytes=28286976),
    # sdpa, layer-major prefill (the sawtooth form)
    "minimax_like_sdpa": VramShape(
        kv_per_token=253952, N=1, chunk=512, hidden=3072, n_heads=48, vocab=200064,
        elem_bytes=2, score_elem_bytes=10.9, alloc_factor=1.0, head_bytes=1229193216,
        layer_major_prefill=True),
    # sdpa, chunk-major expert retention
    "qwen235_like_sdpa": VramShape(
        kv_per_token=192512, N=1, chunk=512, hidden=4096, n_heads=64, vocab=151936,
        elem_bytes=2, score_elem_bytes=10.9, alloc_factor=1.0, head_bytes=1244659712,
        resident_expert_bytes=128 * 11827937, layer_major_prefill=True),
    # flash: no score buffer, so the reserve is the floor
    "qwen3next_like_flash": VramShape(
        kv_per_token=24576, N=1, chunk=512, hidden=2048, n_heads=16, vocab=151936,
        elem_bytes=2, score_elem_bytes=0.0, alloc_factor=1.0, head_bytes=622329856,
        layer_major_prefill=True),
    # nothing here is 512 or 2
    "odd_chunk100_growth3": VramShape(
        kv_per_token=50000, N=1, chunk=100, hidden=512, n_heads=8, vocab=30000,
        elem_bytes=2, score_elem_bytes=10.9, alloc_factor=1.0, head_bytes=10 ** 7,
        kv_growth_factor=3, layer_major_prefill=True),
}
ALL = list(SHAPES)


def plan_for(name, free=FREE):
    shape = SHAPES[name]
    floor = reserve_floor_bytes(shape=shape, base_floor=FLOOR, free_vram=free,
                                min_resident=8 * MB, layer_slack=0)
    return plan_vram(free_vram=free, shape=shape, min_resident=8 * MB,
                     requested_max_seq_len=None, position_limit=2 ** 31 - 1,
                     layer_slack=0, reserve_floor=floor)


@pytest.mark.parametrize("name", ALL)
def test_the_envelope_is_the_largest_peak_on_the_way_to_S(name):
    shape = SHAPES[name]
    top = 12 * shape.chunk + 7
    running = 0
    for s in range(1, top + 1):
        running = max(running, predict_peak_vram(shape, s))
        assert peak_envelope_bytes(shape, s) == running, s


def test_the_envelope_is_not_just_the_peak_at_S():
    """The case the envelope exists for: one token past a chunk boundary the
    layer-major form predicts LESS than at the boundary."""
    shape = SHAPES["minimax_like_sdpa"]
    assert predict_peak_vram(shape, 1025) < predict_peak_vram(shape, 1024)
    assert peak_envelope_bytes(shape, 1025) == predict_peak_vram(shape, 1024)


@pytest.mark.parametrize("name", ALL)
def test_the_envelope_of_nothing_is_nothing(name):
    assert peak_envelope_bytes(SHAPES[name], 0) == 0
    assert peak_envelope_bytes(SHAPES[name], -5) == 0


@pytest.mark.parametrize("name", ALL)
def test_a_request_reserve_stays_between_the_floor_and_the_plan(name):
    shape, plan = SHAPES[name], plan_for(name)
    previous = 0
    for s in list(range(1, 3 * shape.chunk)) + list(range(3 * shape.chunk, plan.max_seq_len,
                                                          max(1, plan.max_seq_len // 257))):
        r = request_reserve_bytes(plan, shape, s, FLOOR)
        assert FLOOR <= r <= plan.reserve_bytes, s
        assert r >= previous, f"the reserve fell as the request grew, at {s}"
        previous = r


@pytest.mark.parametrize("name", ALL)
def test_past_max_seq_len_a_request_needs_the_whole_reserve_and_never_more(name):
    """max_seq_len is the largest context whose peak fits the reserve, so one token
    more needs all of it -- and asking for far more never returns more than the plan
    holds. AT max_seq_len the request may need a hair less (the reserve can be a
    floor the last admissible token does not quite reach)."""
    shape, plan = SHAPES[name], plan_for(name)
    assert request_reserve_bytes(plan, shape, plan.max_seq_len, FLOOR) <= plan.reserve_bytes
    assert request_reserve_bytes(plan, shape, plan.max_seq_len + 1, FLOOR) == plan.reserve_bytes
    assert request_reserve_bytes(plan, shape, 4 * plan.max_seq_len, FLOOR) == plan.reserve_bytes


@pytest.mark.parametrize("name", ["gptoss_like", "minimax_like_sdpa", "qwen235_like_sdpa"])
def test_a_score_materializing_load_has_something_to_lend_to_a_short_request(name):
    shape, plan = SHAPES[name], plan_for(name)
    assert plan.reserve_bytes > FLOOR
    assert request_reserve_bytes(plan, shape, shape.chunk, FLOOR) < plan.reserve_bytes


def test_a_load_whose_reserve_is_the_floor_lends_nothing_at_any_length():
    """Qwen3-Next, MiniMax-M2.7 and DeepSeek-V4 on the server default: reserve 3.50 GiB."""
    shape, plan = SHAPES["qwen3next_like_flash"], plan_for("qwen3next_like_flash")
    assert plan.reserve_bytes == FLOOR
    for s in (1, 511, 512, 513, 4096, plan.max_seq_len):
        assert request_reserve_bytes(plan, shape, s, FLOOR) == plan.reserve_bytes
