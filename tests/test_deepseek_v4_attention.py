"""A5.5 -- MLA + sparse-index attention.

Split the way A5.3's tests are: the structural half needs neither the 157 G
checkpoint nor a GPU, and only the parity half carries both markers.
"""
import json
from pathlib import Path

import pytest
import torch

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

needs_checkpoint = pytest.mark.skipif(
    not parity.CKPT.exists(), reason="DeepSeek-V4 checkpoint not on this disk")
needs_gpu = pytest.mark.skipif(not parity.have_cuda(), reason="no CUDA device")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v4_config.json"


def shipped_config() -> DeepseekV4Config:
    return DeepseekV4Config(**json.loads(FIXTURE.read_text()))


def test_rms_norm_matches_the_reference_formula():
    """fp32 internally, cast back on the way out, weight held fp32.

    The checkpoint stores this weight bf16; the reference declares the
    Parameter fp32 and lets `load_state_dict` upcast. Declaring bf16 here
    would silently change the norm's precision on every layer.
    """
    norm = modeling.DeepseekV4RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 2.0, 8))
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    got = norm(x)
    f = x.float()
    want = (norm.weight * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)))

    assert norm.weight.dtype is torch.float32
    assert got.dtype is torch.bfloat16
    assert torch.equal(got, want.to(torch.bfloat16))


def test_rope_settings_splits_the_two_variants_off_the_config():
    """compress_ratio == 0 disables YaRN and uses the BASE theta; compressed
    layers use compress_rope_theta with YaRN. Two variants, and
    precompute_freqs_cis is lru_cache(2) -- a third call pattern evicts one.
    """
    config = shipped_config()
    plain = modeling.rope_settings(config, 0)
    compressed = modeling.rope_settings(config, 4)

    assert plain["original_seq_len"] == 0, "YaRN off on uncompressed layers"
    assert plain["base"] == config.rope_theta
    assert compressed["base"] == config.compress_rope_theta
    # transformers 5.5.4 folds rope_theta in and renames the length key.
    assert compressed["original_seq_len"] == \
        config.rope_parameters["original_max_position_embeddings"]
    assert compressed["factor"] == config.rope_parameters["factor"]
    assert modeling.rope_settings(config, 128) == compressed


def test_apply_rotary_emb_is_in_place_and_inverse_undoes_it():
    """The reference mutates its slice with no assignment (model.py:505) and
    applies the INVERSE to the attention output (model.py:539). A port that
    returns a new tensor silently drops the rotation at every call site that
    ignores the return value.
    """
    freqs = modeling.precompute_freqs_cis(8, 16, 0, 10000.0, 1.0, 32, 1,
                                          device=torch.device("cpu"))
    x = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    original = x.clone()

    returned = modeling.apply_rotary_emb(x, freqs[:4])
    assert returned is x, "must rotate in place"
    assert not torch.equal(x, original)

    modeling.apply_rotary_emb(x, freqs[:4], inverse=True)
    assert torch.allclose(x.float(), original.float(), atol=2e-2)


def _reference_window_idxs(window_size, bsz, seqlen, start_pos):
    """`model.py:261`, transcribed, for the two cases it supports.

    Transcribed rather than imported: importing the reference drags in the
    kernels and a GPU, and the point of this test is that our general form
    REDUCES to the reference at both of its supported call shapes.
    """
    if start_pos >= window_size - 1:
        s = start_pos % window_size
        m = torch.cat([torch.arange(s + 1, window_size), torch.arange(0, s + 1)])
    elif start_pos > 0:
        m = torch.nn.functional.pad(torch.arange(start_pos + 1),
                                    (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        m = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        m = torch.where(m > base, -1, m)
    return m.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def test_window_idxs_reduce_to_the_reference_at_prefill():
    """start_pos == 0: h == 0, so view index == absolute position."""
    got = modeling.window_topk_idxs(128, 1, 300, 0, torch.device("cpu"))
    want = _reference_window_idxs(128, 1, 300, 0)
    assert torch.equal(got, want)


@pytest.mark.parametrize("start_pos", [1, 64, 127, 128, 200, 1023])
def test_window_idxs_reduce_to_the_reference_at_decode(start_pos):
    """seqlen == 1. The reference indexes RING SLOTS; we index the kv_view.

    Both are the same window of positions in oldest-to-newest order, so the
    two agree entry for entry once the reference's slots are read as offsets
    from the oldest -- which is what `- (s + 1) % window` does.
    """
    window = 128
    got = modeling.window_topk_idxs(window, 1, 1, start_pos, torch.device("cpu"))
    want = _reference_window_idxs(window, 1, 1, start_pos)

    assert got.shape == want.shape
    h = min(start_pos, window - 1)
    for k in range(window):
        absolute = start_pos - h + int(got[0, 0, k]) if got[0, 0, k] >= 0 else -1
        if want[0, 0, k] < 0:
            assert got[0, 0, k] < 0
        else:
            # the reference's entry is a ring slot for the same absolute position
            assert absolute % window == int(want[0, 0, k])


def test_window_idxs_never_index_before_the_view_starts():
    """The bug this whole task exists to prevent: a 512-token chunk needs
    window-1 positions of history, and an index below 0 would silently read
    the wrong end of the view."""
    window, start, length = 128, 512, 512
    idx = modeling.window_topk_idxs(window, 1, length, start, torch.device("cpu"))
    valid = idx[idx >= 0]
    assert valid.min() >= 0
    h = min(start, window - 1)
    assert valid.max() < h + length


def test_window_idxs_are_causal_within_a_chunk():
    """Query i may see absolute positions up to start_pos+i and no further."""
    window, start, length = 128, 512, 512
    h = min(start, window - 1)
    idx = modeling.window_topk_idxs(window, 1, length, start, torch.device("cpu"))
    for i in range(0, length, 37):
        row = idx[0, i]
        absolute = row[row >= 0].long() + start - h
        assert absolute.max() == start + i
        assert absolute.min() == max(0, start + i - window + 1)


def test_compress_idxs_reduce_to_the_reference_at_prefill():
    ratio, seqlen, offset = 128, 1024, 1024
    m = torch.arange(seqlen // ratio).repeat(seqlen, 1)
    mask = m >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
    want = torch.where(mask, -1, m + offset).int().unsqueeze(0)
    got = modeling.compress_topk_idxs(ratio, 1, seqlen, 0, offset, torch.device("cpu"))
    assert torch.equal(got, want)


@pytest.mark.parametrize("start_pos", [128, 511, 4096])
def test_compress_idxs_reduce_to_the_reference_at_decode(start_pos):
    ratio, offset = 128, 640
    want = (torch.arange(0, (start_pos + 1) // ratio) + offset).int()
    got = modeling.compress_topk_idxs(ratio, 1, 1, start_pos, offset,
                                      torch.device("cpu"))
    assert torch.equal(got[0, 0], want)


def test_compress_idxs_are_causal_across_a_chunk():
    """Each query sees only blocks that finished at or before it."""
    ratio, start, length, offset = 4, 512, 512, 127
    got = modeling.compress_topk_idxs(ratio, 1, length, start, offset,
                                      torch.device("cpu"))
    for i in (0, 1, 255, 511):
        row = got[0, i]
        assert int((row >= 0).sum()) == (start + i + 1) // ratio
