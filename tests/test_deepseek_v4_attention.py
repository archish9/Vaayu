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


# ── A5.5 step 4: DeepseekV4Compressor ───────────────────────────────────────


@pytest.fixture
def checkpoint_kernels(monkeypatch):
    """The TileLang kernels ship inside the checkpoint. `act_quant` /
    `fp4_act_quant` are called from `DeepseekV4Compressor.forward`, and
    `KERNEL_DIR` is not set by anything in this test module -- A5.7's loader
    is what normally does it in the server.
    """
    monkeypatch.setattr(modeling, "KERNEL_DIR", parity.CKPT / "inference")
    monkeypatch.setattr(modeling, "_KERNELS", None)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@needs_checkpoint
@needs_gpu
@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_chunked_equals_whole_prompt(ratio, checkpoint_kernels):
    """The acceptance shape, at the Compressor alone.

    Feeding 1024 tokens as one call and as two 512-token calls must leave the
    SAME compressed cache, up to the model's own numerical noise. The
    reference cannot do the second; this is the path A5.5 writes.

    Gated by spec §5.1's methodology, NOT a fixed tolerance: no BLAS
    guarantees bit-invariance across the batch (M) dimension (cuBLAS picks
    tile/split-k strategy by shape), so `wkv`/`wgate` computed over a
    512-token chunk and over the same positions inside a 1024-token call are
    allowed to reduce in a different order and disagree by a few ULPs -- that
    is expected GPU behaviour, not a bug, and a `torch.equal` gate here would
    be measuring the hardware, not the port. So the gate is calibrated
    against the model's OWN noise, measured in the same run: perturb the
    input by one ULP (`nextafter`) and re-run the WHOLE-prompt path against
    itself. That rel-L2 is the floor no implementation can beat -- the
    threshold is exactly that floor (`rel-L2 <= floor`), not a multiple of
    it: "chunking must perturb the output no more than a last-bit change to
    the input does" is a property, not a tuned constant.

    Mutation-verified (2026-09-24, see the A5.5 task-4 report): reinstating
    the overlap-carry bug this test's Task 4 fix removed (`_hold`'s
    `state.kv_state[:, :offset] = state.kv_state[:, offset:]` shift, with the
    explicit carry write deleted) makes `ratio=4` FAIL at rel-L2 2.82x the
    floor, comfortably outside the `<= 1x` gate; correct code sits at 0.017x
    the floor -- a ~160x separation, all margin. A first attempt at this gate
    used `<= 3 * floor` and did NOT catch that same bug (0.0609 landed 6.1%
    under a 0.0648 threshold) -- the statistic (whole-tensor rel-L2) was
    always fine, the multiplier was the defect.

    `ratio=128` never enters the overlap branch the carry bug lives in (it
    isn't `compress_ratio == 4`), so its rel-L2 is 0 with or without that bug
    -- it gives ZERO coverage of the carry logic. Do not read "both
    parametrizations passed" as two independent confirmations of the carry
    path; only `ratio=4` exercises it.
    """
    reference = parity.reference_modules()
    args = parity.reference_args(reference, shipped_config(), max_seq_len=2048)
    config = shipped_config()
    layer_id = next(i for i, r in enumerate(args.compress_ratios[:args.n_layers])
                    if r == ratio)

    with parity.reference_device(reference):
        x = parity.seeded_input(1, 1024, args.dim)
        freqs = modeling.precompute_freqs_cis(
            args.rope_head_dim, 2048, device=torch.device("cuda"),
            **modeling.rope_settings(config, ratio))

        def run(chunks, x):
            comp = modeling.DeepseekV4Compressor(config, ratio, config.head_dim)
            comp.load_state_dict(parity.load_module_weights(
                f"layers.{layer_id}.attn.compressor."))
            comp = comp.cuda().eval()
            state = modeling.DeepseekV4Cache(
                config, max_seq_len=2048, device=torch.device("cuda")
            ).layers[layer_id]
            pos = 0
            for length in chunks:
                comp(x[:, pos:pos + length], pos, state, freqs)
                pos += length
            return state.keys[:, config.sliding_window:].clone()

        whole = run([1024], x)
        split = run([512, 512], x)
        # The smallest possible perturbation this dtype can represent -- pure
        # numerical noise, no logic difference at all.
        whole_perturbed = run([1024], torch.nextafter(x, x + 1))

    got = _rel_l2(whole, split)
    floor = _rel_l2(whole, whole_perturbed)
    print(f"\nratio={ratio}: chunked-vs-whole rel-L2 {got:.6g}, "
          f"null (1-ULP) floor {floor:.6g}")
    threshold = max(floor, 1e-6)
    assert got <= threshold, (
        f"chunked-vs-whole rel-L2 {got:.6g} exceeds the null floor "
        f"{floor:.6g} (ratio {ratio}) -- chunking must perturb the output no "
        f"more than a last-bit change to the input does; this looks like a "
        f"real chunking bug, not GPU noise")


def test_compressor_flushes_once_per_ratio_positions():
    """Structural, no weights: a chunk spanning many block boundaries must
    flush every block it completes, not just the last one."""
    written = modeling.compressor_flush_positions(ratio=4, start_pos=6, seqlen=20)
    # blocks completing in [6, 26): those ending at 7, 11, 15, 19, 23
    assert written == [7, 11, 15, 19, 23]


def test_compressor_flush_positions_reduce_to_the_reference_at_decode():
    """seqlen == 1 flushes iff (start_pos + 1) % ratio == 0 (model.py:352)."""
    for start_pos in range(0, 40):
        got = modeling.compressor_flush_positions(4, start_pos, 1)
        want = [start_pos] if (start_pos + 1) % 4 == 0 else []
        assert got == want, start_pos


# ── A5.5 step 5: DeepseekV4Indexer ──────────────────────────────────────────


def test_indexer_causal_bound_is_per_query_not_per_chunk():
    """The reference masks only when start_pos == 0. Inside a chunk, query i
    may see blocks j < (start_pos + i + 1) // ratio -- NOT the chunk's bound.
    Without this a query early in the chunk reads its own future, which is
    silently plausible text.
    """
    bound = modeling.indexer_causal_bound(ratio=4, start_pos=512, seqlen=512,
                                          device=torch.device("cpu"))
    assert bound.shape == (512, 1)
    assert int(bound[0]) == (512 + 0 + 1) // 4
    assert int(bound[511]) == (512 + 511 + 1) // 4


@needs_checkpoint
@needs_gpu
def test_indexer_selects_index_topk_once_saturated(checkpoint_kernels):
    """Vaayu's working point: at 2048 fed positions the Indexer really does
    return index_topk = 512 columns, which is what makes topk 640."""
    config = shipped_config()
    reference = parity.reference_modules()
    args = parity.reference_args(reference, config, max_seq_len=8192)
    layer_id = next(i for i, r in enumerate(args.compress_ratios[:args.n_layers])
                    if r == 4)

    with parity.reference_device(reference):
        indexer = modeling.DeepseekV4Indexer(config, 4)
        indexer.load_state_dict(parity.load_module_weights(
            f"layers.{layer_id}.attn.indexer."))
        indexer = indexer.cuda().eval()
        state = modeling.DeepseekV4Cache(
            config, max_seq_len=8192, device=torch.device("cuda")).layers[layer_id]
        freqs = modeling.precompute_freqs_cis(
            args.rope_head_dim, 8192, device=torch.device("cuda"),
            **modeling.rope_settings(config, 4))
        x = parity.seeded_input(1, 2048, config.hidden_size)
        qr = parity.seeded_input(1, 2048, config.q_lora_rank, seed=2)
        idx = indexer(x, qr, 0, config.sliding_window, state, freqs)

    assert idx.shape[-1] == config.index_topk


@needs_checkpoint
@needs_gpu
def test_indexer_matches_the_reference_at_prefill(checkpoint_kernels):
    """Value-level parity, not shape-only: at `start_pos == 0` our Indexer and
    the checkpoint's own `Indexer` (`model.py:386`) run the same masking
    formula over the same weights, so their selected block-index SETS must
    agree exactly.

    Fed length is 4096, not 2048: at exactly 2048 tokens `blocks == index_topk
    == 512`, so `score.topk(min(index_topk, blocks), ...)` is asked for
    EVERY candidate and returns a trivial permutation of `0..511` on both
    sides regardless of the scores -- a scorer that always returned the WORST
    512 would pass an index-set comparison at that length exactly as well as
    a correct one, because the causal mask alone (not the score) decides
    which of "all of them" survive. `blocks > index_topk` is required for the
    comparison to actually exercise the einsum/weights_proj/relu path; 4096
    tokens gives `blocks = 1024`, so every row from i=2047 on has more
    visible candidates than slots and a real top-512 selection happens.

    `offset=0` on both sides: offset only shifts VALID selections uniformly
    (`-1` stays `-1`), so it cannot change which blocks are chosen -- passing
    0 makes the returned values the raw block indices directly, comparable
    set-for-set with no conversion.
    """
    config = shipped_config()
    reference = parity.reference_modules()
    args = parity.reference_args(reference, config, max_seq_len=4096)
    layer_id = next(i for i, r in enumerate(args.compress_ratios[:args.n_layers])
                    if r == 4)
    seqlen = 4096
    # Runtime guard for the vacancy the docstring above warns about: at
    # blocks == index_topk, topk returns every candidate regardless of score
    # and this comparison would pass just as well for a broken scorer. Derived
    # from the config, not a literal, so a future edit to seqlen or
    # index_topk can't silently reopen the vacuous length.
    assert seqlen // 4 > config.index_topk, (
        f"seqlen={seqlen} gives blocks={seqlen // 4} <= index_topk="
        f"{config.index_topk}: topk would return every candidate and this "
        f"test would not exercise real scoring")

    with parity.reference_device(reference):
        x = parity.seeded_input(1, seqlen, config.hidden_size)
        qr = parity.seeded_input(1, seqlen, config.q_lora_rank, seed=2)
        freqs = modeling.precompute_freqs_cis(
            args.rope_head_dim, seqlen, device=torch.device("cuda"),
            **modeling.rope_settings(config, 4))
        weights = parity.load_module_weights(f"layers.{layer_id}.attn.indexer.")

        with torch.device("cuda"):
            reference_indexer = reference.Indexer(args, 4)
        reference_indexer.load_state_dict(weights)
        reference_indexer = reference_indexer.eval()
        # Normally wired by Attention.forward (model.py:499-500); nothing
        # else does it when the Indexer is exercised standalone.
        reference_indexer.freqs_cis = freqs
        reference_idx = reference_indexer(x, qr, 0, 0)

        ours = modeling.DeepseekV4Indexer(config, 4)
        ours.load_state_dict(weights)
        ours = ours.cuda().eval()
        state = modeling.DeepseekV4Cache(
            config, max_seq_len=seqlen, device=torch.device("cuda")).layers[layer_id]
        our_idx = ours(x, qr, 0, 0, state, freqs)

    assert reference_idx.shape == our_idx.shape

    reference_sets = [set(row[row >= 0].tolist()) for row in reference_idx[0]]
    our_sets = [set(row[row >= 0].tolist()) for row in our_idx[0]]
    mismatches = [i for i in range(seqlen) if reference_sets[i] != our_sets[i]]
    if mismatches:
        i = mismatches[0]
        only_reference = reference_sets[i] - our_sets[i]
        only_ours = our_sets[i] - reference_sets[i]
        print(f"\nrow {i}: {len(mismatches)}/{seqlen} rows differ; "
              f"row {i} reference-only={sorted(only_reference)[:10]} "
              f"ours-only={sorted(only_ours)[:10]}")
    assert not mismatches, (
        f"{len(mismatches)}/{seqlen} rows selected a different block set than "
        f"the reference at the same start_pos == 0 input -- not a shape or "
        f"offset issue, a real disagreement in which blocks were chosen")


# ── A5.5 step 6: DeepseekV4Attention, and the four goldens ─────────────────


def _golden(name):
    import numpy as np
    z = np.load(parity.GOLDENS / f"{name}.npz", allow_pickle=True)
    out = {}
    for key in z.files:
        if key.startswith("__"):
            continue
        dtype = str(z[f"__dtype__{key}"])
        out[key] = torch.from_numpy(z[key]).view(getattr(torch, dtype))
    return out


def _raw_module_weights(prefix: str) -> dict:
    """The tensors under `prefix` exactly as the shards hold them -- no value
    transforms.

    NOT `parity.load_module_weights`: that loader pre-dequantizes `wo_a` for
    feeding the REFERENCE module (whose `wo_a` Parameter is declared plain
    bf16) and drops `wo_a.scale` entirely in the process. Our own
    `DeepseekV4Attention.load_state_dict` needs the RAW fp8 weight + scale so
    `modeling.dequantize_at_load` -- the production rule -- has something to
    consume; calling it on `parity.load_module_weights`'s already-transformed
    output raises `KeyError: 'wo_a.scale'`, exactly as `_raw_module_weights` in
    `test_deepseek_v4_linear.py` (whose `test_the_production_dequant_...` test
    exists to prove these two paths agree) is used, not `parity`'s.
    """
    from safetensors import safe_open

    index = json.loads((parity.CKPT / "model.safetensors.index.json").read_text())
    wanted = sorted(key for key in index["weight_map"] if key.startswith(prefix))
    raw = {}
    for shard in sorted({index["weight_map"][key] for key in wanted}):
        with safe_open(parity.CKPT / shard, framework="pt", device="cpu") as handle:
            for key in wanted:
                if index["weight_map"][key] == shard:
                    raw[key[len(prefix):]] = handle.get_tensor(key)
    return raw


@needs_checkpoint
@needs_gpu
def test_attention_matches_the_reference_at_a_ratio_0_layer(checkpoint_kernels):
    """Value-level parity against the reference `Attention` (`model.py:442`) on
    a `compress_ratio == 0` layer, at `start_pos == 0`.

    Every other parity test in this file exercises a compressed layer (ratio 4
    or 128); the `attn_*` goldens A5.2 recorded only cover those two, and
    `test_chunked_prefill_equals_whole_prompt[0-*]` compares OUR module against
    itself on both arms, so it is true by construction and proves nothing
    against the reference. Ratio-0 layers (layers 0-1 of the real model) are
    the ONLY users of the YaRN-off rope variant (`rope_settings(config, 0)` --
    `original_seq_len = 0`, base `rope_theta`, model.py:483-484) -- a wrong
    YaRN-off table would leave every existing r0 gate green, because none of
    them check it against anything outside this port.

    A ratio-0 layer skips `compressor`/`indexer` entirely and every op
    (`q`/`kv` projections, rotary, the window-only `sparse_attn`) is
    position-local, so both sides run identical arithmetic at identical
    shapes -- this compares `torch.equal`, not a tolerance. A mismatch here
    means the YaRN-off path itself diverges, not a reassociation artifact.
    """
    config = shipped_config()
    reference = parity.reference_modules()
    args = parity.reference_args(reference, config, max_seq_len=parity.MAX_SEQ_LEN)
    layer_id = next(i for i, r in enumerate(args.compress_ratios[:args.n_layers])
                    if r == 0)
    assert config.compress_ratios[layer_id] == 0

    seqlen = 300
    with parity.reference_device(reference):
        x = parity.seeded_input(1, seqlen, config.hidden_size)

        reference_layer = parity.build_attention(reference, args, layer_id)
        reference_out = reference_layer(x, 0)

        ours = modeling.DeepseekV4Attention(config, layer_id)
        ours.load_state_dict(modeling.dequantize_at_load(
            _raw_module_weights(f"layers.{layer_id}.attn.")))
        ours = ours.cuda().eval()
        cache = modeling.DeepseekV4Cache(config, max_seq_len=parity.MAX_SEQ_LEN,
                                         device=torch.device("cuda"))
        our_out = ours(x, cache, 0)

    assert our_out.shape == reference_out.shape
    assert torch.equal(our_out.cpu(), reference_out.cpu()), (
        f"ratio-0 layer {layer_id} diverges from the reference at start_pos == "
        f"0: rel-L2 "
        f"{((our_out.cpu().float() - reference_out.cpu().float()).norm() / reference_out.cpu().float().norm()).item():.6g}"
        f" -- the YaRN-off rope table or another position-local op is wrong, "
        f"not a reassociation artifact (see docstring)")


@needs_checkpoint
@needs_gpu
@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_attention_reproduces_a5_2_golden(ratio, phase, checkpoint_kernels):
    """A5.3 and A5.4 both landed bit-identical wherever the arithmetic was
    shared. Anything less here is a FINDING that must name the diverging op --
    not a tolerance to widen.

    `attn_prefill_r4` is the one exception, and it is gated the way Task 4
    gates chunk equivalence rather than left on `torch.equal`. Root cause
    (pinned by `test_attn_prefill_r4_divergence_is_tile_width_not_masking`
    below): `sparse_attn_kernel` tiles its online-softmax reduction in
    64-wide chunks; `window_topk_idxs`' general-`seqlen` width (128, needed
    so one function serves prefill/chunk/decode -- the reference's own
    prefill branch instead shrinks to `min(seqlen, window_size)`) pushes this
    golden's 2 real compressed indices into a THIRD kernel tile the
    reference's shrunk, 1-tile call never creates. An online-softmax merge
    across REAL tiles is exact in real arithmetic but not bit-identical to a
    single-pass reduction -- confirmed algebraically (an all-masked EXTRA
    tile, exactly `attn_prefill_r128`'s case since it has zero compressed
    blocks, IS an exact IEEE754 no-op: multiply-by-1.0 and add-0.0 are both
    exact) and empirically (shrinking the window index to the reference's
    width reproduces the golden bit-for-bit -- the pinning test). So the gate
    here is calibrated against the model's OWN last-bit sensitivity, measured
    in the same run: perturb the golden's input by one ULP, re-run OUR
    forward on a fresh cache, and take that rel-L2 as the floor. The
    threshold is exactly that floor, not a multiple of it -- same principle
    as `test_compressor_chunked_equals_whole_prompt`.
    """
    config = shipped_config()
    reference = parity.reference_modules()
    args = parity.reference_args(reference, config, max_seq_len=parity.MAX_SEQ_LEN)
    layer_id = next(i for i, r in enumerate(args.compress_ratios[:args.n_layers])
                    if r == ratio)
    golden = _golden(f"attn_{phase}_r{ratio}")

    with parity.reference_device(reference):
        layer = modeling.DeepseekV4Attention(config, layer_id)
        layer.load_state_dict(modeling.dequantize_at_load(
            _raw_module_weights(f"layers.{layer_id}.attn.")))
        layer = layer.cuda().eval()
        cache = modeling.DeepseekV4Cache(config, max_seq_len=parity.MAX_SEQ_LEN,
                                         device=torch.device("cuda"))
        prefill = golden.get("prefill_x", golden.get("x")).cuda()
        out = layer(prefill, cache, 0)
        if phase == "decode":
            out = layer(golden["x"].cuda(), cache, prefill.shape[1])

        if ratio == 4 and phase == "prefill":
            perturbed_cache = modeling.DeepseekV4Cache(
                config, max_seq_len=parity.MAX_SEQ_LEN, device=torch.device("cuda"))
            perturbed_out = layer(torch.nextafter(prefill, prefill + 1),
                                  perturbed_cache, 0)
            floor = _rel_l2(out.cpu(), perturbed_out.cpu())
            got = _rel_l2(out.cpu(), golden["out"])
            print(f"\nattn_prefill_r4: rel-L2 vs golden {got:.6g}, "
                  f"null (1-ULP) floor {floor:.6g}")
            assert got <= max(floor, 1e-6), (
                f"attn_prefill_r4 rel-L2 {got:.6g} exceeds the null floor "
                f"{floor:.6g} -- no longer explained by tile-width "
                f"reassociation alone, this looks like a real bug")
            return

    assert torch.equal(out.cpu(), golden["out"]), \
        f"max rel {((out.cpu().float() - golden['out'].float()).norm() / golden['out'].float().norm()).item()}"


@needs_checkpoint
@needs_gpu
def test_attn_prefill_r4_divergence_is_tile_width_not_masking(checkpoint_kernels):
    """Pins the root cause of `attn_prefill_r4`'s rel-L2 divergence as tile
    width, not a masking or indexing bug.

    Replays `DeepseekV4Attention.forward`'s exact sequence of operations
    (duplicated rather than parametrizing `forward` itself -- this is a
    diagnostic for exactly one golden, not a knob production code should
    grow) with ONE change: `window_topk_idxs`' output is narrowed to the
    reference's own prefill width, `min(seqlen, window_size)`, before the
    compressed indices are appended -- so the 2 real compressed indices land
    in the SAME `sparse_attn_kernel` tile as the window ones, exactly as the
    reference's own (shrunk) call does. If tile width is really the only
    source of divergence, this reproduces the golden bit-for-bit.

    If it does NOT, `test_attention_reproduces_a5_2_golden`'s null-floor gate
    is hiding a real bug behind FP-noise framing, and that must be
    re-diagnosed before trusting it.

    DRIFT GUARD: this replay duplicates `forward`'s body rather than calling
    it, so it pins a COPY -- if `forward` changes and this copy does not, the
    copy keeps passing while pinning nothing. So the replay also runs once
    WITHOUT the window shrink (`shrink=False` below) and asserts THAT result
    matches `layer(hidden_states, cache, 0)`'s own output exactly: with no
    shrink, the replay is a plain unrolling of `forward` with zero behavioural
    change, so it must agree with the real call bit-for-bit. Drift in either
    body breaks that agreement loudly instead of silently.
    """
    ratio = 4
    config = shipped_config()
    reference = parity.reference_modules()
    args = parity.reference_args(reference, config, max_seq_len=parity.MAX_SEQ_LEN)
    layer_id = next(i for i, r in enumerate(args.compress_ratios[:args.n_layers])
                    if r == ratio)
    golden = _golden(f"attn_prefill_r{ratio}")

    with parity.reference_device(reference):
        layer = modeling.DeepseekV4Attention(config, layer_id)
        layer.load_state_dict(modeling.dequantize_at_load(
            _raw_module_weights(f"layers.{layer_id}.attn.")))
        layer = layer.cuda().eval()
        hidden_states = golden["x"].cuda()
        start_pos = 0

        def replay(shrink: bool):
            cache = modeling.DeepseekV4Cache(
                config, max_seq_len=parity.MAX_SEQ_LEN, device=torch.device("cuda"))
            state = cache.layers[layer_id]
            bsz, seqlen, _ = hidden_states.size()
            rd, win = layer.rope_head_dim, layer.window_size
            device = hidden_states.device
            freqs_cis = layer._freqs(cache.max_seq_len, device)
            positions = freqs_cis[start_pos:start_pos + seqlen]

            qr = layer.q_norm(layer.wq_a(hidden_states))
            q = layer.wq_b(qr).unflatten(-1, (layer.n_heads, layer.head_dim))
            q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + layer.q_norm.eps)
            modeling.apply_rotary_emb(q[..., -rd:], positions)

            kv = layer.kv_norm(layer.wkv(hidden_states))
            modeling.apply_rotary_emb(kv[..., -rd:], positions)
            modeling.kernels().act_quant(kv[..., :-rd], modeling._ACT_QUANT_GROUP,
                                         "ue8m0", modeling._SCALE_DTYPE, True)

            history = state.history(start_pos)
            h = history.shape[1]
            # THE ONE CHANGE under test: shrink to the reference's own
            # prefill width instead of `window_topk_idxs`' general-seqlen
            # `window_size` columns. shrink=False takes the real, unmodified
            # width -- i.e. plain `forward`, unrolled.
            width = min(seqlen, win) if shrink else win
            window_idxs = modeling.window_topk_idxs(win, bsz, seqlen, start_pos,
                                                    device)[:, :, :width].contiguous()
            view = torch.cat([history, kv], dim=1)

            offset = h + seqlen
            layer.compressor(hidden_states, start_pos, state, freqs_cis)
            compress_idxs = layer.indexer(hidden_states, qr, start_pos, offset,
                                          state, freqs_cis)
            blocks = (start_pos + seqlen) // layer.compress_ratio
            view = torch.cat([view, state.keys[:bsz, win:win + blocks]], dim=1)
            topk_idxs = torch.cat([window_idxs, compress_idxs], dim=-1)

            state.write(kv, start_pos)
            state.cumulative_length = start_pos + seqlen

            o = modeling.kernels().sparse_attn(q, view, layer.attn_sink, topk_idxs,
                                               layer.softmax_scale)
            modeling.apply_rotary_emb(o[..., -rd:], positions, True)
            o = o.view(bsz, seqlen, layer.n_groups, -1)
            wo_a = layer.wo_a.weight.view(layer.n_groups, layer.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
            return layer.wo_b(o.flatten(2))

        shrunk_out = replay(shrink=True)
        unshrunk_out = replay(shrink=False)
        real_cache = modeling.DeepseekV4Cache(
            config, max_seq_len=parity.MAX_SEQ_LEN, device=torch.device("cuda"))
        real_out = layer(hidden_states, real_cache, start_pos)

    assert torch.equal(shrunk_out.cpu(), golden["out"]), (
        "shrinking window_topk_idxs to the reference's own prefill width did "
        "NOT reproduce the golden bit-for-bit -- the divergence is not "
        "(only) tile-width reassociation; re-diagnose before trusting the "
        "null-floor gate in test_attention_reproduces_a5_2_golden")
    assert torch.equal(unshrunk_out.cpu(), real_out.cpu()), (
        "the unshrunk replay no longer matches DeepseekV4Attention.forward's "
        "own output -- this test pins a COPY of forward's body, and the copy "
        "has drifted from the real implementation; update the replay before "
        "trusting the tile-width diagnosis above")


@needs_checkpoint
@needs_gpu
@pytest.mark.parametrize("ratio", [4, 128])
def test_attention_does_not_raise_where_the_reference_did(ratio, checkpoint_kernels):
    """A5.2's `attn_chunked` goldens store NO output -- their content is the
    reference's RuntimeError plus `prefill_x`, with the chunk input
    reproducible as seeded_input(1, 512, dim, seed=1). The port must run it.
    """
    import numpy as np
    note = str(np.load(parity.GOLDENS / f"attn_chunked_r{ratio}.npz",
                       allow_pickle=True)["__note__"])
    assert "RuntimeError" in note, "the golden's premise changed"

    config = shipped_config()
    reference = parity.reference_modules()
    args = parity.reference_args(reference, config, max_seq_len=parity.MAX_SEQ_LEN)
    layer_id = next(i for i, r in enumerate(args.compress_ratios[:args.n_layers])
                    if r == ratio)
    golden = _golden(f"attn_chunked_r{ratio}")

    with parity.reference_device(reference):
        layer = modeling.DeepseekV4Attention(config, layer_id)
        layer.load_state_dict(modeling.dequantize_at_load(
            _raw_module_weights(f"layers.{layer_id}.attn.")))
        layer = layer.cuda().eval()
        cache = modeling.DeepseekV4Cache(config, max_seq_len=parity.MAX_SEQ_LEN,
                                         device=torch.device("cuda"))
        prefill = golden["prefill_x"].cuda()
        layer(prefill, cache, 0)
        step = parity.seeded_input(1, parity.CHUNK_LEN, config.hidden_size, seed=1)
        out = layer(step, cache, prefill.shape[1])

    assert out.shape == (1, parity.CHUNK_LEN, config.hidden_size)
    assert torch.isfinite(out.float()).all()


# ── A5.5 step 7: chunk equivalence, the acceptance gate ─────────────────────


def _null_floor(layer, cache_factory, x):
    """How far apart two runs land for reasons that are NOT the chunking.

    Perturb the input in its last representable bit and re-run whole. The
    spread that produces is the floor a chunking difference must beat. D6
    (2026-09-18) is why a fixed epsilon is not evidence here: a null bit change
    moves this model's numbers by more than a careless tolerance would allow.
    """
    cache = cache_factory()
    base = layer(x, cache, 0)
    perturbed = x.clone()
    flat = perturbed.view(-1)
    flat[::997] = (flat[::997].float() * (1 + 2 ** -8)).to(flat.dtype)
    cache = cache_factory()
    other = layer(perturbed, cache, 0)
    return ((base.float() - other.float()).norm()
            / base.float().norm()).item(), base


@needs_checkpoint
@needs_gpu
@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("chunks", [[512, 512], [256, 256, 256, 256],
                                    [511, 513]])
def test_chunked_prefill_equals_whole_prompt(ratio, chunks, checkpoint_kernels):
    """THE ACCEPTANCE CRITERION for A5.5.

    A5.5 has no model, so the object under test is one Attention module on real
    weights, one per compress_ratio. Feed 1024 tokens whole, then in slices,
    and compare the outputs at the final slice's positions against a
    null-perturbation floor measured in the SAME run.

    Two aligned chunk counts, plus one ([511, 513]) whose boundary is NOT a
    multiple of either compress_ratio (4 or 128) -- every other split here
    lands on a block boundary for both ratios, so the Compressor's `held > 0`
    partial-block carry path (exactly where Task 4's only real bug lived) was
    otherwise never exercised by this gate. Widening the tolerance to fit the
    result is the row-43 failure re-run -- STOP and raise instead.
    """
    config = shipped_config()
    reference = parity.reference_modules()
    args = parity.reference_args(reference, config, max_seq_len=2048)
    layer_id = next(i for i, r in enumerate(args.compress_ratios[:args.n_layers])
                    if r == ratio)
    total = sum(chunks)

    with parity.reference_device(reference):
        layer = modeling.DeepseekV4Attention(config, layer_id)
        # Not parity.load_module_weights: that loader pre-dequantizes wo_a for
        # the REFERENCE module and drops wo_a.scale, which raises KeyError
        # here exactly as it does in test_attention_reproduces_a5_2_golden --
        # see _raw_module_weights's docstring above.
        layer.load_state_dict(modeling.dequantize_at_load(
            _raw_module_weights(f"layers.{layer_id}.attn.")))
        layer = layer.cuda().eval()

        def fresh():
            return modeling.DeepseekV4Cache(config, max_seq_len=2048,
                                            device=torch.device("cuda"))

        x = parity.seeded_input(1, total, config.hidden_size)
        floor, whole = _null_floor(layer, fresh, x)

        cache, pos, pieces = fresh(), 0, []
        for length in chunks:
            pieces.append(layer(x[:, pos:pos + length], cache, pos))
            pos += length
        tail = pieces[-1]

    want = whole[:, total - tail.shape[1]:]
    got = ((tail.float() - want.float()).norm() / want.float().norm()).item()
    print(f"\nratio={ratio} chunks={chunks}: chunked-vs-whole rel-L2 {got:.6g}, "
          f"null floor {floor:.6g}")
    # The bound is ONE floor, not a multiple of it, and that is a measured
    # choice rather than a taste: chunking must perturb the output no more than
    # a last-bit change to the input does. Task 4 learned this the hard way --
    # its `3 x floor` gate PASSED with its own motivating bug reinstated
    # (2.82 x floor), while correct code sat at 0.017x floor. Correct and
    # broken are separated by ~160x, so any threshold in between works and the
    # principled one is 1.
    assert got <= max(floor, 1e-6), (
        f"chunked prefill diverges: rel-L2 {got:.6f} against a null floor of "
        f"{floor:.6f}. Do NOT widen this bound -- find the site.")
