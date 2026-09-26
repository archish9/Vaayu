"""A5.2: the parity harness -- goldens from DeepSeek's OWN reference modules.

The reference implementation cannot run whole on this box (155.42 GiB of
weights against ~45.3 GiB of VRAM+RAM), but `world_size = 1` is its default
with no `dist` init, so ONE module runs here with real weights. A5.3-A5.7 are
TDD-able against what those modules produce.

Every test in this file runs off committed fixtures -- no 157 G checkpoint, no
GPU. The tests that record goldens need both and SKIP without them.
"""
import json
from pathlib import Path

import pytest

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

FIXTURES = Path(__file__).parent / "fixtures"
SHIPPED = FIXTURES / "deepseek_v4_config.json"
# The checkpoint's own `inference/config.json`, byte-for-byte. `generate.py:81`
# does `ModelArgs(**json.load(f))` on it, so it IS the reference's ModelArgs for
# this checkpoint -- a free golden for the config -> ModelArgs mapping.
REFERENCE_MODELARGS = FIXTURES / "deepseek_v4_reference_modelargs.json"


@pytest.fixture
def config() -> DeepseekV4Config:
    return DeepseekV4Config(**json.loads(SHIPPED.read_text()))


@pytest.fixture
def reference() -> dict:
    return json.loads(REFERENCE_MODELARGS.read_text())


def test_every_reference_modelargs_key_is_derived_from_the_hf_config(config, reference):
    """The mapping must cover every key the reference's own config sets. A
    missing one silently takes a `ModelArgs` DEFAULT, which for `head_dim` or
    `window_size` is another model's geometry.
    """
    assert set(reference) <= set(parity.model_args_kwargs(config))


def test_the_two_keys_beyond_the_reference_config_are_derived_not_defaulted(config, reference):
    """`inference/config.json` omits `norm_eps` and `hc_eps`, so the reference
    leans on `ModelArgs`' defaults for them. CLAUDE.md §5 forbids us the same
    move -- a default is a literal from some other model -- so the mapping
    derives both from the shipped config and this test pins where from.
    """
    kwargs = parity.model_args_kwargs(config)
    shipped = json.loads(SHIPPED.read_text())

    assert set(kwargs) - set(reference) == {"norm_eps", "hc_eps"}
    assert kwargs["norm_eps"] == shipped["rms_norm_eps"]
    assert kwargs["hc_eps"] == shipped["hc_eps"]


# The ONE field where the shipped HF config and the reference's own config
# disagree. See `test_the_shipped_mtp_layer_count_contradicts_the_reference`.
MTP_DISAGREEMENT = "n_mtp_layers"


def test_derived_modelargs_match_the_reference_config_value_for_value(config, reference):
    """The reference's `inference/config.json` is ground truth for this
    checkpoint: `generate.py:81` does `ModelArgs(**json.load(f))`. Any mapping
    slip -- a wrong rename, a RoPE field read off the un-standardized key --
    shows up here rather than as garbage from layer 43.
    """
    derived = parity.model_args_kwargs(config)

    mismatched = {k: (derived[k], reference[k])
                  for k in reference if derived[k] != reference[k]}

    assert mismatched == {MTP_DISAGREEMENT: (derived[MTP_DISAGREEMENT],
                                             reference[MTP_DISAGREEMENT])}


def test_the_shipped_mtp_layer_count_contradicts_the_reference(config, reference):
    """`num_nextn_predict_layers: 1` in the shipped config is WRONG: the
    reference says 3, the disk holds `mtp.0/1/2`, and `compress_ratios` has 46
    entries for 43 layers (= 43 + 3). A5.7 must not derive the MTP skip from
    that field. Pinned here so the contradiction cannot be quietly "fixed".
    """
    assert parity.model_args_kwargs(config)[MTP_DISAGREEMENT] == 1
    assert reference[MTP_DISAGREEMENT] == 3
    assert len(config.compress_ratios) == config.num_hidden_layers + 3


# ---------------------------------------------------------------- golden store


def test_a_golden_round_trips_through_npz_with_its_dtypes_intact(tmp_path):
    """`.npz` is numpy, and numpy has no bf16 or fp8 -- so the store has to
    carry dtypes itself. bf16 <-> fp32 is exact in both directions, which is
    why fp32 is the wire format and why this test asserts `torch.equal`, not
    `allclose`: a golden that drifts by a ULP is not a golden.
    """
    import torch

    tensors = {
        "x": torch.randn(3, 8, dtype=torch.bfloat16),
        "out": torch.randn(3, 8, dtype=torch.float32),
        "topk_idxs": torch.randint(0, 8, (3, 4), dtype=torch.int32),
    }

    parity.save_golden("roundtrip", tensors, root=tmp_path)
    loaded = parity.load_golden("roundtrip", root=tmp_path)

    assert set(loaded) == set(tensors)
    for name, want in tensors.items():
        assert loaded[name].dtype == want.dtype, name
        assert torch.equal(loaded[name], want), name


# ------------------------------------------------------------------- hadamard
#
# `Indexer.forward` (model.py:420) and its `Compressor` (model.py:375) call
# `rotate_activation`, which imports `fast_hadamard_transform` -- a compiled
# CUDA extension outside the step-1 kernel gate. Vaayu may not require a
# per-model extension (CLAUDE.md §5), so the harness carries an exact torch
# fallback and these tests keep the two honest.


def test_the_torch_hadamard_is_orthonormal_at_the_dim_the_indexer_uses():
    """`rotate_activation` is called on `index_head_dim`-wide vectors with
    `scale = n ** -0.5`, which makes the transform orthonormal: H @ H.T == I.
    A wrong scale or a transposed Sylvester order breaks this identity.
    """
    import torch

    n = 128
    identity = torch.eye(n, dtype=torch.float32)

    h = parity.torch_hadamard(identity, scale=n ** -0.5)

    assert torch.allclose(h @ h.T, identity, atol=1e-5)


@pytest.mark.skipif(not parity.have_hadamard_extension(),
                    reason="fast_hadamard_transform not installed")
def test_the_torch_fallback_agrees_with_the_extension():
    """If these disagree, every Indexer golden recorded on a box without the
    extension is silently a different number from one recorded with it.
    """
    import torch

    n = 128
    x = torch.randn(4, n, dtype=torch.bfloat16, device="cuda")

    mine = parity.torch_hadamard(x, scale=n ** -0.5)
    theirs = parity.hadamard(x, scale=n ** -0.5)

    rel = (mine.float() - theirs.float()).abs().max() / theirs.float().abs().max()
    assert rel < 2e-2, f"torch fallback differs from the extension by {rel:.4g}"


# ------------------------------------------------ where a checkpoint key lives
#
# Two layouts are real: a fresh download holds only the ORIGINAL shards the
# index names; a checkpoint the server has loaded (`delete_original=True`, its
# default) holds only `splitted_model/`, every key renamed by
# `remap_checkpoint_key`. This box is the second, so the first is proven here.

def _fake_checkpoint(root: Path, *, originals: bool, split: bool) -> Path:
    """Three keys with three different remaps: `layers.*` gains `model.`,
    `head.weight` becomes `lm_head.weight`, `hc_head_*` lands in the norm shard."""
    import torch
    from safetensors.torch import save_file

    tensors = {"layers.0.attn.wq_a.weight": torch.full((2, 2), 1.0),
               "head.weight": torch.full((3, 2), 2.0),
               "hc_head_fn": torch.full((4,), 3.0)}
    shard = "model-00001-of-00001.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}}))
    if originals:
        save_file(tensors, str(root / shard))
    if split:
        (root / "splitted_model").mkdir()
        for name, key, stored in (("model.layers.0", "layers.0.attn.wq_a.weight",
                                   "model.layers.0.attn.wq_a.weight"),
                                  ("lm_head", "head.weight", "lm_head.weight"),
                                  ("model.norm", "hc_head_fn", "model.norm.hc_head_fn")):
            save_file({stored: tensors[key]},
                      str(root / "splitted_model" / f"{name}.safetensors"))
    return root


@pytest.mark.parametrize("originals, split", [(True, False), (False, True)],
                         ids=["fresh-download", "split-only"])
def test_every_key_reads_back_by_its_checkpoint_name_in_either_layout(
        tmp_path, originals, split):
    import torch

    ckpt = _fake_checkpoint(tmp_path, originals=originals, split=split)

    got = parity.read_checkpoint_tensors(
        ["layers.0.attn.wq_a.weight", "head.weight", "hc_head_fn"], ckpt=ckpt)

    assert torch.equal(got["layers.0.attn.wq_a.weight"], torch.full((2, 2), 1.0))
    assert torch.equal(got["head.weight"], torch.full((3, 2), 2.0))
    assert torch.equal(got["hc_head_fn"], torch.full((4,), 3.0))
    assert parity.load_module_weights("layers.0.attn.", ckpt=ckpt).keys() == {"wq_a.weight"}


def test_a_key_in_neither_layout_names_itself(tmp_path):
    """A missing weight must say which one -- not a bare KeyError on a split
    name the caller never asked for."""
    ckpt = _fake_checkpoint(tmp_path, originals=False, split=False)

    with pytest.raises(FileNotFoundError, match="head.weight"):
        parity.read_checkpoint_tensors(["head.weight"], ckpt=ckpt)


def test_importing_the_reference_writes_nothing_into_the_checkpoint(tmp_path, monkeypatch):
    """`reference_modules` imports the checkpoint's own `inference/model.py`. A
    plain `exec_module` caches it as `inference/__pycache__/model.*.pyc` -- a
    write into the user's download on every suite run (S5-W). The stubs carry
    only what the harness touches: the attention tile it refits, and a
    `model.py` that imports `kernel`.
    """
    import sys
    import tempfile

    inference = tmp_path / "inference"
    inference.mkdir()
    variant = tmp_path / "variant"
    variant.mkdir()
    # The harness never removes its patched-kernel tempdir; keep this one in tmp_path.
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *args, **kwargs: str(variant))
    (inference / "kernel.py").write_text(
        "def sparse_attn_kernel():\n    num_stages = 2\n    block = 64\n\n\n"
        "def hc_split_sinkhorn_kernel():\n    pass\n")
    (inference / "model.py").write_text("from kernel import sparse_attn_kernel\n")
    monkeypatch.setattr(parity, "_reference", None)
    # The harness registers the module under this name; put back what was there.
    monkeypatch.setitem(sys.modules, "a5_reference_model", None)
    # Bytecode caching as a default interpreter does it, whatever the runner sets.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(sys, "pycache_prefix", None)

    module = parity.reference_modules(tmp_path)

    assert Path(module.__file__) == inference / "model.py"
    assert sorted(p.name for p in inference.iterdir()) == ["kernel.py", "model.py"]
    assert sys.dont_write_bytecode is False


# ------------------------------------------------------- reading real weights
#
# These need the 157 G checkpoint. Everything above does not.

needs_checkpoint = pytest.mark.skipif(
    not parity.have_checkpoint(), reason=f"checkpoint not at {parity.CKPT}")


@needs_checkpoint
def test_module_weights_come_back_keyed_relative_to_the_prefix():
    """`load_state_dict` wants the module's own names, and reading a whole
    shard for one 130 MiB module would defeat the point.
    """
    weights = parity.load_module_weights("layers.7.attn.")

    assert "wq_a.weight" in weights
    assert not any(key.startswith("layers.") for key in weights)


@needs_checkpoint
def test_wo_a_is_dequantised_at_load_because_the_module_declares_bf16():
    """`Attention.__init__` builds `wo_a` with `dtype=torch.bfloat16`, but the
    checkpoint ships it F8_E4M3 with a separate scale. `convert.py` multiplies
    the two out at conversion time. Skip that and `load_state_dict` copies raw
    FP8 values into a bf16 Parameter WITHOUT the scale -- silently wrong
    numbers, no exception, and the scale key left over as unexpected.
    """
    import torch

    weights = parity.load_module_weights("layers.7.attn.")

    assert weights["wo_a.weight"].dtype == torch.bfloat16
    assert "wo_a.scale" not in weights
    # Every other quantized projection keeps its scale as a separate tensor.
    assert weights["wq_a.weight"].dtype == torch.float8_e4m3fn
    assert "wq_a.scale" in weights


@needs_checkpoint
def test_routed_expert_weights_are_reinterpreted_as_packed_fp4():
    """The shards declare I8; the reference's `Linear` declares
    `float4_e2m1fn_x2` and `linear()` dispatches on exactly that dtype. It is a
    bit reinterpretation, not a conversion -- the byte count must not change.
    """
    import torch

    weights = parity.load_module_weights("layers.7.ffn.experts.0.")
    w1 = weights["w1.weight"]

    assert w1.dtype == torch.float4_e2m1fn_x2
    assert w1.shape == (2048, 2048)          # [out, in // 2], derived on disk
    assert weights["w1.scale"].dtype == torch.float8_e8m0fnu


# --------------------------------------------------- running reference modules

needs_gpu = pytest.mark.skipif(not parity.have_cuda(), reason="no CUDA device")


@needs_checkpoint
def test_reference_globals_are_set_or_every_quantised_weight_loads_wrong():
    """`linear()` reads `scale_fmt`/`scale_dtype` and `Linear.__init__` reads
    `default_dtype` -- all three MODULE GLOBALS, set only inside
    `Transformer.__init__` (model.py:881-886), which the harness never runs.

    Leave them and `default_dtype` stays bf16, so `shared_experts.w1` allocates
    bf16 and the F8_E4M3 disk tensor loads cast WITHOUT its scale. No
    exception, just wrong numbers -- exactly the class of bug this harness
    exists to catch, so it is pinned before anything is recorded.
    """
    import torch

    reference = parity.reference_modules()
    parity.reference_args(reference, DeepseekV4Config(**json.loads(SHIPPED.read_text())))

    assert reference.default_dtype == torch.float8_e4m3fn
    assert reference.scale_fmt == "ue8m0"
    assert reference.scale_dtype == torch.float8_e8m0fnu


@needs_checkpoint
@needs_gpu
def test_one_reference_attention_layer_runs_here_with_real_weights(config):
    """A5.2's whole premise: the model cannot run, but a MODULE can. If this
    fails there is no parity harness and A5.3-A5.7 have no reference.

    Layer 3 is a `compress_ratio == 128` layer -- a `Compressor`, no `Indexer`.
    """
    import torch

    reference = parity.reference_modules()
    args = parity.reference_args(reference, config)

    with parity.reference_device(reference):
        layer = parity.build_attention(reference, args, layer_id=3)
        x = parity.seeded_input(1, 16, args.dim)
        out = layer(x, 0)

    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()


@needs_checkpoint
@needs_gpu
@pytest.mark.parametrize("layer_id, routing", [(0, "hash"), (7, "score")])
def test_one_reference_moe_layer_runs_here_with_real_weights(config, layer_id, routing):
    """The other half of the premise, and 94.3 % of the bytes. Both routing
    paths, because `Gate` branches on `layer_id < n_hash_layers`: layers 0-2
    look the experts up in `tid2eid[input_ids]`, the rest score them with
    `sqrtsoftplus` + bias. Scoring a MoE with the wrong function is what made
    MiniMax emit incoherent text while loading perfectly.
    """
    import torch

    reference = parity.reference_modules()
    args = parity.reference_args(reference, config)
    assert (layer_id < args.n_hash_layers) == (routing == "hash")

    with parity.reference_device(reference):
        ffn = parity.build_moe(reference, args, layer_id=layer_id)
        x = parity.seeded_input(1, 4, args.dim)
        input_ids = parity.seeded_token_ids(1, 4, vocab_size=args.vocab_size)
        out = ffn(x, input_ids)

    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()


@needs_checkpoint
@needs_gpu
def test_the_reference_cannot_do_chunked_prefill(config):
    """`model.py:535` does `kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`,
    which only makes sense for `seqlen == 1`. Vaayu chunks prefill at 512, and
    row 43 already cost us a silent context loss of exactly this shape.

    So: does the reference RAISE on a multi-token step past position 0, or does
    it quietly write the wrong thing? Raising is the good outcome -- it means
    A5.5 cannot inherit the bug by accident. If this test fails, the answer was
    "silently", and that is the more dangerous finding.
    """
    reference = parity.reference_modules()
    args = parity.reference_args(reference, config)
    chunk = 512

    with parity.reference_device(reference):
        layer = parity.build_attention(reference, args, layer_id=3)
        prefill = parity.seeded_input(1, 64, args.dim)
        layer(prefill, 0)

        with pytest.raises(RuntimeError):
            layer(parity.seeded_input(1, chunk, args.dim, seed=1), 64)


# ------------------------------------------------------------- the golden set

EXPECTED_GOLDENS = {
    "linear_fp4", "linear_fp8", "linear_bf16",       # A5.3, both weight formats
    "expert", "gate_hash", "gate_score",             # A5.3 / A5.4
    "moe_hash", "moe_score",                         # A5.4
    "attn_prefill_r128", "attn_decode_r128", "attn_chunked_r128",   # A5.5
    "attn_prefill_r4", "attn_decode_r4", "attn_chunked_r4",         # A5.5
    "block", "hc_head", "head",                       # A5.6
}


def test_the_golden_set_covers_every_module_the_later_sub_plans_port():
    """A5.3-A5.7 are TDD-able only for what is recorded here. Attention is
    recorded on BOTH compress geometries -- 21 of the 43 layers carry an
    `Indexer` (`compress_ratio == 4`) and 20 do not (`== 128`), and they are
    different code paths, not different weights.
    """
    assert set(parity.RECIPES) == EXPECTED_GOLDENS


@needs_checkpoint
@needs_gpu
def test_a_golden_regenerates_bit_identically(tmp_path):
    """Success criterion 2. A golden that moves between runs is not a
    reference, and every parity test built on it would be flaky rather than
    wrong-detecting.
    """
    import torch

    first = parity.record_golden("expert", root=tmp_path)
    second = parity.record_golden("expert", root=tmp_path)

    assert set(first) == set(second)
    for key in first:
        assert torch.equal(first[key], second[key]), key


def test_every_committed_golden_loads_back_with_its_shapes_and_dtypes():
    """Success criterion 3, the cheap half: the fixtures are what A5.3-A5.7
    actually consume, so they must be readable on their own.
    """
    import torch

    for name in EXPECTED_GOLDENS:
        tensors = parity.load_golden(name)
        assert tensors, name
        for key, tensor in tensors.items():
            assert torch.isfinite(tensor.float()).all() or tensor.dtype == torch.int64, \
                f"{name}/{key}"


@pytest.mark.parametrize("name", ["attn_chunked_r128", "attn_chunked_r4"])
def test_the_chunked_goldens_record_that_the_reference_refuses_the_step(name):
    """The most valuable thing A5.2 buys. `model.py:535` assumes `seqlen == 1`
    on the ring-buffer branch, Vaayu chunks prefill at 512, and this is the
    recorded proof that the reference has no answer for it -- so A5.5 must
    write one rather than port one.
    """
    assert parity.load_golden(name).keys() == {"prefill_x"}
    assert "raised RuntimeError" in parity.load_note(name)


def test_the_suite_passes_with_the_checkpoint_absent():
    """Success criteria 3 and 4 together: point `A5_CKPT` at nothing and every
    checkpoint-dependent test must SKIP, not fail, while the fixture-backed
    ones still run. This is what makes A5.3-A5.7 developable without 157 G.
    """
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider",
         "-k", "not with_the_checkpoint_absent"],
        env={**os.environ, "A5_CKPT": "/nonexistent-checkpoint"},
        capture_output=True, text=True, cwd=Path(__file__).parent.parent)

    assert result.returncode == 0, result.stdout[-3000:]
    assert " skipped" in result.stdout


@needs_checkpoint
def test_the_reference_rotation_is_routed_through_the_harness():
    """`rotate_activation` (model.py:253) imports `fast_hadamard_transform`
    directly, so the fallback is worth nothing unless the reference itself goes
    through it: without this, a box with no nvcc cannot record the `r4` goldens
    at all, extension-free harness or not. With the extension installed this
    changes no number -- `hadamard()` calls it.
    """
    reference = parity.reference_modules()

    assert reference.rotate_activation is parity.rotate_activation
