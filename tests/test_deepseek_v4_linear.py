"""A5.3: dequant + `Linear` for DeepSeek-V4-Flash.

Two weight formats live in this checkpoint and `config.json` names only one:
routed experts are fp4 packed 2/byte with a per-row/32 E8M0 scale, shared
experts and attention are F8_E4M3 with a per-128x128-block E8M0 scale. The
scale SHAPES differ, not just the dtypes.

The tests that reproduce A5.2's goldens need the 157 G checkpoint (for the
weights -- the goldens carry inputs and outputs, not weights) and a GPU (the
GEMMs are TileLang CUDA kernels). Everything else here runs off synthetic
tensors on the CPU, including the two regressions that would otherwise only
show up on a streamed expert at token 1.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v4_config.json"

needs_checkpoint = pytest.mark.skipif(
    not parity.have_checkpoint(), reason=f"checkpoint not at {parity.CKPT}")
needs_gpu = pytest.mark.skipif(not parity.have_cuda(), reason="no CUDA device")


@pytest.fixture
def config() -> DeepseekV4Config:
    return DeepseekV4Config(**json.loads(FIXTURE.read_text()))


@pytest.fixture
def block_size(config) -> list:
    return config.checkpoint_quantization_config["weight_block_size"]


# ── the stub kernels ────────────────────────────────────────────────────────
#
# `kernel.py` is DeepSeek's, it needs tilelang and a GPU, and these tests are
# about what Vaayu's `Linear` HANDS it. A stub on disk (the loader imports a
# file, so it has to be one) records that, on a box with neither.

STUB_KERNELS = '''
import torch

CALLS = []


def act_quant(x, block_size, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    CALLS.append(dict(fn="act_quant", x=x, block_size=block_size,
                      scale_fmt=scale_fmt, scale_dtype=scale_dtype,
                      default_dtype=torch.get_default_dtype()))
    scale = torch.zeros(*x.shape[:-1], x.shape[-1] // block_size, dtype=scale_dtype)
    return torch.zeros_like(x, dtype=torch.float8_e4m3fn), scale


def _gemm(name, a, a_s, b, b_s, scale_dtype):
    CALLS.append(dict(fn=name, a=a, a_s=a_s, b=b, b_s=b_s, scale_dtype=scale_dtype,
                      default_dtype=torch.get_default_dtype()))
    return torch.zeros(*a.shape[:-1], b.shape[0], dtype=torch.get_default_dtype())


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    return _gemm("fp8_gemm", a, a_s, b, b_s, scale_dtype)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    return _gemm("fp4_gemm", a, a_s, b, b_s, scale_dtype)
'''

UNIMPORTABLE_KERNELS = '''
raise ImportError("No module named 'tilelang'")
'''


@pytest.fixture
def kernel_dir(tmp_path, monkeypatch):
    """Point the lazy kernel loader at a directory this test writes."""
    def install(source: str) -> Path:
        (tmp_path / "kernel.py").write_text(source)
        monkeypatch.setattr(modeling, "KERNEL_DIR", tmp_path)
        monkeypatch.setattr(modeling, "_KERNELS", None)
        return tmp_path

    return install


# ── the lazy kernel import ──────────────────────────────────────────────────


def test_importing_the_modeling_module_does_not_import_tilelang():
    """`import kernel` pulls in tilelang, which pulls in TVM. The config path
    (`AutoConfig` on a box with no GPU, and any CI) imports this module's
    package, so a module-level kernel import would make tilelang a hard
    requirement of reading a config -- CLAUDE.md Section 5.
    """
    probe = ("import sys;"
             "import vaayu.model_code.deepseek_v4.modeling_deepseek_v4;"
             "print('tilelang' in sys.modules)")

    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=Path(__file__).parents[1],
                         env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])})

    assert out.returncode == 0, out.stderr[-2000:]
    # `import vaayu` prints a banner, so the answer is the last line.
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_the_lazy_kernel_import_writes_nothing_beside_the_checkpoints_kernel_py(
        kernel_dir, monkeypatch):
    """When the attention tile needs no refit (a GPU with the shared memory for
    DeepSeek's own), `kernels()` imports `inference/kernel.py` where it lies --
    inside the user's download. A plain `exec_module` caches bytecode as
    `inference/__pycache__/*.pyc` beside it (S5-W). A stub has no attention tile
    to fit, so it takes that branch on any box.
    """
    # Bytecode caching as a default interpreter does it, whatever the runner sets.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(sys, "pycache_prefix", None)
    directory = kernel_dir(STUB_KERNELS)

    module = modeling.kernels()

    assert Path(module.__file__) == directory / "kernel.py"
    assert sorted(p.name for p in directory.rglob("*")) == ["kernel.py"]
    assert sys.dont_write_bytecode is False


def test_a_quantized_forward_without_the_kernels_names_both_pins(kernel_dir, block_size):
    """tilelang 0.1.8 declares `apache-tvm-ffi>=0.1.2,~=0.1.0`, so pip resolves
    a version that imports fine and dies at kernel COMPILE. A bare
    `ModuleNotFoundError: tilelang` sends the next session to `pip install
    tilelang` and straight into that trap, so the error names both pins.
    """
    kernel_dir(UNIMPORTABLE_KERNELS)
    layer = modeling.DeepseekV4Linear(8, 4, dtype=torch.float8_e4m3fn,
                                      weight_block_size=block_size)

    with pytest.raises(ImportError) as raised:
        layer(torch.zeros(2, 8, dtype=torch.bfloat16))

    assert "tilelang==0.1.8" in str(raised.value)
    assert "apache-tvm-ffi==0.1.9" in str(raised.value)


def test_a_bf16_weight_never_reaches_the_kernels(kernel_dir):
    """`wo_a` is dequantised at load and the module declares bf16, so its
    forward is `F.linear`. Loading the kernels for it would make every bf16
    projection need tilelang.
    """
    kernel_dir(UNIMPORTABLE_KERNELS)
    layer = modeling.DeepseekV4Linear(8, 4, dtype=torch.bfloat16)
    nn.init.normal_(layer.weight)
    x = torch.randn(2, 8, dtype=torch.bfloat16)

    out = layer(x)

    assert layer.scale is None
    assert torch.equal(out, torch.nn.functional.linear(x, layer.weight))


# ── the scale shapes, derived ───────────────────────────────────────────────


@pytest.mark.parametrize("in_features, out_features, group", [(4096, 2048, 32),
                                                              (512, 128, 16)])
def test_the_fp4_scale_is_shaped_by_the_group_size_read_off_the_checkpoint(
        in_features, out_features, group):
    """The fp4 group is in NO config field -- it is only knowable from the
    shipped scale shape, which is exactly why it is a constructor argument and
    not a literal. Second geometry per CLAUDE.md Section 5.
    """
    layer = modeling.DeepseekV4Linear(in_features, out_features,
                                      dtype=torch.float4_e2m1fn_x2,
                                      fp4_group_size=group)

    assert layer.weight.shape == (out_features, in_features // 2)
    assert layer.weight.dtype == torch.float4_e2m1fn_x2
    assert layer.scale.shape == (out_features, in_features // group)
    assert layer.scale.dtype == torch.float8_e8m0fnu


def test_an_fp4_linear_refuses_to_guess_the_group_size():
    """Defaulting it to the one this checkpoint happens to use is the exact
    hardcoded-model-number CLAUDE.md Section 5 forbids, and the failure it
    causes is a shape mismatch four sub-plans downstream.
    """
    with pytest.raises(ValueError, match="fp4_group_size"):
        modeling.DeepseekV4Linear(4096, 2048, dtype=torch.float4_e2m1fn_x2)


@pytest.mark.parametrize("block", [[128, 128], [64, 32]])
def test_the_fp8_scale_is_shaped_by_the_weight_block_size(block):
    """`fp8_gemm_kernel` declares `scales_b` as
    `[ceildiv(N, 128), ceildiv(K, 128)]`; the fp4 kernel declares `[N,
    ceildiv(K, 32)]`. Two different layouts -- reading the block size off
    `quantization_config.weight_block_size` is what keeps them apart.
    """
    layer = modeling.DeepseekV4Linear(200, 100, dtype=torch.float8_e4m3fn,
                                      weight_block_size=block)

    assert layer.weight.shape == (100, 200)
    assert layer.scale.shape == (-(-100 // block[0]), -(-200 // block[1]))
    assert layer.scale.dtype == torch.float8_e8m0fnu


# ── the mandatory divergence: `self.scale`, never `self.weight.scale` ───────


def test_forward_reads_self_scale_so_a_streamed_expert_still_works(kernel_dir, block_size):
    """THE reason this module exists rather than the reference's `Linear`.

    The reference aliases the scale onto the Parameter (`model.py:145`) and
    `linear()` reads `weight.scale`. `_inject_expert_weights`
    (`moe_patch.py:1225`) rebinds every streamed key with `setattr(submod,
    base, nn.Parameter(t, requires_grad=False))`, so `weight` becomes a FRESH
    object with no `.scale` -- `AttributeError` on every expert, every token.

    Rebinding both keys the way the engine does is what this test asserts
    survives; revert the forward to `self.weight.scale` and it raises.
    """
    kernel_dir(STUB_KERNELS)
    layer = modeling.DeepseekV4Linear(8, 4, dtype=torch.float8_e4m3fn,
                                      weight_block_size=block_size)

    for base in ("weight", "scale"):                 # _inject_expert_weights
        tensor = getattr(layer, base).data.clone()
        setattr(layer, base, nn.Parameter(tensor, requires_grad=False))
    assert not hasattr(layer.weight, "scale")

    out = layer(torch.zeros(2, 8, dtype=torch.bfloat16))

    assert out.shape == (2, 4)
    gemm = [call for call in modeling.kernels().CALLS if call["fn"] == "fp8_gemm"]
    assert len(gemm) == 1
    assert gemm[0]["b_s"] is layer.scale


def test_the_kernels_are_called_with_the_scale_dtype_read_off_the_scale(kernel_dir,
                                                                        block_size):
    """`act_quant` rounds its scales to powers of two only when `scale_fmt` is
    set, and both GEMMs pick their scale dtype from the argument. The reference
    takes all of it from MODULE GLOBALS set in `Transformer.__init__`, which a
    streamed module never runs -- A5.2 finding (1), the one that fails
    silently. Read it off the scale tensor instead.
    """
    kernel_dir(STUB_KERNELS)
    layer = modeling.DeepseekV4Linear(8, 4, dtype=torch.float8_e4m3fn,
                                      weight_block_size=block_size)

    layer(torch.zeros(2, 8, dtype=torch.bfloat16))

    quant, gemm = modeling.kernels().CALLS
    assert quant["block_size"] == block_size[1]
    assert quant["scale_fmt"] == "ue8m0"
    assert quant["scale_dtype"] == torch.float8_e8m0fnu
    assert gemm["scale_dtype"] == torch.float8_e8m0fnu


def test_the_gemm_runs_under_the_activation_dtype_not_the_process_default(kernel_dir,
                                                                          block_size):
    """`fp8_gemm` allocates its output with `torch.get_default_dtype()`
    (`kernel.py:268`) while the kernel is compiled to write bf16. The reference
    gets away with it because `generate.py:77` sets the process default to
    bf16; Vaayu's is float32, and nothing in an engine may depend on that
    global.
    """
    kernel_dir(STUB_KERNELS)
    layer = modeling.DeepseekV4Linear(8, 4, dtype=torch.float8_e4m3fn,
                                      weight_block_size=block_size)

    assert torch.get_default_dtype() == torch.float32
    out = layer(torch.zeros(2, 8, dtype=torch.bfloat16))

    assert modeling.kernels().CALLS[-1]["default_dtype"] == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert torch.get_default_dtype() == torch.float32       # restored


# ── the load-time dequant ───────────────────────────────────────────────────


@pytest.mark.parametrize("shape, block", [((256, 512), (128, 128)),
                                          ((64, 128), (32, 64))])
def test_dequantize_block_scaled_expands_each_block_scale_over_its_tile(shape, block):
    """Two geometries, because the block shape is read from the scale tensor:
    `scale[i, j]` covers `weight[i*bo:(i+1)*bo, j*bi:(j+1)*bi]`. Transpose the
    two and every number is wrong by a factor that looks plausible.
    """
    out_features, in_features = shape
    weight = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
    scale = torch.randn(out_features // block[0],
                        in_features // block[1]).exp().to(torch.float8_e8m0fnu)

    got = modeling.dequantize_block_scaled(weight, scale)

    expanded = (scale.float().repeat_interleave(block[0], 0)
                .repeat_interleave(block[1], 1))
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, (weight.float() * expanded).bfloat16())


def test_the_load_time_dequant_rule_consumes_wo_a_and_leaves_the_rest_alone():
    """`Attention.__init__` builds `wo_a` with `dtype=torch.bfloat16` and
    `Attention.forward` uses the RAW weight in an einsum, bypassing
    `Linear.forward`. Load it as it ships and `load_state_dict` casts F8_E4M3
    into bf16 WITHOUT the scale -- no exception, wrong numbers -- and leaves
    `wo_a.scale` over as an unexpected key.
    """
    weights = {
        "wo_a.weight": torch.randn(64, 128).to(torch.float8_e4m3fn),
        "wo_a.scale": torch.randn(1, 1).exp().to(torch.float8_e8m0fnu),
        "wq_a.weight": torch.randn(64, 128).to(torch.float8_e4m3fn),
        "wq_a.scale": torch.randn(1, 1).exp().to(torch.float8_e8m0fnu),
    }

    out = modeling.dequantize_at_load(weights)

    assert out["wo_a.weight"].dtype == torch.bfloat16
    assert "wo_a.scale" not in out
    assert out["wq_a.weight"] is weights["wq_a.weight"]
    assert out["wq_a.scale"] is weights["wq_a.scale"]


def _raw_module_weights(prefix: str) -> dict:
    """The tensors under `prefix` exactly as the shards hold them -- no value
    transforms, which is the point: the production rule applies them.
    """
    return parity.raw_module_weights(prefix)


@needs_checkpoint
def test_the_production_dequant_reproduces_the_harness_wo_a_bit_for_bit():
    """A5.2's harness does this transform too, in its own loader. Two copies
    drift unless something says so -- and a drift here is silently wrong
    attention output, not a crash. Bit-identical or the goldens stop meaning
    anything.
    """
    index = json.loads((parity.CKPT / "model.safetensors.index.json").read_text())
    prefix = next(key for key in index["weight_map"] if key.endswith("attn.wo_a.weight"))
    prefix = prefix[:-len("wo_a.weight")]
    raw = _raw_module_weights(prefix)

    mine = modeling.dequantize_at_load(raw)["wo_a.weight"]
    theirs = parity.load_module_weights(prefix)["wo_a.weight"]

    assert torch.equal(mine, theirs)


# ── the goldens ─────────────────────────────────────────────────────────────


@pytest.fixture
def checkpoint_kernels(monkeypatch):
    """The TileLang kernels ship inside the checkpoint. A copy of this module
    sitting next to the shards finds them beside itself; the repo copy has to
    be told, which is what A5.7's loader will do.
    """
    monkeypatch.setattr(modeling, "KERNEL_DIR", parity.CKPT / "inference")
    monkeypatch.setattr(modeling, "_KERNELS", None)


def _linear_from(weights: dict, in_features: int, out_features: int, block_size: list):
    """Build the module the way a loaded checkpoint would, straight off the
    tensors: the dtype, and for fp4 the group size, are read from what the
    shards actually hold.
    """
    weight, scale = weights["weight"], weights.get("scale")
    group = None
    if weight.dtype == torch.float4_e2m1fn_x2:
        group = weight.shape[-1] * 2 // scale.shape[-1]
    with torch.device("cuda"):
        layer = modeling.DeepseekV4Linear(in_features, out_features, dtype=weight.dtype,
                                          weight_block_size=block_size,
                                          fp4_group_size=group)
    layer.load_state_dict({key: value.cuda() for key, value in weights.items()})
    return layer.eval()


def _expert_weights(prefix: str, projection: str = "w1") -> dict:
    loaded = parity.load_module_weights(prefix)
    return {key.split(".", 1)[1]: value for key, value in loaded.items()
            if key.startswith(projection + ".")}


@needs_checkpoint
@needs_gpu
def test_the_fp4_golden_reproduces_through_a_rebound_streamed_weight(
        config, block_size, checkpoint_kernels):
    """Success criteria 1 and 2 in one run, on the format that is 94.3 % of the
    checkpoint: the same weights the reference's `Expert.w1` ran, rebound the
    way `_inject_expert_weights` rebinds a streamed expert, against the golden
    the reference produced.
    """
    golden = parity.load_golden("linear_fp4")
    layer_id = config.compress_ratios.index(128)
    weights = _expert_weights(f"layers.{layer_id}.ffn.experts.0.")

    layer = _linear_from(weights, config.hidden_size, config.moe_intermediate_size,
                         block_size)
    for base in ("weight", "scale"):                 # _inject_expert_weights
        setattr(layer, base, nn.Parameter(getattr(layer, base).data, requires_grad=False))

    with torch.inference_mode():
        out = layer(golden["x"].cuda())

    assert out.dtype == golden["out"].dtype
    assert torch.equal(out.cpu(), golden["out"])


@needs_checkpoint
@needs_gpu
def test_the_fp8_golden_reproduces(config, block_size, checkpoint_kernels):
    """The other format -- same kernel family, a scale of a different SHAPE.
    Feeding an fp8 weight the fp4 scale layout raises rather than drifting,
    which is how the step-1 gate found it.
    """
    golden = parity.load_golden("linear_fp8")
    layer_id = config.compress_ratios.index(128)
    weights = _expert_weights(f"layers.{layer_id}.ffn.shared_experts.")

    layer = _linear_from(weights, config.hidden_size,
                         config.moe_intermediate_size * config.n_shared_experts,
                         block_size)

    with torch.inference_mode():
        out = layer(golden["x"].cuda())

    assert torch.equal(out.cpu(), golden["out"])


@needs_checkpoint
@needs_gpu
def test_the_bf16_golden_reproduces_through_the_load_time_dequant(config, block_size):
    """`wo_a` end to end: F8_E4M3 plus a block scale on disk, dequantised once
    at load, then a plain `F.linear`. If the dequant is wrong this is where it
    shows as numbers rather than as a dtype.
    """
    golden = parity.load_golden("linear_bf16")
    layer_id = config.compress_ratios.index(128)
    attn = modeling.dequantize_at_load(_raw_module_weights(f"layers.{layer_id}.attn."))
    weights = {"weight": attn["wo_a.weight"]}

    layer = _linear_from(weights, golden["x"].shape[-1], golden["out"].shape[-1],
                         block_size)

    with torch.inference_mode():
        out = layer(golden["x"].cuda())

    assert torch.equal(out.cpu(), golden["out"])


# ── the checkpoint's copy ───────────────────────────────────────────────────


@needs_checkpoint
def test_the_checkpoints_copy_of_this_file_has_not_drifted():
    """Stale-copy guard -- see test_deepseek_v4_config.py's twin. No copy is
    needed since S5; one that exists must not have drifted."""
    copy = parity.CKPT / "modeling_deepseek_v4.py"
    if not copy.exists():
        pytest.skip("no checkpoint-side copy -- none is needed since S5 registers "
                    "DeepseekV4ForCausalLM in-process")
    repo = Path(__file__).parents[1] / "vaayu/model_code/deepseek_v4/modeling_deepseek_v4.py"

    assert copy.read_bytes() == repo.read_bytes()
