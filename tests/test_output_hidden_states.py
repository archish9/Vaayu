"""`output_hidden_states=True` returns HF's hidden-states tuple (todo row 18).

On-device gate -- needs a checkpoint:

    HIDDEN_CKPT=/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct \
        ./venv/bin/python -m pytest tests/test_output_hidden_states.py -q

It crashed from 2026-05-25 (`6253fd1`) until row 18: the forward accumulated
hidden states two ways at once -- one sublist per layer, filled inside the
layer loop, AND a flat `+=` of a bare Tensor after every layer -- so the outer
list held 2 x len(self.layers) mixed elements and the final concat hit the
embedding layer's empty sublist:
`ValueError: torch.cat(): expected a non-empty list of Tensors`.

The convention asserted is transformers 5.x's (`utils/output_capturing.py`):
num_hidden_layers + 1 entries -- the embedding output, every decoder layer's
output but the last, then the FINAL-NORM output.

The count and shapes are checked on any checkpoint. The values are checked
against stock transformers only when stock transformers can load it (no
`quantization_config`), which is what catches an off-by-one layer or a
pre-norm last entry -- a count check alone cannot.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from vaayu import AutoModel

CKPT = os.environ.get("HIDDEN_CKPT")
pytestmark = pytest.mark.skipif(not CKPT, reason="needs HIDDEN_CKPT")

PROMPT = "The quick brown fox jumps over the lazy dog because"
# Mean per-token rel-L2 vs an fp32 reference. Measured on Llama-3.2-1B
# (2026-09-11): worst layer 0.018 (bf16 rounding), nearest NEIGHBOUR layer
# >= 0.459 -- so 0.05 passes the right layer and fails a shifted one.
TOL = 0.05


def _rel_l2(a, b):
    """Mean over positions 1.. of per-token relative L2. Position 0 is left
    out: the BOS token carries a massive activation (norm ~790 against ~5 for
    the rest on the 1B) that swamps a whole-tensor norm, which put adjacent
    layers only 0.009 apart."""
    a, b = a.float().cpu()[:, 1:], b.float().cpu()[:, 1:]
    return ((a - b).norm(dim=-1) / b.norm(dim=-1)).mean().item()


@pytest.fixture(scope="module")
def run():
    # One Vaayu load per process (todo.md, On-device gates).
    model = AutoModel.from_pretrained(CKPT, device="cuda", attn_implementation="sdpa")
    ids = model.tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    # Premise: a short prompt runs ONE forward. A chunked prefill returns only
    # the last chunk's hidden states (as it does its logits).
    assert ids.shape[1] <= model.prefill_chunk_size or model.prefill_chunk_size == 0
    out = model.forward(input_ids=ids, use_cache=False, return_dict=True,
                        output_hidden_states=True)
    return model, ids, out


def test_one_entry_per_decoder_layer_plus_embeddings(run):
    model, ids, out = run
    hs = out.hidden_states
    assert isinstance(hs, tuple)
    assert len(hs) == model.config.num_hidden_layers + 1
    for k, h in enumerate(hs):
        assert tuple(h.shape) == (1, ids.shape[1], model.config.hidden_size), k
        assert torch.isfinite(h).all(), k
    # No entry recorded twice (the old flat `+=` pushed each layer's output
    # in addition to its sublist).
    for k in range(len(hs) - 1):
        assert not torch.equal(hs[k], hs[k + 1]), k


def test_output_attentions_is_refused_by_name(run):
    model, ids, _ = run
    with pytest.raises(NotImplementedError, match="output_attentions"):
        model.forward(input_ids=ids, use_cache=False, return_dict=True,
                      output_attentions=True)


def test_values_match_stock_transformers(run):
    cfg = json.loads((Path(CKPT) / "config.json").read_text())
    if "quantization_config" in cfg:
        pytest.skip("stock transformers cannot load a quantized checkpoint")
    from transformers import AutoModelForCausalLM
    model, ids, out = run
    # CPU fp32 reference: the Vaayu load holds most of the GPU.
    ref_model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32)
    with torch.no_grad():
        ref = ref_model(input_ids=ids, output_hidden_states=True).hidden_states
    assert len(ref) == len(out.hidden_states)
    errs = [_rel_l2(h, r) for h, r in zip(out.hidden_states, ref)]
    # An off-by-one layer or a pre-norm last entry lands at the distance
    # between adjacent reference layers. Premise, asserted not assumed: TOL
    # sits below every such gap, or passing would prove nothing.
    gaps = [_rel_l2(ref[k + 1], ref[k]) for k in range(len(ref) - 1)]
    print(f"\nper-token rel-L2 vs HF fp32: max {max(errs):.4f}  "
          f"per-layer {[round(e, 4) for e in errs]}\nadjacent-layer gap: min {min(gaps):.4f}")
    assert TOL < min(gaps), gaps
    assert max(errs) < TOL, errs

