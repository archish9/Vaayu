import inspect
import torch
from vaayu.batch import build_batch_attention_mask, batch_position_ids
import vaayu.vaayu_base as vb

NEG = torch.finfo(torch.float32).min

def test_position_ids_left_pad():
    # stream 0: 1 pad + 3 real ; stream 1: 0 pad + 4 real
    pad = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    pos = batch_position_ids(pad)
    assert pos.tolist() == [[0, 0, 1, 2], [0, 1, 2, 3]]

def test_mask_blocks_pad_keys_and_future():
    pad = torch.tensor([[0, 1, 1]])          # KV=3, first col is pad
    m = build_batch_attention_mask(pad, q_len=3, dtype=torch.float32, device="cpu")
    assert m.shape == (1, 1, 3, 3)
    # query row 1 (KV index 1) may attend key 1 only (key 0 is pad, key 2 is future)
    row1 = m[0, 0, 1]
    assert row1[0] == NEG and row1[1] == 0.0 and row1[2] == NEG

def test_decode_mask_single_query():
    pad = torch.tensor([[0, 1, 1, 1]])       # KV=4
    m = build_batch_attention_mask(pad, q_len=1, dtype=torch.float32, device="cpu")
    assert m.shape == (1, 1, 1, 4)
    # last query attends all real keys, not the pad
    assert m[0, 0, 0].tolist() == [NEG, 0.0, 0.0, 0.0]

def test_forward_has_gated_caller_mask_branch():
    """Guard: forward must branch on a captured caller mask/position, not
    unconditionally null them. Prevents a regression that silently reverts
    left-pad support."""
    src = inspect.getsource(vb.VaayuBaseModel.forward)
    assert "_caller_attention_mask" in src
    assert "_caller_position_ids" in src
    assert "build_batch_attention_mask" in src
