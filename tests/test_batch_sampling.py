import torch
from vaayu.batch import sample_step

def test_greedy_is_argmax():
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])
    assert sample_step(logits, temperature=0.0, top_p=1.0) == 1

def test_greedy_deterministic_across_calls():
    logits = torch.randn(1000)
    a = sample_step(logits, 0.0, 1.0)
    b = sample_step(logits, 0.0, 1.0)
    assert a == b == int(logits.argmax())

def test_top_p_restricts_support():
    # one dominant token; top_p=0.5 must always pick it under sampling
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    torch.manual_seed(0)
    picks = {sample_step(logits, temperature=1.0, top_p=0.5) for _ in range(50)}
    assert picks == {0}

def test_temperature_sampling_stays_in_vocab():
    logits = torch.randn(32)
    torch.manual_seed(0)
    for _ in range(20):
        tok = sample_step(logits, temperature=0.8, top_p=0.9)
        assert 0 <= tok < 32
