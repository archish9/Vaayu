"""S16 (iii): `generate()` seeds Vaayu's OWN KV cache even when prefix caching is off.

Before the fix, `VaayuBaseModel.generate` seeded `past_key_values` only on the
prefix-cache path. With the prefix cache off (the library default), it called
transformers' `generate()` bare, which manufactured a stock `DynamicCache` that
`forward` then used as-is. Two consequences, both measured:
- `kv_cache_mode` was INERT for every model (A5.7b, 2026-09-25, Llama-3.2-1B on device);
- gpt-oss-20b CRASHED once a prompt crossed its 128-token sliding window: HF's cache
  truncates the sliding layers while Vaayu's eager mask is full length
  ("The size of tensor a (128) must match the size of tensor b (201)", reproduced
  2026-10-01 at 7483205).
"""
import contextlib

import pytest
import torch
from transformers.generation.utils import GenerationMixin

from vaayu.vaayu_base import VaayuBaseModel


class _Telemetry:
    def dump_json(self, *a, **k):
        pass


@pytest.fixture
def bare(monkeypatch):
    """A VaayuBaseModel with only what generate() touches on the prefix-off path,
    and HF's generate() replaced by a recorder."""
    seen = {}
    monkeypatch.setattr(GenerationMixin, "generate",
                        lambda self, *a, **kw: seen.update(kw) or "generated")
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m._prefix_cache = None
    m.telemetry = _Telemetry()
    m._keepawake_ctx = lambda: contextlib.nullcontext()
    m._assert_context_fits = lambda n_tokens: None
    made = []
    m._create_kv_cache = lambda: made.append(object()) or made[-1]
    return m, seen, made


def test_prefix_off_generate_seeds_vaayus_own_cache(bare):
    m, seen, made = bare
    out = m.generate(input_ids=torch.zeros(1, 4, dtype=torch.long), max_new_tokens=2)
    assert out == "generated"
    assert len(made) == 1, "generate() must build Vaayu's cache when the caller passed none"
    assert seen["past_key_values"] is made[0]


def test_a_caller_supplied_cache_wins(bare):
    m, seen, made = bare
    mine = object()
    m.generate(input_ids=torch.zeros(1, 4, dtype=torch.long), max_new_tokens=2,
               past_key_values=mine)
    assert made == [], "a caller-supplied cache must not be replaced"
    assert seen["past_key_values"] is mine
