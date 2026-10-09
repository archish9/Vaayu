"""S35: the side channels that read a LIVE cache's per-position K/V refuse a
windowed cache, each with one line that says why. No checkpoint."""
import types
from pathlib import Path

import pytest
import torch

from vaayu.vaayu_base import VaayuBaseModel

BASE_SRC = Path(__file__).resolve().parents[1] / "vaayu" / "vaayu_base.py"


def _stub(windows, enabled=True):
    stub = types.SimpleNamespace(kv_cache_mode="auto", loader_strategy=object(),
                                 _kv_windows=windows, prefill_chunk_size=8,
                                 enable_prefix_cache=enabled, config=None)
    stub._create_kv_cache = lambda: VaayuBaseModel._create_kv_cache(stub)
    return stub


@pytest.mark.parametrize("windows,enabled,expected", [
    ({0: 4}, True, True), ({}, True, False), ({0: 4}, False, False)],
    ids=["windowed", "no-window", "pool-off"])
def test_the_base_records_whether_its_cache_is_windowed(windows, enabled, expected):
    stub = _stub(windows, enabled)
    VaayuBaseModel._refuse_unsupported_prefix_cache(stub)
    assert stub._cache_is_windowed is expected
    assert stub._cache_is_one_length is False, "a windowed cache stays croppable"


def test_prefix_compress_is_refused_on_a_windowed_cache(capsys):
    stub = types.SimpleNamespace(prefix_compress=9, auto_prefix_match=True,
                                 _cache_holds_recurrent_state=False,
                                 _cache_is_one_length=False, _cache_is_windowed=True)
    assert VaayuBaseModel._build_prefix_store(stub, 10 ** 9).compress_bits is None
    out = capsys.readouterr().out
    assert "prefix_compress ignored" in out and "window length" in out


def test_kv_persist_refuses_a_windowed_cache_by_source():
    src = BASE_SRC.read_text()
    guard = src.find('and getattr(self, "_cache_is_windowed", False)):\n'
                     '            print("[KV] enable_kv_persist=True ignored')
    writer = src.find("self._kv_writer = DiskWriter(")
    assert guard != -1, "kv_persist has no windowed-cache guard"
    assert guard < writer


def test_batched_generate_bypasses_the_pool_on_a_windowed_cache_by_source():
    src = BASE_SRC.read_text()
    guard = src.find('elif pc is not None and getattr(self, "_cache_is_windowed", False):')
    lookup = src.find("common_ids = batch_common_prefix(requests) if pc is not None else []")
    assert guard != -1, "batched_generate has no windowed-cache bypass"
    assert lookup != -1 and guard < lookup


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_decode_graphs_refuse_a_windowed_load(capsys):
    from vaayu.decode_graphs import install_decode_graphs
    stub = types.SimpleNamespace(running_device=f"cuda:{torch.cuda.current_device()}",
                                 _kv_windows={0: 4})
    assert install_decode_graphs(stub) is None
    assert "window length" in capsys.readouterr().out
