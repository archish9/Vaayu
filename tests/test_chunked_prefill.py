import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from unittest.mock import patch, MagicMock

from tests.test_vaayu_base_kwargs import _make_base


def test_prefill_chunk_size_default_is_512():
    m = _make_base()
    assert m.prefill_chunk_size == 512


def test_prefill_chunk_size_custom_stored():
    m = _make_base({"prefill_chunk_size": 256})
    assert m.prefill_chunk_size == 256


def test_prefill_chunk_size_zero_stored():
    m = _make_base({"prefill_chunk_size": 0})
    assert m.prefill_chunk_size == 0
