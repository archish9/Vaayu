"""Lookahead routing predictor: accuracy accounting (Phase A, measure-only)."""
from vaayu.moe_patch import lookahead_accuracy_update


def test_accuracy_update_counts_overlap():
    state = {"hits": 0, "total": 0}
    # prediction for layer 5 was {1,2,3,4}; actual routing at layer 5 is {2,3,9,10}
    lookahead_accuracy_update(state, predicted=(5, {1, 2, 3, 4}),
                              layer_idx=5, actual={2, 3, 9, 10})
    assert state == {"hits": 2, "total": 4}


def test_accuracy_update_ignores_layer_mismatch():
    state = {"hits": 0, "total": 0}
    lookahead_accuracy_update(state, predicted=(5, {1, 2}), layer_idx=6, actual={1, 2})
    assert state == {"hits": 0, "total": 0}


def test_accuracy_update_ignores_none():
    state = {"hits": 0, "total": 0}
    lookahead_accuracy_update(state, predicted=None, layer_idx=3, actual={1})
    assert state == {"hits": 0, "total": 0}
