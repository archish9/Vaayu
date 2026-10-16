"""Regression: fused-EXL3 chunk-major prefill crash (found 2026-10-16).

Two prefill staging mechanisms shared `vaayu_ref._prefill_staged`:
  * the prefill LOOKAHEAD stager (`_prefill_stage_worker`, fused-EXL3 only) publishes
    INT-keyed layers: `_prefill_staged[layer_idx] = {eidx: views}`;
  * the per-expert slab tracker (`_stage_prefill_expert`) writes FLAT TUPLE keys
    `_prefill_staged[(layer_idx, eidx)] = slab_idx`, swept by `_release_prefill_staged`
    with `k[0]`.

On the fused-EXL3 prefill pass both are active, so `_release_prefill_staged`'s `k[0]`
sweep met the lookahead stager's INT keys -> `TypeError: 'int' object is not subscriptable`
(moe_patch `_release_prefill_staged`, via `_dispatch_expert_misses` at :2943 and the
forward finally at :3494). Latent since Qwen3-235B (the only fused-EXL3 model) was retired.

The fix gives the per-expert slab tracker its OWN namespace (`_prefill_slab_staged`), so the
`k[0]` sweep only ever touches tuple keys and never the lookahead stager's int-keyed dict.
"""
from types import SimpleNamespace

from vaayu.moe_patch import _slab_staged, _release_prefill_staged


def test_per_expert_slab_staging_is_isolated_from_the_lookahead_stager_dict():
    vaayu_ref = SimpleNamespace()
    # The lookahead stager (fused-EXL3) publishes INT-keyed layers here.
    vaayu_ref._prefill_staged = {0: {5: object()}, 1: {7: object()}}
    slab = _slab_staged(vaayu_ref)
    slab[(0, 5)] = 99
    # A per-expert slab tuple key must land in its own dict, NOT the lookahead dict.
    assert slab[(0, 5)] == 99
    assert (0, 5) not in vaayu_ref._prefill_staged
    # ...and the per-expert dict must not inherit the lookahead stager's int keys.
    assert 0 not in slab and 1 not in slab
    assert slab is not vaayu_ref._prefill_staged


def test_release_prefill_staged_does_not_choke_on_lookahead_int_keys():
    vaayu_ref = SimpleNamespace()
    # Lookahead stager int-keyed entries present at the same time.
    vaayu_ref._prefill_staged = {0: {5: object()}, 2: {9: object()}}
    released = []
    expert_cache = SimpleNamespace(
        arena=SimpleNamespace(release_staging=released.append))
    # Per-expert slab staged for layer 1 (what _stage_prefill_expert records).
    _slab_staged(vaayu_ref)[(1, 3)] = 42
    # _dispatch_expert_misses sweeps OTHER layers (k[0] != layer_idx). Must NOT raise
    # on the lookahead int keys, because it sweeps the per-expert dict only.
    _release_prefill_staged(expert_cache, vaayu_ref, lambda k: k[0] != 1)
    assert released == []          # (1,3) is layer 1, not swept by != 1
    # The layer's own finally sweep (k[0] == layer_idx) releases it.
    _release_prefill_staged(expert_cache, vaayu_ref, lambda k: k[0] == 1)
    assert released == [42]
    # The lookahead stager's int-keyed dict was never touched by the sweep.
    assert vaayu_ref._prefill_staged == {0: {5: vaayu_ref._prefill_staged[0][5]},
                                         2: {9: vaayu_ref._prefill_staged[2][9]}}
