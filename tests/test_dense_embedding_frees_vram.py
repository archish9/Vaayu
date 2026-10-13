"""Dense embedding-free-VRAM decision (S43, builds S41's lever M-6).

Pure, no CUDA / no model load (same discipline as test_vram_plan.py and
test_residency_first_pin.py). The helper decides whether `_init_pinned_layers`
phase 1 should hold a DENSE model's embedding on the CPU instead of VRAM-pinning
it, so the freed VRAM buys MORE resident transformer layers.

The rule (S41 measured +9.77 % decode on Llama-3.3-70B streamed by freeing the
embedding's 1.957 GiB for +4 transformer layers):
  * fire ONLY when the pinning budget cannot hold the whole model -- so phase 2
    will STREAM >= 1 transformer layer anyway, and the freed bytes convert to
    resident layers. A model the budget holds whole (Llama-3.2-1B) keeps the
    embedding VRAM-pinned exactly as before (no streaming -> no gain).
  * AND only when the embedding is at least one transformer layer in size, so
    freeing it buys >= 1 resident layer (net win).

Model-agnostic (CLAUDE.md §5): the caller derives embed_bytes, per_layer_bytes,
total_layers and budget; this helper hardcodes no model number. Two contrasting
dense geometries are parametrized below (70B-like streams, 1B-like fits).
"""
from vaayu.vram_plan import dense_embedding_frees_vram

GB = 1024 ** 3


def _frees(*, embed, per_layer, total_layers, budget, shared_other=0.0):
    return dense_embedding_frees_vram(
        embed_bytes=int(embed * GB), per_layer_bytes=int(per_layer * GB),
        total_layers=int(total_layers), budget=int(budget * GB),
        shared_other_bytes=int(shared_other * GB))


def test_fires_on_a_streamed_dense_model_llama_70b_like():
    # Llama-3.3-70B-exl3 at HEAD (S41): embed 1.957 GiB, one layer 0.4984 GiB,
    # 80 layers, pinning budget 10.955 GiB. Whole model ~41.8 GiB >> budget ->
    # phase 2 streams; embed (1.957) >= one layer (0.4984) -> free it.
    assert _frees(embed=1.957, per_layer=0.4984, total_layers=80, budget=10.955)


def test_does_not_fire_when_the_budget_holds_the_whole_model_llama_1b_like():
    # Llama-3.2-1B fits entirely: embed 0.49 GiB, one layer ~0.10 GiB, 16 layers,
    # whole model ~2.1 GiB << 10.955 GiB budget -> no streaming -> keep pinned.
    assert not _frees(embed=0.49, per_layer=0.10, total_layers=16, budget=10.955)


def test_does_not_fire_when_the_embedding_is_smaller_than_one_layer():
    # Streams (whole model > budget) but freeing the embedding buys < 1 layer,
    # so there is no net gain: do not disturb the pin.
    assert not _frees(embed=0.30, per_layer=0.4984, total_layers=80, budget=10.955)


def test_fires_when_the_embedding_is_exactly_one_layer():
    # Boundary: embed == one layer frees exactly one layer -> still a win.
    assert _frees(embed=0.4984, per_layer=0.4984, total_layers=80, budget=10.955)


def test_does_not_fire_at_the_whole_model_fits_exactly_boundary():
    # whole_model == budget is NOT streaming (phase 2 pins all) -> keep pinned.
    assert not _frees(embed=2.0, per_layer=1.0, total_layers=8, budget=10.0)
    # One byte over the budget streams -> fire.
    assert _frees(embed=2.0, per_layer=1.0, total_layers=8,
                  budget=10.0 - 1.0 / GB)


def test_counts_the_other_shared_layers_toward_the_fit():
    # norm + lm_head push the whole model over the budget: freeing the embedding
    # is still the right call. shared_other only ever makes the fit tighter.
    assert _frees(embed=2.0, per_layer=1.0, total_layers=8, budget=10.5,
                  shared_other=1.0)


def test_degenerate_inputs_never_fire():
    assert not _frees(embed=1.957, per_layer=0.4984, total_layers=80, budget=0.0)
    assert not _frees(embed=1.957, per_layer=0.0, total_layers=80, budget=10.955)
    assert not _frees(embed=1.957, per_layer=0.4984, total_layers=0, budget=10.955)
