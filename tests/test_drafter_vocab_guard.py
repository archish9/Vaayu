"""Bug 12b: drafter and target must share a vocabulary, and it must be CHECKED.

StandardDrafterDecoder's own docstring already required it ("Must share the same
tokenizer as the Vaayu verifier model") -- nothing enforced it, so the pairing
crashed inside a CUDA kernel instead of being refused at build time.

Measured 2026-08-16 on this box: Llama-3.2-1B's embedding has 128256 rows;
MiniMax-M2.7 (vocab 200064) tokenizes ordinary code to ids as high as 131282, so
the drafter's embedding gathered out of bounds --
`vectorized_gather_kernel: Assertion 'ind >= 0 && ind < ind_dim_size' failed`,
surfacing later as CUBLAS_STATUS_INTERNAL_ERROR from the poisoned context. It is
DATA-DEPENDENT: a 3-line prompt (max id 109716) ran fine, a 4-line one (131282)
crashed, which is exactly how it survived a full measurement round undetected.
"""
from vaayu.speculative import drafter_vocab_compatible

# Real vocab sizes, read from each checkpoint's config.json (CLAUDE.md §5: these
# identify the measured case, they are not sizing constants).
LLAMA_1B = LLAMA_70B = 128256
MINIMAX = 200064
QWEN = 151936


def test_the_measured_crash_pairing_is_refused():
    ok, reason = drafter_vocab_compatible(drafter_vocab=LLAMA_1B, target_vocab=MINIMAX)
    assert not ok
    assert "128256" in reason and "200064" in reason, "name both sizes, or the warning is unactionable"


def test_qwen_is_refused_too_though_the_fix9_gate_usually_hides_it():
    """Qwen3-235B has the same mismatch; today it is masked only because
    spec_decode_fits disables speculation first. Remove that gate and this
    pairing would crash the same way -- so the vocab check must stand alone."""
    ok, _ = drafter_vocab_compatible(drafter_vocab=LLAMA_1B, target_vocab=QWEN)
    assert not ok


def test_the_pairing_that_actually_works_is_allowed():
    """Llama-3.3-70B shares the drafter's tokenizer, which is why row 12's
    5.57x dense win ran at all. The guard must not break it."""
    ok, reason = drafter_vocab_compatible(drafter_vocab=LLAMA_1B, target_vocab=LLAMA_70B)
    assert ok and reason == ""


def test_a_LARGER_drafter_vocab_is_refused_as_well():
    """Not >=, but ==. A drafter with extra rows can propose an id the VERIFIER
    cannot gather -- the same crash with the models swapped. 'Same tokenizer'
    means equal, and anything looser ships a subtler version of this bug."""
    ok, _ = drafter_vocab_compatible(drafter_vocab=MINIMAX, target_vocab=LLAMA_70B)
    assert not ok


def test_unknown_vocab_sizes_do_not_refuse():
    """A config that does not expose vocab_size must not disable speculation --
    the guard exists to catch a known-bad pairing, not to gate on missing
    metadata (D3: a diagnostic must never become the failure)."""
    assert drafter_vocab_compatible(drafter_vocab=None, target_vocab=MINIMAX)[0]
    assert drafter_vocab_compatible(drafter_vocab=LLAMA_1B, target_vocab=None)[0]
