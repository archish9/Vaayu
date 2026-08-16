# vaayu/speculative/__init__.py
from .base import SpeculativeDecoder


def build_speculator(
    drafter_model_path=None,
    base_model=None,
    device: str = "cuda:0",
    use_prompt_lookup: bool = False,
):
    """
    Build and return a speculative decoder.

    Choose based on workload:

      drafter_model_path (StandardDrafterDecoder)
        Best for: instruction-following, chat, novel responses.
        Costs: small model resident in VRAM (~0.3–1 GB for 0.5–1.5B drafters).
        Accept rate: 40–80% depending on drafter/verifier alignment.

      use_prompt_lookup (PromptLookupDecoder)
        Best for: code completion, template continuation, repetitive boilerplate.
        Avoid for: instruction→response — novel output tokens yield 0 match rate.
        Costs: zero VRAM; pure CPU n-gram search.
        Accept rate: 100–120% on matched steps; 0% (greedy fallback) on misses.

    Returns None if neither option is enabled (disables speculative decoding).
    """
    if drafter_model_path:
        from .standard import StandardDrafterDecoder
        # Bug 12b: refuse a vocabulary mismatch HERE, before any weights load.
        # Left unchecked it becomes an out-of-bounds embedding gather inside a
        # CUDA kernel, data-dependent on whether the prompt happens to use a
        # high token id. CLAUDE.md §5: warn loudly, never silently degrade.
        target_cfg = getattr(base_model, "config", None)
        ok, reason = drafter_vocab_compatible(
            drafter_vocab=_config_vocab_size(drafter_model_path),
            target_vocab=getattr(target_cfg, "vocab_size", None))
        if not ok:
            print(f"[Vaayu-Spec] WARNING: {reason} — the drafter and the target "
                  f"must share a tokenizer, or one of them will index the "
                  f"other's embedding out of bounds (a CUDA device-side assert, "
                  f"raised only for prompts that happen to use a high token id). "
                  f"DISABLING speculative decoding. Pair this model with a "
                  f"drafter from its own family to re-enable.")
            return None
        return StandardDrafterDecoder(drafter_model_path, device)

    if use_prompt_lookup:
        from .prompt_lookup import PromptLookupDecoder
        eos_token_id = base_model.config.eos_token_id if base_model is not None else None
        return PromptLookupDecoder(eos_token_id=eos_token_id)

    return None


__all__ = ["SpeculativeDecoder", "build_speculator",
           "drafter_vocab_compatible"]


def drafter_vocab_compatible(*, drafter_vocab, target_vocab):
    """(ok, reason) -- may this drafter verify against this target?

    StandardDrafterDecoder always required it ("Must share the same tokenizer as
    the Vaayu verifier model"); nothing enforced it, so a mismatch crashed inside
    a CUDA kernel instead of being refused (bug 12b, 2026-08-16). Measured: the
    drafter's embedding has 128256 rows (Llama-3.2-1B) while MiniMax-M2.7
    (vocab 200064) emits ids up to 131282 for ordinary code, so the gather ran
    off the end -- `vectorized_gather_kernel: Assertion 'ind >= 0 && ind <
    ind_dim_size' failed`, reported later as CUBLAS_STATUS_INTERNAL_ERROR from
    the poisoned context. DATA-DEPENDENT: a prompt whose ids all stay under the
    drafter's vocab runs fine, which is how it survived a measurement round.

    EQUALITY, not `drafter >= target`. A larger drafter vocab is unsafe in the
    other direction: the drafter can propose an id the VERIFIER cannot gather.
    Equality is what "same tokenizer" means.

    Unknown sizes (None) pass: this guard exists to catch a known-bad pairing,
    not to gate on missing metadata (D3 -- a diagnostic must never become the
    failure it is meant to explain).
    """
    if drafter_vocab is None or target_vocab is None:
        return True, ""
    if int(drafter_vocab) == int(target_vocab):
        return True, ""
    return False, (
        f"drafter vocab_size={int(drafter_vocab)} != target vocab_size="
        f"{int(target_vocab)}"
    )


def _config_vocab_size(path):
    """vocab_size off a checkpoint's config, without loading any weights, so a
    refusal costs nothing instead of the drafter's whole VRAM footprint.
    Reads the nested text_config first, as the rest of this repo does."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        inner = getattr(cfg, "text_config", None)
        return getattr(inner, "vocab_size", None) or getattr(cfg, "vocab_size", None)
    except Exception:
        return None      # unknown -> the guard above lets it through
