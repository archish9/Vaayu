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
        return StandardDrafterDecoder(drafter_model_path, device)

    if use_prompt_lookup:
        from .prompt_lookup import PromptLookupDecoder
        eos_token_id = base_model.config.eos_token_id if base_model is not None else None
        return PromptLookupDecoder(eos_token_id=eos_token_id)

    return None


__all__ = ["SpeculativeDecoder", "build_speculator"]
