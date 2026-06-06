# vaayu/speculative/__init__.py
from .base import SpeculativeDecoder


def build_speculator(
    drafter_model_path=None,
    base_model=None,
    device: str = "cuda:0",
):
    """
    Build and return a StandardDrafterDecoder if a drafter_model_path is given.

    Standard drafting uses a separate small HF model that lives permanently in VRAM.
    It works with any Vaayu-loaded model because it is architecturally independent —
    no checkpoint-specific speculator weights are required.

    Returns None if no drafter path is provided (disables speculative decoding).
    """
    if drafter_model_path:
        from .standard import StandardDrafterDecoder
        return StandardDrafterDecoder(drafter_model_path, device)

    return None


__all__ = ["SpeculativeDecoder", "build_speculator"]
