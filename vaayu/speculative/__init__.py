# vaayu/speculative/__init__.py
from .detection import detect_speculative_type
from .base import SpeculativeDecoder


def build_speculator(
    eagle_head_path=None,
    drafter_model_path=None,
    base_model=None,
    device: str = "cuda:0",
):
    """
    Build and return the appropriate SpeculativeDecoder.

    Priority: EAGLE/Medusa > StandardDrafter > None.
    Falls back to standard drafter if eagle_head_path contains no EAGLE markers,
    or if EAGLE initialization fails.
    """
    if eagle_head_path:
        spec_type = detect_speculative_type(eagle_head_path)
        if spec_type in ("eagle", "medusa"):
            try:
                from .eagle import EAGLEDecoder
                return EAGLEDecoder(eagle_head_path, base_model, device)
            except Exception as e:
                print(f"[Vaayu-Spec] EAGLE init failed ({e}), trying standard drafter")

    if drafter_model_path:
        from .standard import StandardDrafterDecoder
        return StandardDrafterDecoder(drafter_model_path, device)

    return None


__all__ = ["SpeculativeDecoder", "build_speculator", "detect_speculative_type"]
