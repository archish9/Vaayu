import json
import os


def detect_speculative_type(model_dir: str) -> str:
    """
    Auto-detect speculative decoding type from checkpoint directory.
    Returns: 'eagle', 'medusa', or 'none'.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return "none"

    with open(config_path) as f:
        config = json.load(f)

    # Medusa (checked first — more specific markers)
    if "medusa_num_heads" in config or "medusa_num_layers" in config:
        return "medusa"
    for fname in os.listdir(model_dir):
        if fname.startswith("medusa_lm_head"):
            return "medusa"

    # EAGLE
    if "speculators_config" in config:
        method = config["speculators_config"].get("method", "")
        if "eagle" in method.lower():
            return "eagle"
    if "eagle_config" in config:
        return "eagle"
    for arch in config.get("architectures", []):
        if "eagle" in arch.lower():
            return "eagle"

    return "none"
