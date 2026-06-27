import json
import re
import struct
from pathlib import Path
import torch

_EXPERT_RE = re.compile(r'\.experts\.(\d+)\.')


class MoEKeyRemapper:
    def __init__(self, mode: str, layer_prefix: str):
        self.mode = mode
        self.layer_prefix = layer_prefix

    def remap_state_dict(self, state_dict: dict) -> dict:
        if self.mode != "bsm_to_mlp_fused":
            return state_dict

        result = {}
        expert_groups = {}  # {expert_idx: {w1: tensor, w2: tensor, w3: tensor}}

        for key, tensor in state_dict.items():
            m = _EXPERT_RE.search(key)
            if m:
                expert_idx = int(m.group(1))
                # e.g. "w1.weight" from the key tail
                weight_name = key.rsplit(".", 2)[-2]  # w1, w2, or w3
                expert_groups.setdefault(expert_idx, {})[weight_name] = tensor
            else:
                # Path rename: block_sparse_moe -> mlp
                new_key = key.replace("block_sparse_moe", "mlp")
                result[new_key] = tensor

        # Fuse experts into batched tensors
        if expert_groups:
            # Assumes expert_idx starts at 0 and is contiguous
            n_experts = max(expert_groups.keys()) + 1
            gate_up_list = []
            down_list = []
            
            for eidx in range(n_experts):
                if eidx in expert_groups:
                    eg = expert_groups[eidx]
                    # Check if w1 and w3 are present. If not, it means this shard
                    # might not have all weights (e.g. if experts are split across shards,
                    # though typically they aren't, or if we are only loading gate).
                    # Actually, if we are loading specific tensors, we just construct what we can.
                    # Wait, if we are loading a full layer, we have w1, w2, w3.
                    if "w1" in eg and "w3" in eg:
                        gate_up_list.append(torch.cat([eg["w1"], eg["w3"]], dim=0))
                    if "w2" in eg:
                        down_list.append(eg["w2"])

            if len(gate_up_list) == n_experts:
                # Note: This is a hacky way to get the layer name prefix.
                # Assuming all expert keys start with the same prefix.
                # Let's just find the prefix dynamically or use a known one.
                # Find one original expert key to get the prefix.
                first_expert_key = list(state_dict.keys())[0] # Not guaranteed to be an expert key
                # Let's find an expert key
                expert_prefix = None
                for k in state_dict.keys():
                    if ".experts." in k:
                        expert_prefix = k.split(".experts.")[0].replace("block_sparse_moe", "mlp")
                        break
                
                if expert_prefix:
                    result[f"{expert_prefix}.experts.gate_up_proj"] = torch.stack(gate_up_list)
                    result[f"{expert_prefix}.experts.down_proj"] = torch.stack(down_list)

        return result


def build_moe_key_remap(model: torch.nn.Module, checkpoint_path: str, layer_names_dict: dict) -> MoEKeyRemapper:
    """
    Probes the first MoE layer shard's header keys and the model's named_parameters().
    Returns a MoEKeyRemapper object configured for the detected structure.
    """
    mode = "identity"
    layer_prefix = layer_names_dict.get("layer_prefix", "model.layers")
    
    # Grab live model's parameter names for the first layer (layer 0)
    model_keys = set()
    try:
        # Resolve the layer 0 module
        layer_list_path = layer_prefix.split(".")
        mod = model
        for part in layer_list_path:
            if hasattr(mod, part):
                mod = getattr(mod, part)
        
        # If it's a ModuleList or list-like, get the first one
        if len(mod) > 0:
            layer0 = mod[0]
            for name, _ in layer0.named_parameters():
                model_keys.add(f"{layer_prefix}.0.{name}")
    except Exception as e:
        print(f"[Vaayu-MoE-Remap] Warning: Could not probe model keys: {e}")

    # Check disk keys from layer 0 shard
    disk_keys = set()
    shard_path = Path(checkpoint_path) / f"{layer_prefix}.0.safetensors"
    if shard_path.exists():
        try:
            with open(shard_path, "rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hdr_len))
                disk_keys = set(hdr.keys())
        except Exception as e:
            print(f"[Vaayu-MoE-Remap] Warning: Could not probe disk keys from {shard_path}: {e}")
    else:
        # Try to find any .safetensors file that looks like a layer
        for p in Path(checkpoint_path).glob("*.safetensors"):
            if layer_prefix in p.name:
                try:
                    with open(p, "rb") as f:
                        hdr_len = struct.unpack("<Q", f.read(8))[0]
                        hdr = json.loads(f.read(hdr_len))
                        disk_keys = set(hdr.keys())
                        break
                except Exception:
                    pass

    has_bsm_disk = any("block_sparse_moe" in k for k in disk_keys)
    has_bsm_model = any("block_sparse_moe" in k for k in model_keys)
    has_mlp_model = any("mlp." in k for k in model_keys)

    if has_bsm_disk and not has_bsm_model and has_mlp_model:
        mode = "bsm_to_mlp_fused"
        print("[Vaayu-MoE-Remap] Detected Mixtral fused experts (bsm_to_mlp_fused)")
    elif has_bsm_disk and has_bsm_model:
        mode = "identity"
        print("[Vaayu-MoE-Remap] Detected standard block_sparse_moe (identity)")
    else:
        mode = "identity"
        print("[Vaayu-MoE-Remap] Defaulting to identity mode")

    return MoEKeyRemapper(mode=mode, layer_prefix=layer_prefix)
