import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .model_persister import ModelPersister

try:
    import numpy as np
    import vaayu_rust
    _RUST_LOADER = True
    print(">>>> vaayu_rust loaded — using mmap fast_load")
except ImportError:
    _RUST_LOADER = False
    print(">>>> vaayu_rust not found — using Python safetensors loader")

# Maps safetensors dtype strings to torch dtypes.
_DTYPE_MAP = {
    "F16":  torch.float16,
    "BF16": torch.bfloat16,
    "F32":  torch.float32,
    "F64":  torch.float64,
    "I8":   torch.int8,
    "I16":  torch.int16,
    "I32":  torch.int32,
    "I64":  torch.int64,
    "U8":   torch.uint8,
    "BOOL": torch.bool,
}


def _raw_to_tensor(data, dtype_str: str, shape: list) -> torch.Tensor:
    """Convert uint8 numpy array + metadata from Rust into a typed torch tensor."""
    torch_dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
    # frombuffer reinterprets the raw bytes as the target dtype (zero-copy until clone).
    tensor = torch.frombuffer(data, dtype=torch.uint8)
    return tensor.view(torch_dtype).reshape(shape).clone()


class SafetensorModelPersister(ModelPersister):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._io_log_header_printed = False

    def model_persist_exist(self, layer_name, saving_path):
        safetensor_exists = Path(saving_path / (layer_name + 'safetensors')).exists()
        done_marker_exists = Path(saving_path / (layer_name + 'safetensors.done')).exists()
        return safetensor_exists and done_marker_exists

    def persist_model(self, state_dict, layer_name, saving_path):
        save_file(state_dict, saving_path / (layer_name + 'safetensors'))
        print(f"saved as: {saving_path / (layer_name + 'safetensors')}")
        (saving_path / (layer_name + 'safetensors.done')).touch()

    def load_model(self, layer_name, path):
        file_path = str(Path(path) / (layer_name + ".safetensors"))

        if _RUST_LOADER:
            t = time.time()
            raw = vaayu_rust.fast_load(file_path)
            elapsed = time.time() - t
            print(f"[VAAYU-IO] rust   {elapsed:.3f}s  {layer_name}")
            return {name: _raw_to_tensor(data, dtype, shape)
                    for name, (data, dtype, shape) in raw.items()}
        else:
            t = time.time()
            result = load_file(file_path, device="cpu")
            elapsed = time.time() - t
            print(f"[VAAYU-IO] python {elapsed:.3f}s  {layer_name}")
            return result
