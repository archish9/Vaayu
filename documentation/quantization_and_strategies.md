# Loader Strategies & EXL3 Quantization Guide

Vaayu uses a highly decoupled, pluggable architecture based on the **Strategy Pattern**. This allows the engine to transparently switch how model weights are structured, loaded, and executed depending on the files present in the checkpoint.

This document details Vaayu's loader strategies and provides the setup instructions for ExLlamaV3 4-bit (EXL3) quantized model execution.

---

## 1. Pluggable `LoaderStrategy` Architecture

The `LoaderStrategy` abstract interface decoupling storage loading from PyTorch execution exposes four clean lifecycle hooks:

```python
class LoaderStrategy(ABC):
    @abstractmethod
    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict:
        """Loads weights from NVMe into CPU Pinned RAM."""
        pass

    @abstractmethod
    def inject_to_gpu(self, model: nn.Module, layer_name: str, tensors: dict, 
                      device: str, dtype: torch.dtype, hf_quantizer=None, 
                      path_remappings: dict | None = None) -> list:
        """Moves tensors to GPU and attaches them to the model architecture."""
        pass

    @abstractmethod
    def free_from_gpu(self, model: nn.Module, layer_module: nn.Module, 
                      moved_layers: list, hf_quantizer=None) -> None:
        """Reclaims GPU VRAM immediately after the layer's forward pass."""
        pass

    @abstractmethod
    def setup_model(self, model: nn.Module, config, device: str, 
                    dtype: torch.dtype, pinned_layer_names: set | None = None) -> None:
        """Performs one-time model-level adjustments at initialization time."""
        pass
```

### Auto-Detection Protocol
When Vaayu instantiates a model via `AutoModel.from_pretrained()`, the engine peeks at the very first layer's `.safetensors` header.
* If the header contains EXL3-specific tensor keys (`q_weight`, `q_invperm`, `q_scale`, etc.), the engine dynamically instantiates the `EXL3LoaderStrategy`.
* Otherwise, it defaults to the `StandardSafetensorsStrategy` for FP16/BF16 weights.

---

## 2. EXL3 Quantization: C++ Kernel Execution

Standard uncompressed BF16 Llama 70B occupies **135 GB** on disk. When quantized to **4.5 bpw (bits per weight)** via ExLlamaV3 (EXL3), the payload shrinks by nearly 75% down to **~38 GB**. 

### 1. One-Time Model Surgery (`setup_model`)
Under the `EXL3LoaderStrategy`, Vaayu performs model-level surgery once at startup. It crawls through the empty model's layers and swaps all `nn.Linear` layers with custom `ExLlamaV3Linear` layers.
* **Exclusions**: Norms, embeddings (`embed_tokens`), and the output layer (`lm_head`) are explicitly excluded to maintain weight-tying properties.

### 2. Custom C++ QMatrix Generation (`inject_to_gpu`)
Instead of loading floating-point parameters, `ExLlamaV3Linear` works directly with compressed integers and scale factors. Inside `inject_to_gpu()`, the strategy moves these quantized tensors to the GPU and compiles a C++ **QMatrix** using ExLlamaV3's native extensions:
```python
q_matrix = exl3_ext.make_q_matrix(
    gpu_tensors.get("q_weight"),
    gpu_tensors.get("q_invperm"),
    gpu_tensors.get("q_scale"),
    gpu_tensors.get("q_scale_max"),
    gpu_tensors.get("q_groups"),
    gpu_tensors.get("q_perm"),
    device_idx
)
module.q_matrix = q_matrix
```
ExLlamaV3's custom CUDA kernels execute fast 4-bit matrix multiplications directly on this `QMatrix` during the forward pass.

### 3. VRAM Safety & Manual C++ Destructors (`free_from_gpu`)
This is the most critical aspect of the EXL3 strategy. Because the C++ `QMatrix` lives directly on the C++ heap outside PyTorch's virtual memory tracking, PyTorch's native Garbage Collector is completely blind to it. 

If you do not explicitly free it, the QMatrix remains in VRAM, leaking 1.6 GB per layer. **The 16 GB GPU will OOM and crash within 8 layers.**

To prevent this, `EXL3LoaderStrategy` overrides `free_from_gpu` to call the explicit C++ deallocator on every eviction:
```python
for sub_module in layer_module.modules():
    if isinstance(sub_module, ExLlamaV3Linear) and sub_module.q_matrix is not None:
        exl3_ext.free_q_matrix(sub_module.q_matrix)  # Explicit C++ deallocation
        sub_module.q_matrix = None
```

---

## 3. How to Quantize a Model to EXL3

To quantize a standard HF bfloat16/float16 model to EXL3 4.5bpw, use our pre-built `quantize.py` wrapper which correctly initializes exllamav3 arguments and maps standard calibration data.

### Foreground Execution Command
```bash
python quantize.py \
  --in_dir   /path/to/models/Llama-3.3-70B-Instruct \
  --out_dir  /path/to/models/Llama-3.3-70B-Instruct-exl3-4.5bpw \
  --work_dir /path/to/models/Llama-3.3-70B-Instruct-exl3-work \
  --bits     4.5
```

### Background Execution Command (With Log Tracking)
```bash
nohup python quantize.py \
  --in_dir   /path/to/models/Llama-3.3-70B-Instruct \
  --out_dir  /path/to/models/Llama-3.3-70B-Instruct-exl3-4.5bpw \
  --work_dir /path/to/models/Llama-3.3-70B-Instruct-exl3-work \
  --bits     4.5 \
  --resume > quantize.log 2>&1 &
```

### Quantization Arguments Guide

| Flag | Purpose | Default / Options |
|------|---------|-------------------|
| `--in_dir` | Path to the source uncompressed BF16 model. | Required |
| `--out_dir` | Path where the completed EXL3 model will be saved. | Required |
| `--work_dir` | Temp folder storing intermediate calibrations (~15 GB space needed). | Required |
| `--bits` | Target bits per weight. 4.5 bpw is recommended for coding tasks. | Required |
| `--resume` | Resumes an interrupted quantization process from its last checkpoint. | Option |
| `--devices` | Target GPU index. | `0` |

*Note: Once quantization completes successfully, the `--work_dir` directory can be safely deleted to free up disk space.*
