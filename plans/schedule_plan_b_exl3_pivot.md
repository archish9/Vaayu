# Schedule Plan B: The EXL3 Payload Pivot

**Priority:** 2 (Crucial Pathfinder)
**Effort:** Very High (C++/Python Integration)
**Risk:** Highest

## Objective
Shrink the physical size of the streamed data from 135GB to ~38GB by utilizing ExLlamaV3 (EXL3) 4.5 bpw quantization. This drops the fundamental hardware floor from ~23 seconds per token (current Era 5 baseline) to ~6 seconds per token.

## The Engineering Challenge
Currently, Vaayu streams `.safetensors` using standard PyTorch types. ExLlamaV3 uses custom C++/CUDA kernels specifically designed to perform matrix multiplication directly on 4-bit compressed data without decompressing it to 16-bit first. We must build a bridge between Vaayu's NVMe buffer and ExLlamaV3's compute kernels.

## Implementation Steps

### 0. Model Quantization (Prerequisite)
- Install ExLlamaV3 and its quantization toolchain (`pip install exllamav3`).
- Run the quantization script against the source `.safetensors` model to produce an EXL3 4.5 bpw artifact on disk. This step is compute-intensive and may take several hours.
- Verify the output file structure. EXL3 uses its own chunked binary format, which is fundamentally different from the sequential `.safetensors` shards Vaayu currently reads.

### 1. Model Preparation
- Analyze the EXL3 file structure to understand how layer data is addressed (byte offsets, chunk boundaries, metadata headers).
- This analysis directly informs the `safetensor_model_persister.py` rewrite in Step 3.

### 2. Pinning the Shared Layers
- Identify the shared/embedding layers and pin them permanently into ~10GB of the RTX 5060 Ti's VRAM.
- Leave ~6GB of VRAM for the KV Cache, OS overhead, and the Vaayu layer-streaming buffer.

### 3. The Custom Bridge
- Modify `safetensor_model_persister.py` to understand the EXL3 metadata.
- When `io_uring` drops the compressed 4-bit data into the pinned RAM buffer, wrap that memory pointer in a custom Python binding.
- Call ExLlamaV3's `forward()` equivalent, passing the raw tensor pointers directly into the EXL3 CUDA kernels.

## Verification
- Run a single forward pass without Speculative Decoding.
- Token generation time should drop to ~5-6 seconds. 
- Ensure outputs remain highly coherent, proving the 4.5 bpw quantization was successfully computed without precision corruption.
