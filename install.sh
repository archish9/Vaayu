#!/usr/bin/env bash
# Vaayu install script — handles build-order dependencies correctly.
# Usage: bash install.sh

set -e

echo "[Vaayu] Step 1/3: Installing core deps (torch, transformers, accelerate...)"
pip install \
    torch>=2.6.0 \
    transformers==5.5.4 \
    accelerate>=0.28.0 \
    safetensors>=0.4.0 \
    optimum>=1.17.0 \
    huggingface-hub>=0.20.0 \
    bitsandbytes>=0.43.0 \
    ninja \
    tqdm>=4.65.0 \
    einops>=0.7.0 \
    scipy>=1.11.0 \
    sentencepiece>=0.2.0 \
    peft>=0.10.0 \
    evaluate>=0.4.0 \
    scikit-learn>=1.3.0 \
    wandb>=0.16.0 \
    kernels>=0.13.0 \
    maturin

echo ""
echo "[Vaayu] Step 2/3: Installing flash-attn (compiles CUDA kernels, takes 5-15 min)..."
# --no-build-isolation lets the build see the torch already installed above.
# Without this flag, pip builds in a sandbox where torch is invisible and the build fails.
pip install flash-attn --no-build-isolation

echo ""
echo "[Vaayu] Step 3/3: Installing exllamav3 (flash-attn already satisfied)..."
# If this OOMs during compile: MAX_JOBS=4 pip install exllamav3
pip install exllamav3>=0.0.18

echo ""
echo "[Vaayu] Install complete."
echo "  System requirements: CUDA Toolkit 12.x + gcc/g++ must be present."
echo "  If exllamav3 compile OOMed, retry with: MAX_JOBS=4 pip install exllamav3"
