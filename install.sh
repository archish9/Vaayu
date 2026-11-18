#!/usr/bin/env bash
# Vaayu install script — handles build-order dependencies correctly.
#
# Usage (uv — recommended):
#   uv venv --python 3.12 --seed      # --seed is REQUIRED: a bare `uv venv` ships
#   source .venv/bin/activate         # no pip/setuptools/wheel, which the
#   bash install.sh                   # --no-build-isolation builds below need
#
# Usage (pip fallback):
#   python3.12 -m venv venv && source venv/bin/activate && bash install.sh
#
# Prefers `uv pip` (pip-compatible: same flags, including --no-build-isolation).
# Falls back to plain `pip` if uv is not installed.

set -e

PIP="uv pip"
command -v uv >/dev/null 2>&1 || PIP="pip"
echo "[Vaayu] Installer: $PIP"

echo "[Vaayu] Step 1/4: Installing core deps (torch, transformers, server, ...)"
# setuptools/wheel/packaging are listed explicitly: a `uv venv` without --seed has
# none of them, and the flash-attn / exllamav3 source builds below (run under
# --no-build-isolation) import them from the live env.
$PIP install \
    setuptools wheel packaging \
    "torch>=2.6.0" \
    "transformers==5.5.4" \
    "accelerate>=0.28.0" \
    "safetensors>=0.4.0" \
    "optimum>=1.17.0" \
    "huggingface-hub>=0.20.0" \
    "bitsandbytes>=0.43.0" \
    "compressed-tensors>=0.18.0" \
    "psutil>=5.9" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "pylint>=3.0" \
    ninja \
    "tqdm>=4.65.0" \
    "einops>=0.7.0" \
    "scipy>=1.11.0" \
    "sentencepiece>=0.2.0" \
    "peft>=0.10.0" \
    "evaluate>=0.4.0" \
    "scikit-learn>=1.3.0" \
    "wandb>=0.16.0" \
    "kernels>=0.13.0" \
    "vaayu-rust>=0.3.0" \
    Pillow

echo ""
echo "[Vaayu] Step 2/4: Installing flash-attn (compiles CUDA kernels, takes 5-15 min)..."
# --no-build-isolation lets the build see the torch already installed above.
# Without this flag, the build runs in a sandbox where torch is invisible and fails.
$PIP install flash-attn --no-build-isolation

echo ""
echo "[Vaayu] Step 3/4: Installing exllamav3 1.6.2 (EXL3 quant KERNELS only)..."
# Vaayu uses exllamav3 only as a kernel provider (BC_LinearEXL3 + g_tensor_cache);
# new-model support rides transformers + checkpoint modeling. Pinned 1.6.2 (upgraded
# from 0.0.34, S57 2026-10-15). The PyPI dist is py3-none-any and builds the C++/CUDA
# ext from source, so --no-build-isolation is required (needs CUDA 12/13 toolkit on
# PATH + ninja). If this OOMs during compile: prefix with MAX_JOBS=4.
#
# FAST PATH: if your box matches a published prebuilt wheel (e.g. this dev box:
# py3.12 / torch 2.11+cu130 / sm_120 uses the cu132 wheel), install that wheel with
# --no-deps instead — see the exllamav3 note in requirements.txt. No source build.
$PIP install "exllamav3==1.6.2" --no-build-isolation

echo ""
echo "[Vaayu] Step 4/4: Downloading exllamav3 calibration data files..."
# The source install does not include the .utf8 calibration data files.
# calibration_data.py hardcodes the path — must download into the installed package dir.
CAL_DIR="$(python3 -c "import exllamav3.conversion.calibration_data as m; import os; print(os.path.join(os.path.dirname(os.path.abspath(m.__file__)), 'standard_cal_data'))")"
BASE="https://raw.githubusercontent.com/turboderp-org/exllamav3/master/exllamav3/conversion/standard_cal_data"
for f in c4.utf8 code.utf8 multilingual.utf8 technical.utf8 tiny.utf8 wiki.utf8; do
    echo "  Downloading $f..."
    curl -s -o "$CAL_DIR/$f" "$BASE/$f"
done
echo "  Calibration data ready at: $CAL_DIR"

echo ""
echo "[Vaayu] Install complete."
echo "  System requirements: CUDA Toolkit 12.x/13.x + gcc/g++ must be present."
echo "  Optional (one architecture's reference kernels only): apache-tvm-ffi==0.1.9 + tilelang==0.1.8"
echo "  — see requirements.txt. Not installed by default; no other model path needs them."
echo ""
echo "  To quantize a model to EXL3:"
echo "    python quantize.py --in_dir <model_dir> --out_dir <out_dir> --work_dir <work_dir> --bits 4.5"
