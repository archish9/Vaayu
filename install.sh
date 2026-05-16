#!/usr/bin/env bash
# Vaayu install script — handles build-order dependencies correctly.
# Usage: bash install.sh

set -e

echo "[Vaayu] Step 1/4: Installing core deps (torch, transformers, accelerate...)"
pip install \
    "torch>=2.6.0" \
    "transformers==5.5.4" \
    "accelerate>=0.28.0" \
    "safetensors>=0.4.0" \
    "optimum>=1.17.0" \
    "huggingface-hub>=0.20.0" \
    "bitsandbytes>=0.43.0" \
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
    maturin \
    Pillow

echo ""
echo "[Vaayu] Step 2/4: Installing flash-attn (compiles CUDA kernels, takes 5-15 min)..."
# --no-build-isolation lets the build see the torch already installed above.
# Without this flag, pip builds in a sandbox where torch is invisible and the build fails.
pip install flash-attn --no-build-isolation

echo ""
echo "[Vaayu] Step 3/4: Installing exllamav3 from source (compiled C++ extensions)..."
# PyPI wheel is pure Python (no compiled CUDA extensions) — must install from source.
# --no-build-isolation lets the build see flash-attn already installed.
# If this OOMs during compile: MAX_JOBS=4 pip install git+https://github.com/turboderp-org/exllamav3 --no-build-isolation
pip install git+https://github.com/turboderp-org/exllamav3 --no-build-isolation

echo ""
echo "[Vaayu] Step 4/4: Downloading exllamav3 calibration data files..."
# The GitHub source install does not include the .utf8 calibration data files.
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
echo "  System requirements: CUDA Toolkit 12.x + gcc/g++ must be present."
echo ""
echo "  To quantize a model to EXL3:"
echo "    python quantize.py --in_dir <model_dir> --out_dir <out_dir> --work_dir <work_dir> --bits 4.5"
