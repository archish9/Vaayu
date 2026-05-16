# EXL3 Installation & Quantization Debug Log

## Environment
- OS: Linux (Ubuntu)
- Python: 3.12.3
- PyTorch: 2.11.0+cu130
- CUDA: 13.0
- GPU: RTX 5060 Ti 16GB

---

## Installation

### Problem 1: flash-attn fails in pip install -r requirements.txt

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```
Occurred when pip tried to build `flash-attn` from source. pip builds C++ packages in an isolated sandbox where the venv's torch is invisible. `flash-attn`'s `setup.py` imports torch to detect CUDA version — fails in sandbox.

**Fix:** Install `flash-attn` before `requirements.txt` using `--no-build-isolation`:
```bash
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```
`--no-build-isolation` disables the sandbox so the build can see the already-installed torch.

---

### Problem 2: exllamav3 pure Python wheel — missing compiled extensions

**Error:**
```
ModuleNotFoundError: No module named 'exllamav3.modules.attention'
```
pip downloaded `exllamav3-0.0.33-py3-none-any.whl` — a pure Python wheel with no compiled C++ extensions. The `attention` module is a compiled CUDA extension that was never built.

**Fix:** Install from GitHub source so C++ extensions are compiled:
```bash
pip install git+https://github.com/turboderp-org/exllamav3 --no-build-isolation
```
This installed `exllamav3-0.0.34` with the compiled `exllamav3_ext.cpython-312-x86_64-linux-gnu.so` extension.

---

### Problem 3: exllamav3 is a transitive dependency of flash-attn — same sandbox failure

Even after commenting out `flash-attn` from `requirements.txt`, `exllamav3` depends on `flash-attn` internally. pip tried to build it as a transitive dep and hit the same sandbox error.

**Fix:** Same — install `flash-attn` first with `--no-build-isolation` before anything else.

---

### Correct fresh install order

```bash
# Step 1: core deps (torch must exist before flash-attn builds)
pip install torch>=2.6.0 transformers accelerate safetensors optimum \
    huggingface-hub bitsandbytes ninja tqdm einops scipy sentencepiece \
    peft evaluate scikit-learn wandb kernels maturin

# Step 2: flash-attn — compiles CUDA kernels, takes 5-15 min
pip install flash-attn --no-build-isolation

# Step 3: exllamav3 from source (compiled C++ extensions)
pip install git+https://github.com/turboderp-org/exllamav3 --no-build-isolation
# If OOM during compile: MAX_JOBS=4 pip install ...
```

Or use `bash install.sh` which handles this order automatically.

---

### Recurring warning (ignorable)

```
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions.
  PyTorch 2.10.0+cu128 (you have 2.11.0+cu130)
  Python 3.10.19 (you have 3.12.3)
```
xformers was built for an older PyTorch/Python version. Not used by Vaayu or exllamav3. Safe to ignore.

---

## Quantization

### Problem 4: Wrong module path

**Error:**
```
/venv/bin/python: No module named exllamav3.convert
```
**Tried:** `python -m exllamav3.convert` — module doesn't exist.

**Fix:** Correct module is `exllamav3.conversion.convert_model`.

---

### Problem 5: Wrong argument names

**Error:** Silent exit with no output.

`--input` and `--output` don't exist. Correct argument names are:
- `--in_dir` (not `--input`)
- `--out_dir` (not `--output`)
- `--work_dir` (required — stores calibration data, ~10-20 GB, resumable with `--resume`)

---

### Problem 6: No `__main__` block in convert_model.py

**Error:** Silent exit even with correct module and args.

`exllamav3.conversion.convert_model` has no `if __name__ == "__main__":` block. Running it as a module does nothing.

**Fix:** Created `quantize.py` wrapper that imports the module-level `parser` and calls `prepare()` + `main()` explicitly:

```python
from exllamav3.conversion.convert_model import parser, prepare, main
parsed = parser.parse_args()
args, job_state, success, error = prepare(parsed)
if success:
    main(args, job_state)
```

---

### Problem 7: prepare() signature mismatch

**Error:**
```
TypeError: prepare() missing 1 required positional argument: 'args'
```
First version of `quantize.py` called `prepare()` with no args. `prepare()` requires the argparse namespace.

**Fix:** Parse args first, then pass to prepare:
```python
parsed = parser.parse_args()
args, job_state, success, error = prepare(parsed)
```

---

## Final working quantization command

```bash
python quantize.py \
  --in_dir   /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct \
  --out_dir  /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-4.5bpw \
  --work_dir /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-work \
  --bits     4.5
```

**Contents of `quantize.py`:**
```python
import sys
from exllamav3.conversion.convert_model import parser, prepare, main

if __name__ == "__main__":
    parsed = parser.parse_args()
    args, job_state, success, error = prepare(parsed)
    if not success:
        print(f"Error: {error}")
        sys.exit(1)
    main(args, job_state)
```

**Key flags:**
| Flag | Purpose | Default |
|------|---------|---------|
| `--in_dir` | Source model directory (BF16) | required |
| `--out_dir` | Output directory for EXL3 model | required |
| `--work_dir` | Temp dir for calibration + checkpoints | required |
| `--bits` | Bits per weight (e.g. 4.5) | required |
| `--head_bits` | Precision for lm_head layer | 6 |
| `--devices` | GPU indices e.g. `0` or `0,1` | 0 |
| `--resume` | Resume interrupted job from work_dir | false |

**work_dir** can be deleted after quantization completes. If job is interrupted, re-run same command with `--resume` added and it picks up from last checkpoint.

---

---

### Problem 8: Missing calibration data files in source install

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory:
'.../exllamav3/conversion/standard_cal_data/c4.utf8'
```
GitHub source install (`pip install git+https://...`) does not include the large `.utf8` calibration data files — only `__init__.py` is present in `standard_cal_data/`. The PyPI wheel bundles them but the source tree doesn't commit them (they're likely in `.gitignore` or a release asset).

`calibration_data.py` hardcodes the path relative to its own `__file__` — no CLI flag to override. Only fix is to download the files into that directory.

**Fix:** Download all 6 calibration files directly from the GitHub raw URL:
```bash
CAL_DIR="$(python3 -c "import exllamav3.conversion.calibration_data as m; import os; print(os.path.join(os.path.dirname(os.path.abspath(m.__file__)), 'standard_cal_data'))")"
BASE="https://raw.githubusercontent.com/turboderp-org/exllamav3/master/exllamav3/conversion/standard_cal_data"
for f in c4.utf8 code.utf8 multilingual.utf8 technical.utf8 tiny.utf8 wiki.utf8; do
    curl -s -o "$CAL_DIR/$f" "$BASE/$f"
done
```

**Files downloaded (~5.7 MB total):**
| File | Size | Content |
|------|------|---------|
| c4.utf8 | 1.3 MB | C4 dataset (general text) |
| code.utf8 | 1.3 MB | Code samples |
| multilingual.utf8 | 287 KB | Multilingual text |
| technical.utf8 | 342 KB | Technical documents |
| tiny.utf8 | 256 KB | TinyStories |
| wiki.utf8 | 2.0 MB | Wikipedia articles |

---

### Problem 9: Pillow not installed

**Error:**
```
ModuleNotFoundError: No module named 'PIL'
```
`requirements.txt` didn't include Pillow. Added `Pillow` to requirements.txt.

**Fix:** `pip install Pillow` (or already in requirements.txt now).

---

## Status
- [x] flash-attn installed (2.8.3)
- [x] exllamav3 installed from source (0.0.34, compiled extensions present)
- [x] quantize.py wrapper created
- [x] Calibration data files downloaded (c4.utf8, code.utf8, multilingual.utf8, technical.utf8, tiny.utf8, wiki.utf8)
- [ ] Quantization run confirmed working (pending next run)
