# CPU-Only Mode for Vaayu

Add an **opt-in** CPU-only inference mode so users can run ≤ 8B models without a GPU. GPU (`cuda:0`) remains the default — CPU is only used when the user explicitly passes `device="cpu"`.

## Design Decisions (Resolved)

| Question | Decision |
|---|---|
| Compression on CPU? | **Not in Phase 1.** Ship CPU mode without compression. Raise a clear error if user tries compression + CPU. |
| Profiler on CPU? | **Use psutil** to report RAM usage instead of GPU VRAM. Already a project dependency. |
| float16 → float32 on CPU? | **Auto-upgrade with info message.** float32 is 10-50x *faster* than float16 on CPU (CPUs lack native float16 ALUs). Respect explicit `bfloat16` from user. |
| Default device? | **GPU (`cuda:0`) remains the default.** CPU is opt-in only. Zero change for existing users. |

---

## Proposed Changes

### Component 1: Guard CUDA calls in core library

---

#### [MODIFY] [utils.py](file:///c:/laragon/www/AI-ML/Vaayu/vaayu/utils.py)

**`clean_memory()` (line 75-82)** — Guard `torch.cuda.empty_cache()`. This is the **#1 blocker**: called ~30+ times per forward pass.

```diff
 def clean_memory():
     gc.collect()
     try:
         ctypes.CDLL("libc.so.6").malloc_trim(0)
     except Exception as ex:
         # maybe platform
         pass
-    torch.cuda.empty_cache()
+    if torch.cuda.is_available():
+        torch.cuda.empty_cache()
```

---

#### [MODIFY] [profiler.py](file:///c:/laragon/www/AI-ML/Vaayu/vaayu/profiler.py)

**`add_profiling_time()` (line 19-22)** — Use psutil for RAM reporting on CPU:

```diff
         if self.print_memory:
-            free_mem = torch.cuda.mem_get_info()[0]
-            self.min_free_mem = min(self.min_free_mem, free_mem)
-            print(f"free vmem @{item}: {free_mem/1024/1024/1024:.02f}GB, min free: {self.min_free_mem/1024/1024/1024:.02f}GB")
+            if torch.cuda.is_available():
+                free_mem = torch.cuda.mem_get_info()[0]
+                self.min_free_mem = min(self.min_free_mem, free_mem)
+                print(f"free vmem @{item}: {free_mem/1024/1024/1024:.02f}GB, min free: {self.min_free_mem/1024/1024/1024:.02f}GB")
+            else:
+                import psutil
+                mem = psutil.virtual_memory()
+                free_ram = mem.available
+                self.min_free_mem = min(self.min_free_mem, free_ram)
+                print(f"free RAM @{item}: {free_ram/1024/1024/1024:.02f}GB, min free: {self.min_free_mem/1024/1024/1024:.02f}GB")
```

---

#### [MODIFY] [vaayu_base.py](file:///c:/laragon/www/AI-ML/Vaayu/vaayu/vaayu_base.py)

**Change 1: `__init__` dtype handling (lines 123-135)** — Auto-upgrade float16 → float32 on CPU:

```diff
         if dtype == torch.float16 and device.startswith("cuda"):
             try:
                 cc = torch.cuda.get_device_capability()
                 if cc[0] >= 8:
                     self.running_dtype = torch.bfloat16
                     print(f"✅ GPU supports BF16 (compute capability {cc[0]}.{cc[1]}). Using bfloat16 for better stability.")
                 else:
                     self.running_dtype = dtype
                     print(f"ℹ️ GPU compute capability {cc[0]}.{cc[1]} — using float16.")
             except Exception:
                 self.running_dtype = dtype
-        else:
+        elif not device.startswith("cuda") and dtype == torch.float16:
+            # CPU lacks native float16 ALUs — float32 is 10-50x faster on x86/ARM
+            self.running_dtype = torch.float32
+            print("ℹ️ CPU mode: using float32 (float16 is very slow on CPU).")
+        else:
             self.running_dtype = dtype
```

**Change 2: `__init__` compression guard (lines 102-104)** — Reject compression on CPU:

```diff
         if compression is not None:
             if not bitsandbytes_installed:
                 raise ImportError('WARNING: bitsandbytes not found...')
+            if not device.startswith("cuda"):
+                raise ValueError(
+                    "Compression (4bit/8bit) currently requires a CUDA GPU. "
+                    "For CPU-only mode, use compression=None. "
+                    "CPU compression support is planned for a future release."
+                )
```

---

### Component 2: Device-aware test scripts

All 4 test/example scripts use hardcoded `.cuda()`. Update them to auto-detect the device while keeping GPU as the default.

---

#### [MODIFY] [test_quick.py](file:///c:/laragon/www/AI-ML/Vaayu/test_quick.py)

```diff
+import torch
+
+# Device selection: GPU by default, falls back to CPU if no GPU available
+DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
+print(f"Using device: {DEVICE}")
+
 MAX_LENGTH = 128
 
 print("Loading model...")
-model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
+model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=DEVICE)
 
 ...
 
 print("Generating output...")
-print(input_tokens['input_ids'].cuda())
+print(input_tokens['input_ids'].to(DEVICE))
 generation_output = model.generate(
-    input_tokens['input_ids'].cuda(),
+    input_tokens['input_ids'].to(DEVICE),
     max_new_tokens=20,
```

---

#### [MODIFY] [test_small_model.py](file:///c:/laragon/www/AI-ML/Vaayu/test_small_model.py)

Same pattern: add `DEVICE` auto-detection, replace `.cuda()` → `.to(DEVICE)`, pass `device=DEVICE` to `from_pretrained()`.

---

#### [MODIFY] [test_local.py](file:///c:/laragon/www/AI-ML/Vaayu/test_local.py)

Same pattern.

---

#### [MODIFY] [inference_example.py](file:///c:/laragon/www/AI-ML/Vaayu/inference_example.py)

Same pattern.

---

### Component 3: Documentation

#### [MODIFY] [README.md](file:///c:/laragon/www/AI-ML/Vaayu/README.md)

Add a "CPU-Only Mode" section:

```markdown
## CPU-Only Mode

Vaayu supports running models on CPU for machines without a GPU:

​```python
model = AutoModel.from_pretrained("model-name", device="cpu")
​```

**Requirements:**
- Recommended: 32GB RAM for 8B models
- Peak memory: ~1-2GB (only one layer loaded at a time)
- Speed: ~2-10 seconds per token depending on CPU

**Limitations:**
- Compression (4bit/8bit) is not yet supported on CPU
- float16 is auto-upgraded to float32 for performance
```

---

## Files NOT modified (already work correctly)

| File | Why it's fine |
|---|---|
| `auto_model.py` | Already passes `**kwargs` to constructor → `device="cpu"` works |
| `vaayu_base.py:255` | Prefetch stream already guarded: `if prefetching and device.startswith("cuda")` |
| `vaayu_base.py:465` | `pin_memory()` already guarded: `if torch.cuda.is_available()` |
| `vaayu_base.py:353` | `diagnose_attention()` already guarded: `if torch.cuda.is_available()` |
| All model subclasses | Use `self.running_device` from base class — device-agnostic |

---

## Summary of changes

| File | Change | Lines |
|---|---|---|
| `vaayu/utils.py` | Guard `torch.cuda.empty_cache()` | ~2 |
| `vaayu/profiler.py` | psutil RAM reporting fallback | ~8 |
| `vaayu/vaayu_base.py` | dtype auto-upgrade + compression guard | ~10 |
| `test_quick.py` | Device auto-detection | ~6 |
| `test_small_model.py` | Device auto-detection | ~6 |
| `test_local.py` | Device auto-detection | ~6 |
| `inference_example.py` | Device auto-detection | ~6 |
| `README.md` | CPU-Only Mode docs | ~15 |
| **Total** | | **~60 lines** |

---

## Verification Plan

### Automated Tests

1. **Syntax check** all modified files:
   ```bash
   python -m py_compile vaayu/utils.py vaayu/vaayu_base.py vaayu/profiler.py
   ```

2. **Import test** (no GPU needed):
   ```bash
   python -c "from vaayu import AutoModel; print('Import OK')"
   ```

3. **CPU inference test** with TinyLlama:
   ```bash
   python test_quick.py
   ```

4. **GPU regression test** (if GPU available) — confirm zero behavioral change for existing GPU users

### Manual Verification

- Confirm `clean_memory()` no longer crashes on CPU-only machines
- Confirm compression + CPU raises a clear `ValueError`
- Confirm profiler prints RAM info on CPU
