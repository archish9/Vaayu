# Subclass Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse 5 trivially-empty `VaayuBaseModel` subclasses into the base class via kwargs, add `_auto_detect_layer_names()` as a fallback for unknown architectures, and delete the dead files.

**Architecture:** Add `layer_names_dict` and `tokenizer_class` kwargs to `VaayuBaseModel.__init__`. After `init_model()`, run auto-detect to patch any layer name key that doesn't resolve on the actual model. `AutoModel` passes the right kwargs directly instead of dispatching to a dedicated subclass. Two subclasses with genuinely-different attention APIs (`VaayuQWen` v1, `VaayuChatGLM`) are kept unchanged.

**Tech Stack:** Python 3.10+, PyTorch 2.x, HuggingFace Transformers, pytest

---

## File Map

| File | Change |
|---|---|
| `vaayu/vaayu_base.py` | Add module-level helpers, new kwargs, flip default, auto-detect call |
| `vaayu/auto_model.py` | Return `(class, kwargs)` directly, add imports |
| `vaayu/__init__.py` | Remove 5 dead imports |
| `vaayu/vaayu_qwen2.py` | **Delete** |
| `vaayu/vaayu_mistral.py` | **Delete** |
| `vaayu/vaayu_mixtral.py` | **Delete** |
| `vaayu/vaayu_internlm.py` | **Delete** |
| `vaayu/vaayu_baichuan.py` | **Delete** |
| `tests/test_layer_detection.py` | **Create** — unit tests for helpers |
| `tests/test_auto_model_dispatch.py` | **Create** — unit tests for AutoModel dispatch |

---

## Task 0: Create branch

- [ ] **Step 1: Create and switch to feature branch**

```bash
git checkout -b refactor/subclass-consolidation
```

Expected: `Switched to a new branch 'refactor/subclass-consolidation'`

---

## Task 1: Add module-level helpers to `vaayu_base.py`

Two pure functions added above the `VaayuBaseModel` class (after line 51, before line 53). No class changes yet.

**Files:**
- Modify: `vaayu/vaayu_base.py` (before line 53)
- Create: `tests/test_layer_detection.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/__init__.py` (empty) and `tests/test_layer_detection.py`:

```python
import torch
import torch.nn as nn
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vaayu.vaayu_base import _layer_path_exists, _auto_detect_layer_names


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(16, 16, bias=False)


class _LlamaLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 16)
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(4)])
        self.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 100, bias=False)

    # simulate HF nesting: model.embed_tokens etc.
    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(100, 16)
            self.layers = nn.ModuleList([_FakeLayer() for _ in range(4)])
            self.norm = nn.LayerNorm(16)


class _HFLike(nn.Module):
    """Mimics HuggingFace structure: model.embed_tokens, model.layers, lm_head at top."""
    def __init__(self):
        super().__init__()
        self.model = _LlamaLike()
        self.lm_head = nn.Linear(16, 100, bias=False)


# --- _layer_path_exists ---

def test_layer_path_exists_shallow():
    m = _LlamaLike()
    assert _layer_path_exists(m, "embed_tokens") is True

def test_layer_path_exists_nested():
    m = _HFLike()
    assert _layer_path_exists(m, "model.embed_tokens") is True

def test_layer_path_exists_missing():
    m = _LlamaLike()
    assert _layer_path_exists(m, "model.norm") is False

def test_layer_path_exists_empty_string():
    m = _LlamaLike()
    assert _layer_path_exists(m, "") is False


# --- _auto_detect_layer_names ---

def test_auto_detect_llama_like():
    m = _LlamaLike()
    result = _auto_detect_layer_names(m)
    assert result["layer_prefix"] == "layers"
    assert result["embed"] == "embed_tokens"
    assert result["lm_head"] == "lm_head"
    assert result["norm"] == "norm"

def test_auto_detect_hf_nested():
    m = _HFLike()
    result = _auto_detect_layer_names(m)
    assert result["layer_prefix"] == "model.layers"
    assert result["embed"] == "model.embed_tokens"
    assert result["norm"] == "model.norm"
    assert result["lm_head"] == "lm_head"

def test_auto_detect_empty_model():
    m = nn.Module()
    result = _auto_detect_layer_names(m)
    assert result == {}
```

- [ ] **Step 2: Run tests — expect ImportError (functions don't exist yet)**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -m pytest tests/test_layer_detection.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name '_layer_path_exists'`

- [ ] **Step 3: Add `import torch.nn as nn` to `vaayu_base.py` imports**

In `vaayu/vaayu_base.py`, after line 8 (`import torch`), add:

```python
import torch.nn as nn
```

- [ ] **Step 4: Add module-level helpers before the `VaayuBaseModel` class**

In `vaayu/vaayu_base.py`, insert the following block immediately before line 53 (`class VaayuBaseModel`). Insert after the `try/except` blocks for `bitsandbytes` and `cache_utils`:

```python
def _layer_path_exists(model: nn.Module, path: str) -> bool:
    if not path:
        return False
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return True


def _auto_detect_layer_names(model: nn.Module) -> dict:
    EMBED = {"embed_tokens", "word_embeddings", "wte", "embed", "token_embeddings"}
    NORM = {"norm", "final_layernorm", "ln_f", "final_norm", "layer_norm", "ln_out"}
    HEAD = {"lm_head", "output_layer", "embed_out", "output", "head"}

    best_path, best_len = "", 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
            best_len, best_path = len(mod), name

    if not best_path:
        return {}

    embed_path = norm_path = lm_head_path = None
    for name, mod in model.named_modules():
        if name.startswith(best_path + "."):
            continue
        leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if embed_path is None and leaf in EMBED and isinstance(mod, nn.Embedding):
            embed_path = name
        if norm_path is None and leaf in NORM:
            norm_path = name
        if lm_head_path is None and leaf in HEAD and isinstance(mod, nn.Linear):
            lm_head_path = name

    return {
        "embed": embed_path,
        "layer_prefix": best_path,
        "norm": norm_path,
        "lm_head": lm_head_path,
    }
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -m pytest tests/test_layer_detection.py -v
```

Expected:
```
tests/test_layer_detection.py::test_layer_path_exists_shallow PASSED
tests/test_layer_detection.py::test_layer_path_exists_nested PASSED
tests/test_layer_detection.py::test_layer_path_exists_missing PASSED
tests/test_layer_detection.py::test_layer_path_exists_empty_string PASSED
tests/test_layer_detection.py::test_auto_detect_llama_like PASSED
tests/test_layer_detection.py::test_auto_detect_hf_nested PASSED
tests/test_layer_detection.py::test_auto_detect_empty_model PASSED
7 passed
```

- [ ] **Step 6: Commit**

```bash
git add vaayu/vaayu_base.py tests/__init__.py tests/test_layer_detection.py
git commit -m "feat: add _auto_detect_layer_names and _layer_path_exists helpers"
```

---

## Task 2: Update `VaayuBaseModel` — kwargs, flip default, tokenizer, auto-detect call

**Files:**
- Modify: `vaayu/vaayu_base.py`

- [ ] **Step 1: Write failing tests for new kwargs**

Create `tests/test_vaayu_base_kwargs.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
import pytest

# We test VaayuBaseModel internals without loading a real model by patching
# all the heavy init methods.
PATCH_TARGETS = [
    "vaayu.vaayu_base.find_or_create_local_splitted_path",
    "vaayu.vaayu_base.AutoConfig.from_pretrained",
    "vaayu.vaayu_base.AutoTokenizer.from_pretrained",
    "vaayu.vaayu_base.GenerationConfig.from_pretrained",
    "vaayu.vaayu_base.build_loader_strategy",
    "vaayu.vaayu_base.VaayuBaseModel.init_model",
    "vaayu.vaayu_base.VaayuBaseModel._detect_moe",
    "vaayu.vaayu_base._auto_detect_layer_names",
    "vaayu.vaayu_base._layer_path_exists",
]


def _make_mock_model():
    import torch
    import torch.nn as nn

    class FakeLayer(nn.Module):
        pass

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(100, 16)
            self.layers = nn.ModuleList([FakeLayer() for _ in range(4)])
            self.norm = nn.LayerNorm(16)
            self.lm_head = nn.Linear(16, 100, bias=False)

    return FakeModel()


def _make_base(extra_kwargs=None):
    """Construct VaayuBaseModel with all heavy I/O patched out."""
    import torch
    from vaayu.vaayu_base import VaayuBaseModel

    fake_model = _make_mock_model()

    mock_config = MagicMock()
    mock_config.architectures = ["LlamaForCausalLM"]
    mock_config.vocab_size = 100
    mock_config.hidden_size = 16
    mock_config.num_hidden_layers = 4
    mock_config.num_attention_heads = 4
    mock_config.num_key_value_heads = 4
    mock_config.intermediate_size = 64
    mock_config.rms_norm_eps = 1e-5
    mock_config.max_position_embeddings = 512
    mock_config.hidden_act = "silu"
    mock_config.rope_theta = 10000.0
    mock_config.pad_token_id = None
    mock_config.eos_token_id = 2
    mock_config.bos_token_id = 1

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.bos_token_id = 1

    with patch("vaayu.vaayu_base.find_or_create_local_splitted_path", return_value=("/fake", "/fake/split")), \
         patch("vaayu.vaayu_base.AutoConfig.from_pretrained", return_value=mock_config), \
         patch("vaayu.vaayu_base.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("vaayu.vaayu_base.GenerationConfig.from_pretrained", side_effect=Exception("no gen config")), \
         patch("vaayu.vaayu_base.build_loader_strategy", return_value=MagicMock()), \
         patch.object(VaayuBaseModel, "init_model", lambda self: setattr(self, "model", fake_model)), \
         patch.object(VaayuBaseModel, "_detect_moe", lambda self: None), \
         patch("vaayu.vaayu_base._auto_detect_layer_names", return_value={}), \
         patch("vaayu.vaayu_base._layer_path_exists", return_value=True):

        kwargs = {"device": "cpu", "dtype": torch.float32}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        m = VaayuBaseModel.__new__(VaayuBaseModel)
        VaayuBaseModel.__init__(m, "/fake/model", **kwargs)
    return m


def test_default_layer_names_are_llama():
    m = _make_base()
    assert m.layer_names_dict["embed"] == "model.embed_tokens"
    assert m.layer_names_dict["layer_prefix"] == "model.layers"
    assert m.layer_names_dict["norm"] == "model.norm"
    assert m.layer_names_dict["lm_head"] == "lm_head"


def test_layer_names_dict_kwarg_overrides():
    custom = {"embed": "transformer.wte", "layer_prefix": "transformer.h",
              "norm": "transformer.ln_f", "lm_head": "lm_head"}
    m = _make_base({"layer_names_dict": custom})
    assert m.layer_names_dict["embed"] == "transformer.wte"
    assert m.layer_names_dict["layer_prefix"] == "transformer.h"


def test_layer_names_dict_kwarg_partial_override():
    m = _make_base({"layer_names_dict": {"embed": "custom.embed"}})
    assert m.layer_names_dict["embed"] == "custom.embed"
    assert m.layer_names_dict["lm_head"] == "lm_head"  # default unchanged


def test_get_use_better_transformer_default_false():
    from vaayu.vaayu_base import VaayuBaseModel
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    assert m.get_use_better_transformer() is False


def test_tokenizer_class_kwarg_used():
    mock_tok_cls = MagicMock()
    mock_tok_cls.from_pretrained.return_value = MagicMock()
    m = _make_base({"tokenizer_class": mock_tok_cls})
    mock_tok_cls.from_pretrained.assert_called_once()
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -m pytest tests/test_vaayu_base_kwargs.py -v 2>&1 | head -40
```

Expected: `TypeError` (unexpected kwargs) or `AssertionError` on `get_use_better_transformer`.

- [ ] **Step 3: Add `layer_names_dict` and `tokenizer_class` kwargs to `__init__`**

In `vaayu/vaayu_base.py`, replace line 64–67:

```python
    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False):
```

with:

```python
    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False,
                 use_torch_compile=False,
                 layer_names_dict=None, tokenizer_class=None):
```

- [ ] **Step 4: Store the new kwargs before `set_layer_names_dict()` call**

In `vaayu/vaayu_base.py`, replace lines 117–119:

```python
        # Save parameters

        self.set_layer_names_dict()
```

with:

```python
        # Save parameters
        self._layer_names_override = layer_names_dict or {}
        self._tokenizer_class = tokenizer_class
        self.set_layer_names_dict()
```

- [ ] **Step 5: Update `set_layer_names_dict()` to apply the override**

In `vaayu/vaayu_base.py`, replace lines 55–60:

```python
    # customize layer names here
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                       'layer_prefix': 'model.layers',
                       'norm': 'model.norm',
                       'lm_head': 'lm_head',}
```

with:

```python
    # customize layer names here; subclasses or layer_names_dict kwarg override
    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }
        self.layer_names_dict.update(self._layer_names_override)
```

- [ ] **Step 6: Flip `get_use_better_transformer()` default to `False`**

In `vaayu/vaayu_base.py`, replace lines 307–308:

```python
    def get_use_better_transformer(self):
        return True
```

with:

```python
    def get_use_better_transformer(self):
        return False
```

- [ ] **Step 7: Update `get_tokenizer()` to check `_tokenizer_class`**

In `vaayu/vaayu_base.py`, replace lines 301–305:

```python
    # a chance to customize tokenizer
    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)
```

with:

```python
    # a chance to customize tokenizer
    def get_tokenizer(self, hf_token=None):
        if self._tokenizer_class is not None:
            return self._tokenizer_class.from_pretrained(
                self.model_local_path, use_fast=False, trust_remote_code=True
            )
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)
```

- [ ] **Step 8: Insert auto-detect call after `self.init_model()` (line 245)**

In `vaayu/vaayu_base.py`, replace line 245–246:

```python
        self.init_model()

        # get layer count:
```

with:

```python
        self.init_model()

        # fill in any layer name keys that don't resolve on this model (unknown architectures)
        _detected = _auto_detect_layer_names(self.model)
        for key, val in _detected.items():
            if val and not _layer_path_exists(self.model, self.layer_names_dict.get(key, "")):
                self.layer_names_dict[key] = val

        # get layer count:
```

- [ ] **Step 9: Run tests — expect all pass**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -m pytest tests/test_vaayu_base_kwargs.py tests/test_layer_detection.py -v
```

Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_vaayu_base_kwargs.py
git commit -m "feat: add layer_names_dict/tokenizer_class kwargs and auto-detect fallback to VaayuBaseModel"
```

---

## Task 3: Update `auto_model.py` — return `(class, kwargs)` directly

**Files:**
- Modify: `vaayu/auto_model.py`
- Create: `tests/test_auto_model_dispatch.py`

- [ ] **Step 1: Write failing dispatch tests**

Create `tests/test_auto_model_dispatch.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
import pytest


def _mock_config(arch: str):
    cfg = MagicMock()
    cfg.architectures = [arch]
    return cfg


def _dispatch(arch: str):
    from vaayu.auto_model import AutoModel
    with patch("vaayu.auto_model.AutoConfig.from_pretrained", return_value=_mock_config(arch)):
        return AutoModel.get_module_class("fake/path")


def test_qwen2_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("Qwen2ForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_qwen2_5_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("Qwen2_5ForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_qwen3_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("Qwen3ForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_qwen_v1_returns_vaayuqwen():
    from vaayu.vaayu_qwen import VaayuQWen
    cls, kwargs = _dispatch("QWenLMHeadModel")
    assert cls is VaayuQWen
    assert kwargs == {}


def test_baichuan_returns_base_with_tokenizer_class():
    from vaayu.vaayu_base import VaayuBaseModel
    from vaayu.tokenization_baichuan import BaichuanTokenizer
    cls, kwargs = _dispatch("BaichuanForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs.get("tokenizer_class") is BaichuanTokenizer


def test_chatglm_returns_vaayuchatglm():
    from vaayu.vaayu_chatglm import VaayuChatGLM
    cls, kwargs = _dispatch("ChatGLMModel")
    assert cls is VaayuChatGLM
    assert kwargs == {}


def test_mistral_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("MistralForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_mixtral_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("MixtralForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_internlm_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("InternLMForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_llama_returns_vaayullama2():
    from vaayu.vaayu import VaayuLlama2
    cls, kwargs = _dispatch("LlamaForCausalLM")
    assert cls is VaayuLlama2
    assert kwargs == {}


def test_unknown_arch_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("FutureMysteryModel")
    assert cls is VaayuBaseModel
    assert kwargs == {}
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -m pytest tests/test_auto_model_dispatch.py -v 2>&1 | head -40
```

Expected: `TypeError` — `get_module_class` currently returns a tuple of strings, not `(class, dict)`.

- [ ] **Step 3: Rewrite `auto_model.py`**

Replace the entire contents of `vaayu/auto_model.py` with:

```python
import pathlib
import subprocess
from transformers import AutoConfig
from sys import platform

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    from vaayu import VaayuLlamaMlx


def _check_nvme_scheduler(model_path: str) -> None:
    if not pathlib.Path(model_path).exists():
        return
    try:
        dev = subprocess.check_output(
            ["df", "--output=source", model_path], text=True
        ).strip().splitlines()[-1]
        nvme = subprocess.check_output(
            ["lsblk", "-no", "pkname", dev], text=True
        ).strip() or dev.replace("/dev/", "")
        if not nvme.startswith("nvme"):
            return
        nvme_char = nvme.split("n")[0]  # nvme0n1 → nvme0

        scheduler = pathlib.Path(f"/sys/block/{nvme}/queue/scheduler").read_text()
        fix_cmd = f"sudo bash scripts/vaayu_optimize_os.sh {model_path}  # one-time per machine"

        if "[none]" not in scheduler:
            print(f"[Vaayu Warning] {nvme} I/O scheduler is not optimized "
                  f"(current: {scheduler.strip()}).")
            print(f"  You are losing ~5% NVMe throughput.")
            print(f"  Fix (once): {fix_cmd}")
        else:
            print(f"[Vaayu] {nvme} I/O scheduler: optimized (none)")

        apst_candidates = [
            pathlib.Path("/sys/module/nvme_core/parameters/default_ps_max_latency_us"),
            pathlib.Path(f"/sys/class/nvme/{nvme_char}/power/ps_max_latency_us"),
        ]
        apst_path = next((p for p in apst_candidates if p.exists()), None)
        if apst_path:
            apst_val = apst_path.read_text().strip()
            if apst_val != "0":
                print(f"[Vaayu Warning] {nvme_char} APST (power management) is active "
                      f"(ps_max_latency_us={apst_val}).")
                print(f"  Drive enters micro-sleep during GPU compute phases — latency spikes over 80+ layers.")
                print(f"  Fix (once): {fix_cmd}")
            else:
                print(f"[Vaayu] {nvme_char} APST: optimized (disabled)")
    except Exception:
        pass


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """Return (model_class, extra_kwargs) for the given model path/repo."""
        from .vaayu_base import VaayuBaseModel
        from .vaayu import VaayuLlama2
        from .vaayu_chatglm import VaayuChatGLM
        from .vaayu_qwen import VaayuQWen
        from .tokenization_baichuan import BaichuanTokenizer

        if 'hf_token' in kwargs:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token']
            )
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )

        arch = config.architectures[0]

        if "Qwen2ForCausalLM" in arch or "Qwen2_5ForCausalLM" in arch or "Qwen3ForCausalLM" in arch:
            return VaayuBaseModel, {}
        elif "QWen" in arch:
            return VaayuQWen, {}
        elif "Baichuan" in arch:
            return VaayuBaseModel, {"tokenizer_class": BaichuanTokenizer}
        elif "ChatGLM" in arch:
            return VaayuChatGLM, {}
        elif "InternLM" in arch or "Mistral" in arch or "Mixtral" in arch:
            return VaayuBaseModel, {}
        elif "Gemma" in arch or "Llama" in arch:
            return VaayuLlama2, {}
        else:
            print(f"unknown architecture: {arch}, using VaayuBaseModel with auto-detect...")
            return VaayuBaseModel, {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if not is_on_mac_os:
            _check_nvme_scheduler(pretrained_model_name_or_path)

        if is_on_mac_os:
            return VaayuLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        model_cls, extra_kwargs = AutoModel.get_module_class(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        # user-supplied kwargs win over architecture defaults
        return model_cls(pretrained_model_name_or_path, *inputs, **{**extra_kwargs, **kwargs})
```

- [ ] **Step 4: Run dispatch tests — expect all pass**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -m pytest tests/test_auto_model_dispatch.py -v
```

Expected: 11 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add vaayu/auto_model.py tests/test_auto_model_dispatch.py
git commit -m "refactor: AutoModel.get_module_class returns (class, kwargs) directly"
```

---

## Task 4: Delete 5 dead subclass files and clean up `__init__.py`

**Files:**
- Delete: `vaayu/vaayu_qwen2.py`, `vaayu/vaayu_mistral.py`, `vaayu/vaayu_mixtral.py`, `vaayu/vaayu_internlm.py`, `vaayu/vaayu_baichuan.py`
- Modify: `vaayu/__init__.py`

- [ ] **Step 1: Delete the 5 files**

```bash
rm vaayu/vaayu_qwen2.py vaayu/vaayu_mistral.py vaayu/vaayu_mixtral.py vaayu/vaayu_internlm.py vaayu/vaayu_baichuan.py
```

- [ ] **Step 2: Update `vaayu/__init__.py`**

Replace the entire contents of `vaayu/__init__.py` with:

```python
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .vaayu_llama_mlx import VaayuLlamaMlx
    from .auto_model import AutoModel
else:
    from .vaayu import VaayuLlama2
    from .vaayu_chatglm import VaayuChatGLM
    from .vaayu_qwen import VaayuQWen
    from .vaayu_base import VaayuBaseModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException
```

- [ ] **Step 3: Verify no broken imports**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -c "from vaayu import AutoModel, VaayuBaseModel, VaayuChatGLM, VaayuQWen; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 4: Run full test suite**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add vaayu/__init__.py && git rm vaayu/vaayu_qwen2.py vaayu/vaayu_mistral.py vaayu/vaayu_mixtral.py vaayu/vaayu_internlm.py vaayu/vaayu_baichuan.py
git commit -m "refactor: delete 5 trivially-empty subclasses, clean up __init__.py"
```

---

## Task 5: Smoke test — verify AutoModel still routes correctly

No new code. Validates the full chain works end-to-end without loading a real large model.

- [ ] **Step 1: Run final import + dispatch smoke test**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -c "
from unittest.mock import patch, MagicMock
from vaayu.auto_model import AutoModel
from vaayu.vaayu_base import VaayuBaseModel
from vaayu.vaayu_chatglm import VaayuChatGLM
from vaayu.vaayu_qwen import VaayuQWen
from vaayu.vaayu import VaayuLlama2

cases = [
    ('Qwen2ForCausalLM', VaayuBaseModel),
    ('Qwen2_5ForCausalLM', VaayuBaseModel),
    ('Qwen3ForCausalLM', VaayuBaseModel),
    ('QWenLMHeadModel', VaayuQWen),
    ('BaichuanForCausalLM', VaayuBaseModel),
    ('ChatGLMModel', VaayuChatGLM),
    ('MistralForCausalLM', VaayuBaseModel),
    ('MixtralForCausalLM', VaayuBaseModel),
    ('InternLMForCausalLM', VaayuBaseModel),
    ('LlamaForCausalLM', VaayuLlama2),
    ('GemmaForCausalLM', VaayuLlama2),
    ('FutureMysteryModel', VaayuBaseModel),
]

for arch, expected_cls in cases:
    cfg = MagicMock()
    cfg.architectures = [arch]
    with patch('vaayu.auto_model.AutoConfig.from_pretrained', return_value=cfg):
        cls, _ = AutoModel.get_module_class('fake/path')
    status = '✅' if cls is expected_cls else '❌'
    print(f'{status} {arch} -> {cls.__name__} (expected {expected_cls.__name__})')
"
```

Expected: all lines print `✅`.

- [ ] **Step 2: Run full test suite one last time**

```bash
cd /Users/archish/Documents/ai-ml/Vaayu-veriants/Vaayu && python -m pytest tests/ -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: subclass consolidation complete — 5 files deleted, VaayuBaseModel extended with kwargs and auto-detect"
```

---

## Self-Review

**Spec coverage check:**
- ✅ `layer_names_dict` kwarg — Task 2 steps 3–5
- ✅ `tokenizer_class` kwarg — Task 2 steps 3–4, 7
- ✅ `_auto_detect_layer_names()` — Task 1
- ✅ `_layer_path_exists()` — Task 1
- ✅ Auto-detect inserted post-`init_model()` — Task 2 step 8
- ✅ `get_use_better_transformer()` default flipped — Task 2 step 6
- ✅ `AutoModel` returns `(class, kwargs)` — Task 3
- ✅ All 5 files deleted — Task 4 step 1
- ✅ `__init__.py` cleaned — Task 4 step 2
- ✅ All changes on new branch — Task 0
- ✅ `get_tokenizer()` checks `_tokenizer_class` — Task 2 step 7

**Type/name consistency:** `_layer_path_exists` and `_auto_detect_layer_names` defined in Task 1 and used in Tasks 2, 3 — names consistent throughout.

**Placeholders:** None.
