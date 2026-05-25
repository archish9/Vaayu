# Subclass Consolidation — Design Spec
Date: 2026-05-25

## Problem

Linux Vaayu has 7 architecture-specific subclasses of `VaayuBaseModel`. 5 of them are effectively empty — every subclass returns `False` from `get_use_better_transformer()` and most only override `get_generation_config()` with `return GenerationConfig()`, which the base class already falls back to. This means adding a new model architecture requires creating a new file, a new class, and a new dispatch entry in `AutoModel`, even when there is nothing to override.

The two genuinely-different subclasses (`VaayuQWen` v1 and `VaayuChatGLM`) override multiple methods with stateful, model-specific logic that cannot be cleanly kwarg-ized.

## Goal

Collapse the 5 trivial subclasses into `VaayuBaseModel` directly. Keep the 2 legitimate subclasses. Add auto-detect as a fallback for unknown future architectures so no new subclass is ever needed for a standard HF model.

## Approach

**Kwargs on VaayuBaseModel + auto-detect fallback.**

Add two new optional kwargs to `VaayuBaseModel.__init__`:
- `layer_names_dict: dict | None` — explicit layer name overrides, merged over Llama defaults
- `tokenizer_class: type | None` — custom tokenizer class (Baichuan's only real difference)

`AutoModel` passes the right kwargs instead of dispatching to a dedicated subclass.

After `init_model()`, run `_auto_detect_layer_names()` to fill in any layer name key that doesn't actually resolve on the loaded model. This covers future architectures without any user or developer action.

## Files Changed

### `vaayu/vaayu_base.py`

**`__init__` signature** — two new optional kwargs before existing ones:
```python
def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16,
             max_seq_len=512, layer_shards_saving_path=None, profiling_mode=False,
             compression=None, hf_token=None, prefetching=True, delete_original=False,
             use_torch_compile=False,
             layer_names_dict: dict | None = None,
             tokenizer_class=None):
```

Stored before `set_layer_names_dict()` call:
```python
self._layer_names_override = layer_names_dict or {}
self._tokenizer_class = tokenizer_class
self.set_layer_names_dict()
```

**`set_layer_names_dict()`** — applies `_layer_names_override` on top of Llama defaults:
```python
def set_layer_names_dict(self):
    self.layer_names_dict = {
        'embed': 'model.embed_tokens',
        'layer_prefix': 'model.layers',
        'norm': 'model.norm',
        'lm_head': 'lm_head',
    }
    self.layer_names_dict.update(self._layer_names_override)
```

**`get_use_better_transformer()`** — default flipped to `False`:
```python
def get_use_better_transformer(self):
    return False
```

**`get_tokenizer()`** — checks `_tokenizer_class` first:
```python
def get_tokenizer(self, hf_token=None):
    if self._tokenizer_class is not None:
        return self._tokenizer_class.from_pretrained(
            self.model_local_path, use_fast=False, trust_remote_code=True
        )
    if hf_token is not None:
        return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)
```

**Post-`init_model()` auto-detect** — inserted between `self.init_model()` and the layer count walk:
```python
self.init_model()
detected = _auto_detect_layer_names(self.model)
for key, val in detected.items():
    if val and not _layer_path_exists(self.model, self.layer_names_dict.get(key, "")):
        self.layer_names_dict[key] = val
```

**Module-level helpers** added above the class:
```python
def _layer_path_exists(model: nn.Module, path: str) -> bool:
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

### `vaayu/auto_model.py`

`get_module_class()` returns `(class, extra_kwargs: dict)` instead of `(module_str, class_str)`:

```python
@classmethod
def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
    # config loading unchanged
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
```

`from_pretrained()` merges kwargs:
```python
model_cls, extra_kwargs = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
return model_cls(pretrained_model_name_or_path, *inputs, **{**extra_kwargs, **kwargs})
```

### `vaayu/__init__.py`

Remove imports of the 5 deleted subclasses. Keep `VaayuQWen` and `VaayuChatGLM`.

## Files Deleted

| File | Reason |
|---|---|
| `vaayu/vaayu_qwen2.py` | Only overrides `get_use_better_transformer()` → False |
| `vaayu/vaayu_mistral.py` | Same + `get_generation_config()` no-op |
| `vaayu/vaayu_mixtral.py` | Same |
| `vaayu/vaayu_internlm.py` | Same |
| `vaayu/vaayu_baichuan.py` | Custom tokenizer only — replaced by `tokenizer_class` kwarg |

## Files Kept Unchanged

| File | Reason |
|---|---|
| `vaayu/vaayu_qwen.py` | QWen v1 legacy attention API (`layer_past`, NTK RoPE) |
| `vaayu/vaayu_chatglm.py` | Custom layer names, `kv_cache`, `rotary_pos_emb` API |
| `vaayu/tokenization_baichuan.py` | Still referenced by AutoModel for Baichuan dispatch |

## Invariants

- `get_generation_config()` in base already falls back to `GenerationConfig()` on exception — no behavior regression for Mistral/Mixtral/InternLM.
- `tokenizer_class` kwarg user-supplied kwargs take precedence over `extra_kwargs` in `from_pretrained()` merge (`{**extra_kwargs, **kwargs}`).
- Auto-detect only patches keys where the current path doesn't resolve — never overwrites a working explicit mapping.
- All changes land on a new branch. `main` untouched.

## Testing

- Load Qwen2.5 model via `AutoModel.from_pretrained()` — verify `VaayuBaseModel` is returned, layer names resolve correctly.
- Load Baichuan model — verify `BaichuanTokenizer` is used.
- Load a Mistral/Mixtral/InternLM model — verify `VaayuBaseModel` is returned, no crash.
- Load an unknown architecture — verify auto-detect fires and layer names are resolved from model structure.
- Load ChatGLM — verify `VaayuChatGLM` still returned, behavior unchanged.
- Load QWen v1 — verify `VaayuQWen` still returned, behavior unchanged.
