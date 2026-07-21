# Module-List VRAM Expert Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `_make_modulelist_sparse_forward` use the VRAM expert cache, cached `BC_LinearEXL3` objects, async miss loading, and disk telemetry — lifting MiniMax-M2.7 decode from 0.62 tok/s toward and past Qwen's 1.02 tok/s.

**Architecture:** A cached expert is a per-submodule state dict (`{submod_path: {tensor_name: gpu_tensor}, ..., "bc": {submod_path: qmatrix}}`) stored in the existing `ExpertVRAMCache`. `_inject_expert_weights` is changed to *return* that state instead of a bool, and a new `_attach_cached_expert` re-binds it on a hit. The per-expert teardown (`submod.bc = None` + `expert_mod.to("meta")`) runs unconditionally on both hit and miss — it detaches the live `nn.ModuleList` from the tensors without freeing them, which is what makes cache eviction actually reclaim VRAM and preserves the one-expert-at-a-time GPU residency bound. Separately, the fused path's miss-loading machinery is extracted into shared helpers so the module-list path gets async disk loads and `_expert_disk_s` telemetry without duplicating slab/pin logic.

**Tech Stack:** Python, PyTorch, exllamav3 (EXL3 kernels via pybind11), pytest, safetensors.

## Global Constraints

Copied from `docs/superpowers/specs/2026-07-21-modulelist-vram-expert-cache-design.md` and CLAUDE.md §0/§5. Every task's requirements implicitly include these.

- **No hardcoded model numbers.** No `62`, `256`, `94`, `128`, `8.9`, `11.28`, `4.40` literals in code. Layer count, expert count, top_k, expert bytes, working-set size all come from `self._moe_geometry` / config / the offset index. Only hardware limits (16GB VRAM, 32GB RAM) may be literals.
- **Do not regress the fused/Qwen path.** `_make_fused_sparse_forward_exl3` is the actively-tested path. Task 4 edits it; that task carries its own Qwen before/after run.
- **Do not regress the per-token GPU residency bound.** Load → inject → compute → detach, one expert at a time. Never inject all routed experts before computing any (that caused a real OOM-kill; see the comment at `vaayu/moe_patch.py:~636`).
- **Dense load+generate smoke test must stay green.**
- **Tests parametrized over both geometries** — a fix proven on one model only is not done.
- **Every task ends with `pytest tests/` green** (296 tests at time of writing) before its commit.
- Run tests with the project venv: `source venv/bin/activate` first.

---

### Task 1: `_inject_expert_weights` returns the state it built

Today `_inject_expert_weights` returns `True`/`False` and discards the GPU tensors it created. `build_exl3_qmatrix` likewise builds `gpu_tensors`, hands them to the C++ `BC_LinearEXL3`, and drops every Python reference — so the GPU bytes are invisible to any Python-side accounting. Nothing can be cached until both return their state.

**Files:**
- Modify: `vaayu/persist/loader_strategy.py:282-335` (`build_exl3_qmatrix`)
- Modify: `vaayu/moe_patch.py:418-486` (`_inject_expert_weights`)
- Modify: `vaayu/moe_patch.py:656` (the one call site)
- Test: `tests/test_moe_patch.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `build_exl3_qmatrix(mod, tensors_by_base, device, non_blocking=True) -> dict[str, torch.Tensor]` — now returns the `gpu_tensors` dict it built. Existing caller `EXL3LoaderStrategy.inject_to_gpu` ignores the return; behavior unchanged.
  - `_inject_expert_weights(expert_mod, raw, expert_idx, layer_name, device, dtype) -> dict | None` — returns the per-submodule state on success (shape below), `None` on failure. Replaces the old `bool`.

  State shape produced:
  ```python
  {
    "w1": {"trellis": tensor, "suh": tensor, "svh": tensor},  # per-submodule GPU tensors
    "w2": {...},
    "bc": {"w1": qmatrix_obj, "w2": qmatrix_obj},              # only for EXL3 submodules
  }
  ```
  Top-level keys are submodule paths exactly as `_EXPERT_PREFIX_RE` grouping produces them, plus the literal key `"bc"`. `ExpertVRAMCache._group_bytes` already skips `"bc"` and sums `.nbytes` over every other role's values, so this shape needs no cache change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_moe_patch.py`:

```python
def test_inject_expert_weights_returns_state_non_exl3():
    """_inject_expert_weights returns the per-submodule tensors it injected."""
    from vaayu.moe_patch import _inject_expert_weights

    expert = MockExpert(hidden=8)
    raw = {
        "layers.0.block_sparse_moe.experts.3.w1.weight": torch.randn(8, 8),
        "layers.0.block_sparse_moe.experts.3.w2.weight": torch.randn(8, 8),
    }
    state = _inject_expert_weights(expert, raw, 3, "layers.0", "cpu", torch.float32)

    assert state is not None
    assert set(state.keys()) == {"w1", "w2"}, f"unexpected roles: {list(state)}"
    assert state["w1"]["weight"].shape == (8, 8)
    # Raw tensors, not nn.Parameter: expert_mod.to("meta") rebinds Parameter.data
    # and must not disturb what the cache holds.
    assert not isinstance(state["w1"]["weight"], nn.Parameter)
    # And the weights really landed on the module.
    assert expert.w1.weight.shape == (8, 8)


def test_inject_expert_weights_returns_none_on_failure():
    """A submodule path that doesn't exist on the module yields None, not a partial dict."""
    from vaayu.moe_patch import _inject_expert_weights

    expert = MockExpert(hidden=8)
    raw = {"layers.0.block_sparse_moe.experts.3.nonexistent.weight": torch.randn(8, 8)}
    state = _inject_expert_weights(expert, raw, 3, "layers.0", "cpu", torch.float32)
    assert state is None


def test_inject_expert_weights_collects_bc_objects():
    """EXL3 submodules contribute their QMatrix under the 'bc' key and their GPU
    tensors under the submodule path."""
    from vaayu.moe_patch import _inject_expert_weights
    from vaayu.persist.loader_strategy import ExLlamaV3Linear

    class _FakeEXL3Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = ExLlamaV3Linear.__new__(ExLlamaV3Linear)
            nn.Module.__init__(self.w1)
            self.w1.bc = None
            self.w1.bias_param = None
            self.w1.weight_fallback = None
            self.w1.in_features = 8
            self.w1.out_features = 16

    expert = _FakeEXL3Expert()
    gpu_tensors = {"trellis": torch.randn(2, 4), "suh": torch.randn(8), "svh": torch.randn(16)}
    sentinel_bc = object()

    def _fake_build(mod, bases, device, non_blocking=True):
        mod.bc = sentinel_bc
        return gpu_tensors

    raw = {
        "layers.0.block_sparse_moe.experts.0.w1.trellis": gpu_tensors["trellis"],
        "layers.0.block_sparse_moe.experts.0.w1.suh": gpu_tensors["suh"],
        "layers.0.block_sparse_moe.experts.0.w1.svh": gpu_tensors["svh"],
    }
    with mock_patch("vaayu.persist.loader_strategy.build_exl3_qmatrix", _fake_build):
        state = _inject_expert_weights(expert, raw, 0, "layers.0", "cpu", torch.float32)

    assert state is not None
    assert state["bc"]["w1"] is sentinel_bc
    assert state["w1"] is gpu_tensors
```

`mock_patch` and `nn` are already imported at the top of this file; `MockExpert` is defined at `tests/test_moe_patch.py:9`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "inject_expert_weights" -v
```

Expected: FAIL — `assert state is not None` fails because the function currently returns `True` (a bool has no `.keys()`, so the following assert errors) and `test_inject_expert_weights_returns_none_on_failure` fails because it currently returns `False`.

- [ ] **Step 3: Make `build_exl3_qmatrix` return its GPU tensors**

In `vaayu/persist/loader_strategy.py`, change the signature line and add the return. The function body is otherwise untouched:

```python
def build_exl3_qmatrix(
    mod: "ExLlamaV3Linear", tensors_by_base: dict, device: str, non_blocking: bool = True
) -> dict:
```

Append to the end of the function, after the `mod.bc = exl3_ext.exllamav3_ext.BC_LinearEXL3(...)` assignment:

```python
    # Returned so callers can keep their own reference for byte accounting: the
    # QMatrix holds these tensors C++-side, where ExpertVRAMCache._group_bytes
    # cannot see them. inject_to_gpu ignores this; the module-list expert cache
    # (moe_patch.py) does not.
    return gpu_tensors
```

Also extend the docstring's first paragraph with one sentence:

```
    Returns the dict of GPU-resident quant tensors it built, so a caller that
    caches the QMatrix can also hold (and account for) the memory behind it.
```

- [ ] **Step 4: Make `_inject_expert_weights` return the state**

In `vaayu/moe_patch.py`, change the signature's return annotation:

```python
def _inject_expert_weights(
    expert_mod: nn.Module,
    raw: dict,
    expert_idx: int,
    layer_name: str,
    device: str,
    dtype,
) -> dict | None:
```

Replace the docstring's `Returns` line:

```
    Returns the per-submodule state it injected on success, None if any submodule
    failed (expert skipped). The returned dict is
    {submod_path: {tensor_name: gpu_tensor}, ..., "bc": {submod_path: qmatrix}} —
    the exact shape ExpertVRAMCache stores, whose _group_bytes skips the "bc" key.
```

Then in the body, add a state accumulator before the loop over `groups.items()`:

```python
    state: dict = {}
    bc_objs: dict = {}
```

In the EXL3 branch, capture both outputs:

```python
            if isinstance(submod, ExLlamaV3Linear) and any(
                b in _EXL3_QTENSOR_KEYS for b in bases
            ):
                # non_blocking=False: these tensors come from disk loads /
                # the RAM cache, not guaranteed page-locked — see
                # build_exl3_qmatrix's docstring for why async here leaks host RAM.
                state[submod_path] = build_exl3_qmatrix(
                    submod, bases, device, non_blocking=False)
                bc_objs[submod_path] = submod.bc
                continue
```

In the plain-tensor branch, record the raw device tensor (never the `nn.Parameter` — `expert_mod.to("meta")` rebinds `Parameter.data`, and the cache must be unaffected by that):

```python
            for base, tensor in bases.items():
                t = tensor.to(device, dtype=dtype, non_blocking=False)
                state.setdefault(submod_path, {})[base] = t
                if isinstance(submod, ExLlamaV3Linear):
```

(the rest of that branch is unchanged)

Replace the two terminal returns:

```python
        except Exception as e:
            _log.warning(
                f"Expert weight inject failed: {submod_path} ({list(bases)}): {e}"
            )
            return None
    if bc_objs:
        state["bc"] = bc_objs
    return state
```

- [ ] **Step 5: Update the single call site**

At `vaayu/moe_patch.py:656`, inside `_make_modulelist_sparse_forward`:

```python
            state = _inject_expert_weights(expert_mod, raw, eidx, layer_name, device, dtype)
            if not state:
                continue
```

(`not state` covers both `None` and an empty dict — an expert whose `raw` matched no submodule is as unusable as a failed one.)

- [ ] **Step 6: Run the new tests and the full suite**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "inject_expert_weights" -v
pytest tests/ -q
```

Expected: the three new tests PASS; full suite PASS with no new failures.

- [ ] **Step 7: Commit**

```bash
git add vaayu/moe_patch.py vaayu/persist/loader_strategy.py tests/test_moe_patch.py
git commit -m "refactor(moe): _inject_expert_weights returns injected state

build_exl3_qmatrix now returns its gpu_tensors dict; _inject_expert_weights
returns {submod_path: tensors, ..., 'bc': {submod_path: qmatrix}} instead of a
bool. The QMatrix holds its tensors C++-side where ExpertVRAMCache._group_bytes
cannot see them, so a cache entry must carry them explicitly or the byte budget
is unenforceable. No behavior change yet — the call site only tests truthiness.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `_attach_cached_expert` — the inverse of injection

**Files:**
- Modify: `vaayu/moe_patch.py` (add function immediately after `_inject_expert_weights`, before `patch_moe_blocks`)
- Test: `tests/test_moe_patch.py`

**Interfaces:**
- Consumes: the state dict shape produced by `_inject_expert_weights` (Task 1).
- Produces: `_attach_cached_expert(expert_mod: nn.Module, entry: dict) -> bool` — re-binds a cached entry onto a live expert module. Returns `True` on success, `False` on any structural mismatch so the caller can fall through to a normal load.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_moe_patch.py`:

```python
def test_attach_cached_expert_round_trips_non_exl3():
    """inject -> detach to meta -> attach restores usable weights."""
    from vaayu.moe_patch import _inject_expert_weights, _attach_cached_expert

    expert = MockExpert(hidden=8)
    w1 = torch.randn(8, 8)
    raw = {
        "layers.0.block_sparse_moe.experts.3.w1.weight": w1,
        "layers.0.block_sparse_moe.experts.3.w2.weight": torch.randn(8, 8),
    }
    entry = _inject_expert_weights(expert, raw, 3, "layers.0", "cpu", torch.float32)
    assert entry is not None

    expert.to("meta")
    assert expert.w1.weight.is_meta

    assert _attach_cached_expert(expert, entry) is True
    assert not expert.w1.weight.is_meta
    assert torch.equal(expert.w1.weight.detach(), w1)
    # Attaching must not consume the entry — it is reused on every later hit.
    assert torch.equal(entry["w1"]["weight"], w1)


def test_attach_cached_expert_restores_bc():
    """A cached QMatrix is re-bound to the live submodule."""
    from vaayu.moe_patch import _attach_cached_expert
    from vaayu.persist.loader_strategy import ExLlamaV3Linear

    class _FakeEXL3Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = ExLlamaV3Linear.__new__(ExLlamaV3Linear)
            nn.Module.__init__(self.w1)
            self.w1.bc = None

    expert = _FakeEXL3Expert()
    sentinel_bc = object()
    entry = {"w1": {"trellis": torch.randn(2, 4)}, "bc": {"w1": sentinel_bc}}

    assert _attach_cached_expert(expert, entry) is True
    assert expert.w1.bc is sentinel_bc


def test_attach_cached_expert_returns_false_on_mismatch():
    """A submodule path absent from the live module is reported, not raised."""
    from vaayu.moe_patch import _attach_cached_expert

    expert = MockExpert(hidden=8)
    entry = {"nonexistent": {"weight": torch.randn(8, 8)}}
    assert _attach_cached_expert(expert, entry) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "attach_cached_expert" -v
```

Expected: FAIL with `ImportError: cannot import name '_attach_cached_expert' from 'vaayu.moe_patch'`.

- [ ] **Step 3: Implement `_attach_cached_expert`**

Insert into `vaayu/moe_patch.py` directly after `_inject_expert_weights` ends (just before `def patch_moe_blocks`):

```python
def _attach_cached_expert(expert_mod: nn.Module, entry: dict) -> bool:
    """
    Re-bind a VRAM-cached expert (an entry built by _inject_expert_weights) onto a
    live expert module. The inverse of injection, minus the disk read and the H2D
    copy — that saving is the entire point of the VRAM expert cache.

    Returns True if every role in the entry was attached, False on any structural
    mismatch (caller falls through to a normal load rather than computing with a
    half-attached module).

    Layout-agnostic on purpose: it walks whatever submodule paths the entry
    carries, so it serves EXL3 and non-EXL3 module-list experts alike without
    knowing anything about w1/w2/w3 or a gated-MLP shape.
    """
    from .persist.loader_strategy import ExLlamaV3Linear

    bc_objs = entry.get("bc") or {}
    for submod_path, tensors in entry.items():
        if submod_path == "bc":
            continue
        try:
            submod = expert_mod
            for part in submod_path.split("."):
                submod = getattr(submod, part)

            bc = bc_objs.get(submod_path)
            if bc is not None:
                # EXL3: the QMatrix already wraps the cached GPU tensors, so
                # re-binding it is the whole attach — no Parameter to restore.
                submod.bc = bc
                continue

            for base, t in tensors.items():
                if isinstance(submod, ExLlamaV3Linear):
                    if base == "weight":
                        submod.weight_fallback = t
                    elif base == "bias":
                        submod.bias_param = nn.Parameter(t, requires_grad=False)
                    else:
                        setattr(submod, base, nn.Parameter(t, requires_grad=False))
                else:
                    setattr(submod, base, nn.Parameter(t, requires_grad=False))
        except Exception as e:
            _log.warning(f"Cached expert attach failed: {submod_path}: {e}")
            return False
    return True
```

- [ ] **Step 4: Run the new tests and the full suite**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "attach_cached_expert" -v
pytest tests/ -q
```

Expected: three new tests PASS; full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/moe_patch.py tests/test_moe_patch.py
git commit -m "feat(moe): add _attach_cached_expert for module-list experts

Inverse of _inject_expert_weights: re-binds a cached entry's QMatrix or
Parameters onto a live expert module, skipping the disk read and H2D copy.
Walks whatever submodule paths the entry carries, so it serves EXL3 and
non-EXL3 module-list layouts without assuming a gated-MLP shape.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Wire the VRAM cache into the module-list forward

The payload task. After this, MiniMax stops re-reading and re-uploading every expert on every token.

**Files:**
- Modify: `vaayu/moe_patch.py:590-695` (`_make_modulelist_sparse_forward`)
- Modify: `tests/test_moe_patch.py:19-52` (mock classes take `num_experts`)
- Test: `tests/test_moe_patch.py`

**Interfaces:**
- Consumes: `_inject_expert_weights(...) -> dict | None` (Task 1), `_attach_cached_expert(expert_mod, entry) -> bool` (Task 2), `ExpertVRAMCache.get/put/has` (`vaayu/persist/expert_cache.py:437`).
- Produces: no new public names. `_make_modulelist_sparse_forward`'s signature is unchanged.

**Critical invariant — always detach.** The teardown in the `finally` block runs on hit *and* miss, unchanged. Detach is not free: on a hit the cache entry holds the surviving references, so this drops only the *module's* references. Leaving weights attached to `experts_list[eidx]` would mean an entry evicted from the cache is still referenced by the live `nn.ModuleList`, so eviction would reclaim nothing and the byte budget would stop corresponding to real VRAM. It also keeps the one-expert-at-a-time GPU residency bound exactly as it is today. This is why there is deliberately **no `_owned_by_cache` branch here**, unlike the fused path at `vaayu/moe_patch.py:1310` — there, the local variables are the only references; here the ModuleList is a second reference holder, and that is the hazard.

- [ ] **Step 1: Make the mock MoE model take an expert count**

In `tests/test_moe_patch.py`, thread `num_experts` through the three mock classes (additive defaults — every existing call site keeps working):

```python
class MockTransformerLayer(nn.Module):
    def __init__(self, hidden=8, num_experts=8):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.block_sparse_moe = MockMoEBlock(hidden, num_experts)
```

```python
class MockModel(nn.Module):
    def __init__(self, num_layers=3, hidden=8, num_experts=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, hidden)
        self.layers = nn.ModuleList(
            [MockTransformerLayer(hidden, num_experts) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden)
```

`MockMoEBlock` already accepts `num_experts` (`tests/test_moe_patch.py:20`); no change there.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_moe_patch.py`. The two parametrizations stand in for the two on-device geometries: `(8, 2)` is Qwen-like (narrow routing over a smaller expert pool), `(16, 8)` is MiniMax-like (wide routing over a larger pool).

```python
# ── Module-list VRAM expert cache ─────────────────────────────────────────────

def _patch_modulelist_with_vram(tmp_path, num_experts, top_k, vram_budget=10**9):
    """Build a patched module-list MoE block with a VRAM expert cache attached.
    Returns (model, vaayu_ref, vram_cache)."""
    from vaayu.persist.expert_cache import ExpertVRAMCache

    model = MockModel(num_layers=1, hidden=8, num_experts=num_experts)
    _make_checkpoint(tmp_path, model)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=100 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()
    vram = ExpertVRAMCache(budget_bytes=vram_budget)
    vaayu_ref._expert_vram_cache = vram
    vaayu_ref._vram_cacheable_layers = None

    patch_moe_blocks(
        model=model,
        expert_cache=cache,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=top_k,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
    )
    return model, vaayu_ref, vram


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_second_pass_hits_vram_cache(tmp_path, num_experts, top_k):
    """Identical routing on a second forward must be served from the VRAM cache."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    moe.forward(hidden)                      # pass 1: all misses, populates
    assert vram.hits == 0
    assert vram.current_usage > 0, "nothing cached after first pass"
    misses_after_first = vram.misses
    assert misses_after_first > 0

    moe.forward(hidden)                      # pass 2: same routing -> hits
    assert vram.hits == misses_after_first, (
        f"expected {misses_after_first} hits on identical routing, got {vram.hits}")
    assert vram.misses == misses_after_first, "second pass should add no misses"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_vram_hit_skips_reinjection(tmp_path, num_experts, top_k):
    """A VRAM hit must not re-read from disk or re-inject weights."""
    import vaayu.moe_patch as mp

    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    moe.forward(hidden)

    real_inject = mp._inject_expert_weights
    calls = []

    def _counting_inject(*a, **kw):
        calls.append(1)
        return real_inject(*a, **kw)

    with mock_patch.object(mp, "_inject_expert_weights", _counting_inject):
        moe.forward(hidden)

    assert calls == [], f"re-injected {len(calls)} experts that were VRAM-resident"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_detaches_experts_after_every_pass(tmp_path, num_experts, top_k):
    """Routed experts return to meta after a hit as well as a miss — otherwise the
    live ModuleList keeps evicted entries alive and eviction reclaims nothing."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    for _ in range(2):                       # pass 1 = miss, pass 2 = hit
        moe.forward(hidden)
        for eidx in vaayu_ref._expert_history[0]:
            assert moe.experts[eidx].w1.weight.is_meta, (
                f"expert {eidx} still holds real weights after forward")


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_vram_byte_accounting_matches_tensors(tmp_path, num_experts, top_k):
    """current_usage equals the summed nbytes of every non-'bc' role."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    expected = 0
    for entry in vram.cache.values():
        for role, tensors in entry.items():
            if role == "bc":
                continue
            expected += sum(t.nbytes for t in tensors.values())

    assert expected > 0
    assert vram.current_usage == expected


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_skips_vram_cache_during_prefill(tmp_path, num_experts, top_k):
    """Prefill sweeps ~every expert once; caching there only churns the LRU."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref._moe_prefill_pass = True

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 4, 8))

    assert vram.current_usage == 0
    assert vram.hits == 0 and vram.misses == 0, "cache was consulted during prefill"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_respects_vram_cacheable_layers(tmp_path, num_experts, top_k):
    """A layer outside the cacheable prefix must not touch the VRAM cache."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref._vram_cacheable_layers = set()   # layer 0 excluded

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    assert vram.current_usage == 0
    assert vram.hits == 0 and vram.misses == 0


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_output_unchanged_by_vram_cache(tmp_path, num_experts, top_k):
    """Caching is a performance change only — a hit must produce the same numbers
    as the miss that populated it."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    out1, _ = moe.forward(hidden)
    out2, _ = moe.forward(hidden)
    assert torch.allclose(out1, out2, atol=1e-6)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "modulelist" -v
```

Expected: `test_modulelist_second_pass_hits_vram_cache` FAILs on `assert vram.current_usage > 0` (the forward never consults the cache, so it stays empty). `test_modulelist_vram_hit_skips_reinjection` FAILs on `assert calls == []`.

- [ ] **Step 4: Rewrite the expert loop**

In `vaayu/moe_patch.py`, inside `_make_modulelist_sparse_forward`. Insert the gating block immediately after `_idx_cpu = expert_indices.cpu()` and before `for eidx in unique_experts:`:

```python
        # VRAM expert cache (Fix #7, ported from the fused path). Same two gates:
        # prefill routes ~every expert once, so caching there only churns the LRU;
        # and _vram_cacheable_layers restricts to the layer prefix that fits when
        # the budget can't hold a whole token's working set (None = no restriction).
        _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
        if _vram is not None and getattr(vaayu_ref, "_moe_prefill_pass", False):
            _vram = None
        _vram_layers = getattr(vaayu_ref, '_vram_cacheable_layers', None)
        if _vram is not None and _vram_layers is not None and layer_idx not in _vram_layers:
            _vram = None
```

Replace the body of the `for eidx in unique_experts:` loop, from `cached = expert_cache.get(...)` down to and including `if not ok: continue`, with:

```python
        for eidx in unique_experts:
            expert_mod = experts_list[eidx]

            entry = _vram.get(layer_idx, eidx) if _vram is not None else None
            if entry is not None and not _attach_cached_expert(expert_mod, entry):
                entry = None      # structural mismatch — fall through to a real load

            if entry is None:
                cached = expert_cache.get(layer_idx, eidx)
                if cached is not None:
                    raw = cached
                else:
                    metas = offset_index.get_expert_offsets(layer_name, eidx)
                    if metas is None:
                        print(f"[Vaayu-MoE] warn: no offset for {layer_name} expert {eidx}")
                        continue
                    raw = load_expert_sparse(shard_file, metas)
                    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                        expert_cache.put(layer_idx, eidx, raw)

                entry = _inject_expert_weights(expert_mod, raw, eidx, layer_name, device, dtype)
                if not entry:
                    continue
                if _vram is not None:
                    # Return value deliberately ignored: unlike the fused path there
                    # is no _owned_by_cache branch here. The teardown below must run
                    # whether or not the cache accepted the entry — see its comment.
                    _vram.put(layer_idx, eidx, entry)
```

The `try: ... finally: ...` compute-and-teardown block that follows is unchanged except for its comment. Replace the teardown comment with:

```python
            finally:
                try:
                    # Detach this expert from the live module — ALWAYS, hit or miss.
                    # ExLlamaV3Linear's QMatrix (.bc) is a C++ object PyTorch's GC
                    # does not manage, and .to("meta") only clears real torch
                    # parameters/buffers, so both refs must be dropped explicitly.
                    #
                    # On a VRAM-cache hit this frees nothing: the cache entry holds
                    # the surviving references. That is exactly the point — if the
                    # weights stayed bound to experts_list[eidx], an entry the LRU
                    # evicted would still be referenced by the live ModuleList, so
                    # eviction would reclaim no VRAM and the byte budget would stop
                    # matching reality. Detaching also keeps peak GPU residency at
                    # one uncached expert regardless of routing width (the bound
                    # whose absence previously caused an OOM-kill).
                    #
                    # try/finally so a compute exception still detaches before the
                    # exception propagates.
                    for submod in expert_mod.modules():
                        if isinstance(submod, ExLlamaV3Linear):
                            submod.bc = None
                            submod.weight_fallback = None
                    expert_mod.to("meta")
                except Exception as e:
                    _log.warning(f"Failed to unload expert {eidx} to meta: {e}")
```

Delete the now-orphaned `expert_mod = experts_list[eidx]` line that used to sit between the load and the inject (it moved to the top of the loop).

- [ ] **Step 5: Run the new tests and the full suite**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "modulelist" -v
pytest tests/ -q
```

Expected: all 14 parametrized module-list tests PASS; full suite PASS.

- [ ] **Step 6: Dense smoke test**

Confirm the non-MoE path is untouched:

```bash
source venv/bin/activate
pytest tests/ -q -k "dense"
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add vaayu/moe_patch.py tests/test_moe_patch.py
git commit -m "perf(moe): use the VRAM expert cache in the module-list path

MiniMax-M2.7 is EXL3 but module-list layout, so it dispatched to the naive
forward and never consulted the 9.7GB VRAM expert cache (vram=0hit/0miss/0evict
while the cache sat allocated and idle). Each token re-read and re-uploaded
every routed expert and rebuilt every QMatrix.

The per-expert detach now runs on hit as well as miss. It frees nothing on a
hit — the cache entry holds the references — but without it an evicted entry
would still be referenced by the live ModuleList, so eviction would reclaim no
VRAM. This also preserves the one-expert-at-a-time GPU residency bound.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 8: First on-device measurement**

This is the change the whole plan exists for; measure before going further.

Edit `test_generate.py`: set `MAX_NEW_TOKENS = 256` (already 256) and bump `SESSION_ID` to `"minimax-256-vram-t3"` — a stale session id would hit the prefix cache and skip prefill entirely, invalidating TTFT.

```bash
source venv/bin/activate
python -u test_generate.py > run-task3.log 2>&1
grep -E "vram=|expert_cache=|Decode throughput|Time to first token" run-task3.log
```

Baseline to beat: **0.62 tok/s**, TTFT 53.1s, `vram=0hit/0miss/0evict`.
Record the numbers in the commit message of Task 6 or in a follow-up note. Do not treat a <10% delta from one sample as a win — run-to-run variance on this stack is real (TTFT 43.0-53.7 observed within one session).

---

### Task 4: Extract the fused path's miss-loading machinery

Behavior-preserving refactor of the actively-tested Qwen path, so the module-list path can reuse ~60 lines of slab/pin/arena logic instead of duplicating it. **This is the one task that can regress Qwen** — it lands alone, with its own verification run.

**Files:**
- Modify: `vaayu/moe_patch.py:1040-1108` (extract), `vaayu/moe_patch.py:1128-1147` (extract), new module-level helpers above `_make_fused_sparse_forward_exl3`
- Test: `tests/test_moe_patch.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  ```python
  def _dispatch_expert_misses(
      unique_experts, *, layer_idx, layer_name, shard_file, offset_index,
      expert_cache, vaayu_ref, staged_layer=None,
  ) -> tuple[dict, list, set]:
      """Returns (expert_tensors, miss_futures, pinned_keys)."""

  def _iter_expert_loads(expert_tensors, miss_futures, vaayu_ref, mp=None):
      """Generator of (eidx, raw): RAM/staged hits first, then disk misses in
      completion order. Accumulates true disk stall into vaayu_ref._expert_disk_s
      (and mp['disk_load'] when profiling)."""
  ```

- [ ] **Step 1: Write the characterization test**

This test must pass *before and after* the extraction — it pins the behavior the refactor may not change. Append to `tests/test_moe_patch.py`:

```python
def test_dispatch_expert_misses_splits_hits_and_misses(tmp_path):
    """RAM-cached experts come back inline; uncached ones become futures."""
    from vaayu.moe_patch import _dispatch_expert_misses, _iter_expert_loads

    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()

    cache.put(0, 1, _make_mock_exl3_raw("layers.0", 1, in_dim=8, out_dim=16))

    expert_tensors, futures, pinned = _dispatch_expert_misses(
        [1, 2], layer_idx=0, layer_name="layers.0",
        shard_file=str(tmp_path / "layers.0.safetensors"),
        offset_index=idx, expert_cache=cache, vaayu_ref=vaayu_ref,
    )

    assert 1 in expert_tensors, "RAM-cached expert should be an inline hit"
    assert 2 not in expert_tensors, "uncached expert should be dispatched as a future"
    assert len(futures) == 1

    seen = dict(_iter_expert_loads(expert_tensors, futures, vaayu_ref))
    assert set(seen) == {1, 2}
    assert vaayu_ref._expert_disk_s >= 0.0, "disk stall telemetry not accumulated"
```

`_make_fused_exl3_checkpoint` is defined at `tests/test_moe_patch.py:~570` and `_make_mock_exl3_raw` above it.

- [ ] **Step 2: Run to verify it fails**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "dispatch_expert_misses" -v
```

Expected: FAIL with `ImportError: cannot import name '_dispatch_expert_misses'`.

- [ ] **Step 3: Capture the Qwen baseline before touching the fused path**

Point `MODEL_PATH` in `test_generate.py` at the Qwen reference already noted there (`/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw`), bump `SESSION_ID` to `"qwen-256-pre-extract"`, and run:

```bash
source venv/bin/activate
python -u test_generate.py > run-qwen-before.log 2>&1
grep -E "Decode throughput|Time to first token|expert_cache=|vram=" run-qwen-before.log
```

Expected: ~1.02 tok/s. Record it — Step 7 compares against this exact run, not against the number in this document.

- [ ] **Step 4: Extract the two helpers**

Add above `def _make_fused_sparse_forward_exl3` in `vaayu/moe_patch.py`. The bodies are the existing code moved verbatim, with closure variables promoted to keyword arguments:

```python
def _dispatch_expert_misses(
    unique_experts,
    *,
    layer_idx: int,
    layer_name: str,
    shard_file: str,
    offset_index,
    expert_cache,
    vaayu_ref,
    staged_layer=None,
) -> tuple[dict, list, set]:
    """
    Split this layer's routed experts into RAM/staged hits and disk misses, submit
    the misses to the persistent pool, and return without waiting — so the caller
    computes hits while the reads are in flight.

    Returns (expert_tensors, miss_futures, pinned_keys). The caller MUST unpin
    every key in pinned_keys in a finally block; a leaked pin permanently blocks
    that key's eviction (a slow cache-capacity leak).

    Shared by the fused-EXL3 and module-list forwards: the slab/pin/arena
    discipline below is subtle enough that a second copy would rot.
    """
    from .persist.safetensor_model_persister import load_expert_sparse, load_expert_into_slab

    _do_pin = expert_cache.arena is not None and not getattr(
        vaayu_ref, "_moe_prefill_pass", False)
    pinned_keys: set = set()
    expert_tensors: dict = {}
    misses: list = []

    for eidx in unique_experts:
        if staged_layer is not None and eidx in staged_layer:
            expert_tensors[eidx] = staged_layer[eidx]
            continue
        cached = expert_cache.get(layer_idx, eidx, pin=_do_pin)
        if cached is not None:
            expert_tensors[eidx] = cached
            if _do_pin:
                pinned_keys.add((layer_idx, eidx))
        else:
            misses.append(eidx)

    # Pin misses BEFORE dispatching their loads: the key isn't cached yet, so
    # it can't be evicted until put_slab() inserts it — at which point the pin
    # is already set and the evictor will skip it.
    if _do_pin:
        for _e in misses:
            expert_cache.pin(layer_idx, _e)
            pinned_keys.add((layer_idx, _e))

    miss_futures = []
    if misses:
        def _load_one(eidx: int):
            metas = offset_index.get_expert_offsets(layer_name, eidx)
            if metas is None:
                return eidx, None
            # Prefill: transient load, never cached (routes ~all experts
            # once; caching would clone content decode replaces).
            if getattr(vaayu_ref, "_moe_prefill_pass", False):
                return eidx, load_expert_sparse(shard_file, metas)
            if expert_cache.arena is None:
                # No arena (e.g. compute_slab_bytes()==0 or CUDA-less unit
                # tests constructing ExpertRAMCache directly) — legacy
                # clone path, still cached.
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
            # Zero-clone decode path: io_uring reads straight into a pinned
            # slab; the cache adopts views (no clone anywhere disk→RAM→GPU).
            slab_idx = expert_cache.acquire_slab()
            if slab_idx is None:
                # arena exhausted and nothing evictable — legacy fallback
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
            slab = expert_cache.arena.tensor(slab_idx)
            try:
                # tensor_align=False: decode's cached experts are only ever
                # consumed via per-component .to(device) copies (always
                # naturally aligned on a fresh cross-device copy), never via
                # views_from_packed_copy — skip the scratch-pool repack, which
                # regressed decode throughput when applied unconditionally.
                views = load_expert_into_slab(
                    shard_file, metas, slab,
                    allow_direct=expert_cache.arena.pin,
                    tensor_align=False,
                )
            except Exception:
                expert_cache.arena.release(slab_idx)
                raise
            expert_cache.put_slab(layer_idx, eidx, views, slab_idx)
            return eidx, views

        # Submit and DON'T wait: hit experts compute while these reads run.
        miss_futures = [vaayu_ref._moe_executor.submit(_load_one, e) for e in misses]

    return expert_tensors, miss_futures, pinned_keys


def _iter_expert_loads(expert_tensors: dict, miss_futures: list, vaayu_ref, mp=None):
    """Yield (eidx, raw): cache hits immediately (compute overlaps the in-flight
    miss reads), then misses in completion order. Blocking time in the wait() is
    the true disk stall and is accumulated into vaayu_ref._expert_disk_s
    (telemetry: stall, not total read time)."""
    for _e, _raw in expert_tensors.items():
        yield _e, _raw
    _pending = set(miss_futures)
    while _pending:
        _w0 = time.perf_counter()
        _done, _pending = _futures_wait(_pending, return_when=FIRST_COMPLETED)
        _stall = time.perf_counter() - _w0
        vaayu_ref._expert_disk_s = getattr(vaayu_ref, '_expert_disk_s', 0.0) + _stall
        if mp is not None:
            mp['disk_load'] += _stall
        for _f in _done:
            _e, _raw = _f.result()
            if _raw is not None:
                yield _e, _raw
```

Note the one intentional difference from the original iterator: it took `unique_experts` and looked each up in `expert_tensors`; this iterates `expert_tensors` directly. Same set, same hits-before-misses ordering, one less dict lookup per expert.

- [ ] **Step 5: Rewrite the fused path to call them**

In `_make_fused_sparse_forward_exl3`, replace lines `1040-1108` (the hit/miss split, the pin loop, the `_load_one` definition and its `submit` calls) with:

```python
        expert_tensors, _miss_futures, _pinned_keys = _dispatch_expert_misses(
            unique_experts,
            layer_idx=layer_idx, layer_name=layer_name, shard_file=shard_file,
            offset_index=offset_index, expert_cache=expert_cache,
            vaayu_ref=vaayu_ref, staged_layer=_staged_layer,
        )
```

Delete the local `def _iter_experts_hits_then_misses():` (lines 1128-1147) and change its single call site in the compute loop:

```python
            for eidx, raw in _iter_expert_loads(expert_tensors, _miss_futures, vaayu_ref, _mp if _prof else None):
```

Remove any imports and locals in `_make_fused_sparse_forward_exl3` that only the deleted code used (`misses`, `_do_pin`) — but leave `_pinned_keys` in use by the existing `finally` unpin loop, which is unchanged.

- [ ] **Step 6: Run the tests**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -v
pytest tests/ -q
```

Expected: the new characterization test PASSes, and every pre-existing EXL3 fused test (`test_exl3_forward_handles_cache_miss_via_persistent_executor`, `test_exl3_forward_skips_ram_cache_put_during_prefill`, `test_exl3_forward_source_overlaps_disk_and_compute`, `test_exl3_vram_hit_reuses_bc_objects`) still PASSes unchanged. Full suite PASS.

- [ ] **Step 7: Verify Qwen did not regress**

Same config as Step 3, fresh session id:

```bash
source venv/bin/activate
# test_generate.py: MODEL_PATH = Qwen, SESSION_ID = "qwen-256-post-extract"
python -u test_generate.py > run-qwen-after.log 2>&1
grep -E "Decode throughput|Time to first token|expert_cache=|vram=" run-qwen-after.log
```

Expected: decode tok/s within run-to-run variance of Step 3's number (~1.02), VRAM cache hit rate in the same 39-69% band. **If Qwen regressed, revert this task and give the module-list path its own copy of the loader rather than shipping a fused-path regression.**

- [ ] **Step 8: Commit**

```bash
git add vaayu/moe_patch.py tests/test_moe_patch.py
git commit -m "refactor(moe): extract expert miss dispatch + load iterator

_dispatch_expert_misses and _iter_expert_loads lift the slab/pin/arena
discipline and the hits-before-misses ordering out of the fused-EXL3 forward so
the module-list path can reuse them instead of carrying a second copy that
would rot. Behavior-preserving; Qwen decode verified unchanged before and after.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Async miss loading + disk telemetry in the module-list path

Today the module-list forward reads misses serially and inline, and never accumulates `_expert_disk_s` — so its disk time is invisible, hidden inside `compute=1.6s`. That is why the brief could not attribute the cost.

The disk futures are **CPU-side** (they land in the RAM cache / slab arena, bounded by `_expert_budget`), so they are separable from the GPU residency bound: dispatch all misses up front, consume hits first, but keep GPU injection strictly one expert at a time.

**Files:**
- Modify: `vaayu/moe_patch.py` (`_make_modulelist_sparse_forward`)
- Test: `tests/test_moe_patch.py`

**Interfaces:**
- Consumes: `_dispatch_expert_misses(...)`, `_iter_expert_loads(...)` (Task 4); `_attach_cached_expert` (Task 2).
- Produces: no new names.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_accumulates_disk_telemetry(tmp_path, num_experts, top_k):
    """Module-list disk stall lands in _expert_disk_s instead of hiding in compute."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    assert not hasattr(vaayu_ref, "_expert_disk_s") or vaayu_ref._expert_disk_s == 0.0

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    assert hasattr(vaayu_ref, "_expert_disk_s"), "_expert_disk_s never set"
    assert vaayu_ref._expert_disk_s > 0.0, "cold misses recorded zero disk stall"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_gpu_residency_stays_bounded(tmp_path, num_experts, top_k):
    """Async disk loads must not become async GPU injection: at most one uncached
    expert holds real weights at a time."""
    import vaayu.moe_patch as mp

    model, vaayu_ref, vram = _patch_modulelist_with_vram(
        tmp_path, num_experts, top_k, vram_budget=0)   # 0 budget -> nothing cached
    moe = model.layers[0].block_sparse_moe

    peak = 0
    real_inject = mp._inject_expert_weights

    def _peek_inject(*a, **kw):
        nonlocal peak
        out = real_inject(*a, **kw)
        live = sum(1 for e in moe.experts if not e.w1.weight.is_meta)
        peak = max(peak, live)
        return out

    with mock_patch.object(mp, "_inject_expert_weights", _peek_inject):
        moe.forward(torch.randn(1, 1, 8))

    assert peak == 1, f"{peak} experts held real weights at once; bound is 1"
```

Note `vram_budget=0`: `ExpertVRAMCache.put` refuses any entry larger than the whole budget, so nothing is cached and every expert takes the inject path — which is what makes the residency bound observable.

- [ ] **Step 2: Run to verify it fails**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "disk_telemetry or gpu_residency" -v
```

Expected: `test_modulelist_accumulates_disk_telemetry` FAILs on `assert hasattr(vaayu_ref, "_expert_disk_s")` — the module-list path never sets it. `test_modulelist_gpu_residency_stays_bounded` should already PASS (Task 3 preserved the bound); it is a guard against this task breaking it.

- [ ] **Step 3: Switch the loop to dispatched loads**

In `_make_modulelist_sparse_forward`, the RAM-cache-hit / disk-read section inside the loop is replaced by a pre-pass. Insert before the `for eidx in unique_experts:` loop, after the `_vram` gating block from Task 3:

```python
        # Experts already VRAM-resident need no RAM lookup and no disk read at all.
        _vram_resident = (
            {e for e in unique_experts if _vram.has(layer_idx, e)}
            if _vram is not None else set()
        )
        _to_load = [e for e in unique_experts if e not in _vram_resident]

        # Dispatch disk misses up front and consume hits first, so compute overlaps
        # the in-flight reads. These futures are CPU-side (RAM cache / slab arena,
        # bounded by the RAM expert budget) — GPU injection below stays strictly
        # one expert at a time, so the residency bound is untouched.
        _expert_tensors, _miss_futures, _pinned_keys = _dispatch_expert_misses(
            _to_load,
            layer_idx=layer_idx, layer_name=layer_name, shard_file=shard_file,
            offset_index=offset_index, expert_cache=expert_cache,
            vaayu_ref=vaayu_ref,
        )
```

Restructure the loop to run VRAM hits first, then loaded experts. Factor the per-expert compute into a local closure so both arms share it verbatim:

```python
        def _run_expert(eidx: int, expert_mod) -> None:
            b_idx, s_idx, k_idx = torch.where(_idx_cpu == eidx)
            if b_idx.numel() == 0:
                return
            b_gpu = b_idx.to(hidden_states.device)
            s_gpu = s_idx.to(hidden_states.device)
            k_gpu = k_idx.to(hidden_states.device)
            _x_in = hidden_states[b_gpu, s_gpu]
            expert_out = expert_mod(_x_in)                               # (M, H)
            _trace_expert(vaayu_ref, layer_idx, eidx, expert_mod, _x_in, expert_out)
            w = routing_weights[b_gpu, s_gpu, k_gpu].unsqueeze(-1)
            final_hidden.index_put_((b_gpu, s_gpu),
                                    (w * expert_out).to(final_hidden.dtype),
                                    accumulate=True)

        def _detach_expert(eidx: int, expert_mod) -> None:
            try:
                # Detach from the live module — ALWAYS, hit or miss. See the
                # extended rationale on the miss arm below.
                for submod in expert_mod.modules():
                    if isinstance(submod, ExLlamaV3Linear):
                        submod.bc = None
                        submod.weight_fallback = None
                expert_mod.to("meta")
            except Exception as e:
                _log.warning(f"Failed to unload expert {eidx} to meta: {e}")

        try:
            for eidx in _vram_resident:
                expert_mod = experts_list[eidx]
                entry = _vram.get(layer_idx, eidx)
                if entry is None or not _attach_cached_expert(expert_mod, entry):
                    _to_load.append(eidx)      # raced out of the cache — load it
                    continue
                try:
                    _run_expert(eidx, expert_mod)
                finally:
                    _detach_expert(eidx, expert_mod)

            for eidx, raw in _iter_expert_loads(_expert_tensors, _miss_futures, vaayu_ref):
                expert_mod = experts_list[eidx]
                entry = _inject_expert_weights(
                    expert_mod, raw, eidx, layer_name, device, dtype)
                if not entry:
                    continue
                if _vram is not None:
                    # Return value deliberately ignored: unlike the fused path there
                    # is no _owned_by_cache branch. _detach_expert must run whether
                    # or not the cache accepted the entry — on a hit it frees
                    # nothing (the entry holds the refs); without it an evicted
                    # entry would still be referenced by the live ModuleList, so
                    # eviction would reclaim no VRAM. It also keeps peak GPU
                    # residency at one uncached expert regardless of routing width.
                    _vram.put(layer_idx, eidx, entry)
                try:
                    _run_expert(eidx, expert_mod)
                finally:
                    _detach_expert(eidx, expert_mod)
        finally:
            # Release every key THIS call pinned, even if compute raised partway
            # through — a leaked pin permanently blocks that key's eviction.
            for _pk in _pinned_keys:
                expert_cache.unpin(*_pk)
```

Two details:
- Experts that raced out of the VRAM cache between `has()` and `get()` are appended to `_to_load` but their loads were never dispatched; they are simply skipped this token. That is correct-but-lossy for one token, and vanishingly rare (single-threaded decode). Leave it — do not add a re-dispatch path for it.
- `_dispatch_expert_misses` handles the RAM-cache lookup and the `expert_cache.put`, so the old inline `expert_cache.get` / `load_expert_sparse` / `expert_cache.put` block is deleted entirely, along with the now-unused `load_expert_sparse` import at the top of `_make_modulelist_sparse_forward` if nothing else uses it.

- [ ] **Step 4: Run the tests**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "modulelist" -v
pytest tests/ -q
```

Expected: all module-list tests PASS, including the Task 3 set (hit/miss counts, detach, byte accounting, gating, output equivalence) and the two new ones. Full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/moe_patch.py tests/test_moe_patch.py
git commit -m "perf(moe): async miss loading + disk telemetry for module-list

Dispatch this layer's disk misses up front and compute VRAM hits while they are
in flight, via the shared _dispatch_expert_misses/_iter_expert_loads helpers.
The futures are CPU-side (RAM cache / slab arena), so GPU injection stays one
expert at a time and the residency bound is unchanged — now covered by a test.

_expert_disk_s is accumulated here for the first time; the module-list path's
disk time was previously invisible, hidden inside the compute total.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Measure, decide on the deferred slab port, document

**Files:**
- Modify: `test_generate.py` (`SESSION_ID` only)
- Modify: `PERF-BRIEF-minimax-vram-cache.md` (results section)
- Create: none

**Interfaces:** none.

- [ ] **Step 1: Run the MiniMax measurement**

Set `MODEL_PATH` back to `/home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3`, `MAX_NEW_TOKENS = 256`, and `SESSION_ID = "minimax-256-vram-final"` (fresh — a reused id hits the prefix cache and skips prefill, invalidating TTFT).

```bash
source venv/bin/activate
python -u test_generate.py > run-minimax-after.log 2>&1
grep -E "vram=|expert_cache=|total=|compute=|expert_disk=|Decode throughput|Time to first token" run-minimax-after.log
```

Baseline for comparison: **0.62 tok/s**, TTFT 53.1s, `vram=0hit/0miss/0evict`, `compute=1.6s`, RSS flat at 18.50-18.51GB.

Success signals:
- `vram=` shows real hits (nonzero hit and evict counts)
- `compute=` drops materially from ~1.6s
- `expert_disk=` is now nonzero (it was a telemetry gap, not a real zero)
- decode ≥1.0 tok/s

- [ ] **Step 2: Sanity-check correctness, not just speed**

From the same log:
- Generated code is coherent Python, not garbage — a routing or attach bug shows up as incoherence long before it shows up as a crash.
- No `NaN` in the output or warnings in the log.
- RSS plateaus around 17.5-18.5GB. It was dead flat at 18.50-18.51GB in the baseline, so **any** RSS change is caused by this work — investigate before shipping.

- [ ] **Step 3: Confirm the result is not variance**

Run it a second time with another fresh `SESSION_ID` (`"minimax-256-vram-final-2"`). Run-to-run variance on this stack is real (TTFT 43.0-53.7 observed within a single session on varying code). A <10% delta from one sample is not a win.

- [ ] **Step 4: Decide on the deferred zero-clone slab port**

Design §8 deferred the fused path's slab-backed single-H2D upload behind this measurement, on the theory that MiniMax's 4.40GB working set fits the 9.7GB VRAM budget, so after warmup there are no misses and no H2D left to optimize.

Read `expert_disk=` and the profiling `h2d` figure from Step 1:
- **Both near zero after warmup** → the theory held. Do not build the slab port. Record that in the brief and stop.
- **Either still material** → the working set is not staying resident (check `vram=` evictions). Write a follow-up plan for the slab port, referencing `vaayu/moe_patch.py:1203-1226` as the implementation to mirror, and note the measured numbers that justify it.

Do not build it on speculation either way — the numbers decide.

- [ ] **Step 5: Record the results**

Append a `## 12. Results` section to `PERF-BRIEF-minimax-vram-cache.md` with the before/after table (tok/s, TTFT, `vram=`, `expert_cache=`, `compute=`, `expert_disk=`, RSS), both sample runs, and the Step 4 decision with its supporting numbers.

- [ ] **Step 6: Commit**

```bash
git add PERF-BRIEF-minimax-vram-cache.md test_generate.py
git commit -m "docs(moe): record module-list VRAM cache results

Before: 0.62 tok/s, TTFT 53.1s, vram=0hit/0miss/0evict, compute=1.6s.
After: <fill from the measurement>.

Includes the decision on the deferred zero-clone slab port, with the
expert_disk/h2d numbers that drove it.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Spec Coverage

| Spec section | Task |
|---|---|
| §3 cache entry representation | Task 1 (shape produced), Task 3 (stored) |
| §3.1 entry must carry `gpu_tensors` | Task 1 Step 3 |
| §4 always-detach lifetime contract | Task 3 Step 4, Task 5 Step 3 (`_detach_expert`), tested in Task 3 Step 2 and Task 5 Step 1 |
| §5 prefill + layer-range gating | Task 3 Step 4, tested Task 3 Step 2 |
| §6 helpers (`_inject_expert_weights`, `_attach_cached_expert`) | Tasks 1, 2 |
| §7 async miss loading + telemetry, shared extraction | Tasks 4, 5 |
| §8 deferred zero-clone slabs | Task 6 Step 4 (decision gate) |
| §9 testing (4 behaviors, both geometries, regression guards) | Task 3 Step 2 (behaviors 1-4), every task's full-suite step, Task 3 Step 6 (dense), Task 4 Step 7 (Qwen) |
| §10 verification | Task 3 Step 8 (interim), Task 6 |
| §11 constraints | Global Constraints |
