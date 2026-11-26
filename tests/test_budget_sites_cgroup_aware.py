"""Budget sites in vaayu_base must size from the cgroup-aware basis."""
import ast
from pathlib import Path

import pytest

import vaayu.vaayu_base as vb

SRC = Path(vb.__file__)
# helper -> (keyword name, positional index) of its available-bytes argument
HELPERS = {
    "derive_expert_ceiling": ("available_bytes", 0),
    "compute_expert_budget": ("available_bytes", 0),
    "compute_prefix_cache_budget": ("free_ram_bytes", 0),
    "spec_preload_refusal": ("available_bytes", 2),
    "auto_detect_cache_budget": ("available_bytes", 1),  # dense RAM layer cache
    "describe_cache_budget": ("available_bytes", 1),
}


def _calls_eff(node):
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None))
               == "effective_available_bytes" for n in ast.walk(node))


def _parents(tree):
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            c._parent = p


def test_vaayu_base_budget_reads_go_through_effective_available():
    tree = ast.parse(SRC.read_text())
    _parents(tree)
    checked = {k: 0 for k in HELPERS}
    problems = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        name = getattr(call.func, "id", getattr(call.func, "attr", None))
        if name not in HELPERS:
            continue
        kw, pos = HELPERS[name]
        arg = next((k.value for k in call.keywords if k.arg == kw), None)
        if arg is None and len(call.args) > pos:
            arg = call.args[pos]
        if arg is None:
            continue
        fn = call
        while fn is not None and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = getattr(fn, "_parent", None)
        checked[name] += 1
        where = f"{name} @ line {call.lineno}"
        if any(isinstance(n, ast.Attribute) and n.attr == "available" for n in ast.walk(arg)):
            problems.append(f"{where}: reads .available directly")
            continue
        if _calls_eff(arg):
            continue
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} if fn else set()
        callees = {id(n.func) for n in ast.walk(arg) if isinstance(n, ast.Call)}
        for nm in {n.id for n in ast.walk(arg)
                   if isinstance(n, ast.Name) and id(n) not in callees}:
            if nm in params:
                continue  # a helper forwarding its own parameter
            assigns = [a for a in ast.walk(fn or tree)
                       if isinstance(a, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == nm for t in a.targets)]
            if not assigns or not all(
                    _calls_eff(a.value)
                    or (isinstance(a.value, ast.Constant) and a.value.value is None)  # except-path sentinel
                    for a in assigns):
                problems.append(f"{where}: '{nm}' does not originate from effective_available_bytes")
    assert not problems, problems
    assert all(v >= 1 for v in checked.values()), checked


GEOMS = {
    "qwen3next_like": dict(layers=48, experts=512, expert_bytes=int(1.69 * 1024 ** 2), top_k=10),
    "minimax_like": dict(layers=62, experts=256, expert_bytes=8 * 1024 ** 2, top_k=8),
}


@pytest.mark.parametrize("geo", list(GEOMS))
def test_expert_helpers_safe_over_two_geometries_when_basis_shrinks(geo):
    g = GEOMS[geo]
    ws = g["top_k"] * g["layers"] * g["expert_bytes"]
    host, shrunk = 28 * 1024 ** 3, 12 * 1024 ** 3

    def run(avail):
        ceil = vb.derive_expert_ceiling(
            available_bytes=avail, prefix_sessions=2,
            kv_bytes_per_token=100_000, max_seq_len=8192, prefix_enabled=True)
        budget, ne, _per_layer = vb.compute_expert_budget(
            available_bytes=avail, working_set_bytes=ws,
            num_moe_layers=g["layers"], expert_bytes=g["expert_bytes"],
            moe_top_k=g["top_k"], moe_num_experts=g["experts"], ceiling_bytes=ceil)
        return ceil, budget, ne

    h_ceil, h_budget, _h_ne = run(host)
    s_ceil, s_budget, s_ne = run(shrunk)
    assert s_ceil >= vb._EXPERT_CACHE_OOM_CEILING and h_ceil >= vb._EXPERT_CACHE_OOM_CEILING
    assert s_budget >= 0 and h_budget >= 0 and s_ne >= 0
    assert s_ceil <= h_ceil
    assert s_budget <= h_budget


def test_projection_and_sizing_use_the_same_margin():
    """The plan line's ceiling and the tier actually created must not disagree:
    both `expert_vram_budget_for` call sites in the loader take the same
    `margin_bytes`, so a reader of the log sees the tier that was built."""
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(vb.VaayuBaseModel.__init__)))
    margins = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        name = getattr(call.func, "id", getattr(call.func, "attr", None))
        if name in ("expert_vram_budget_for", "_expert_budget_for"):
            kw = next((k.value for k in call.keywords if k.arg == "margin_bytes"), None)
            assert kw is not None, f"line {call.lineno}: no margin_bytes keyword"
            margins.append(ast.dump(kw))
    assert len(margins) == 2, f"expected two budget call sites, found {len(margins)}"
    assert margins[0] == margins[1]
