import sys
import types
import importlib.machinery

# transformers calls importlib.util.find_spec("mlx") which requires a proper
# ModuleSpec — MagicMock fails. Use types.ModuleType with a real spec instead.
def _fake_module(name: str, is_package: bool = False) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    if is_package:
        mod.__path__ = []
    return mod

for _name, _pkg in [
    ("mlx", True),
    ("mlx.core", False),
    ("mlx.nn", False),
    ("mlx.core.transforms", False),
    ("mlx.optimizers", False),
]:
    sys.modules.setdefault(_name, _fake_module(_name, _pkg))
