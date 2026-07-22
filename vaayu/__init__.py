import os as _os
# Expandable CUDA segments before the allocator's first allocation. On a tight
# card, a second large-context prefill against a warm VRAM expert cache churns
# per-expert GPU allocations and fragments freed space (a 14MB alloc failing with
# 631MB reserved-but-unallocated). Expandable segments coalesce instead of
# fragmenting. Set here (import time, before any model allocation) rather than via
# a launch flag so `import vaayu` just works; the runtime allocator setter proved
# unreliable post-context-init. torch.cuda.is_available() does not allocate, so
# this lands before the caching allocator reads the setting. setdefault: an
# explicit launch env var still wins.
_alloc_conf = _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if "expandable_segments:true" not in _alloc_conf.lower():
    # A caller-set PYTORCH_CUDA_ALLOC_CONF wins (setdefault above) — correct, we
    # never override explicit intent — but a config without expandable_segments
    # silently drops the fragmentation fix and the multi-session prefill OOM
    # returns. Name the condition so it is diagnosable, not a silent regression.
    print(f"[Vaayu] WARNING: PYTORCH_CUDA_ALLOC_CONF={_alloc_conf!r} does not enable "
          f"expandable_segments — a large-context prefill against a warm VRAM expert "
          f"cache can OOM from allocator fragmentation. Add 'expandable_segments:True'.")

from .auto_model import AutoModel
from .vaayu_base import VaayuBaseModel
from .vaayu_chatglm import VaayuChatGLM
from .vaayu_qwen import VaayuQWen
from .vaayu import VaayuLlama2
from .utils import split_and_save_layers
from .utils import NotEnoughSpaceException

__all__ = [
    "AutoModel",
    "VaayuBaseModel",
    "VaayuChatGLM",
    "VaayuQWen",
    "VaayuLlama2",
    "split_and_save_layers",
    "NotEnoughSpaceException",
]
