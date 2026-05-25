from sys import platform

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    try:
        from .vaayu_llama_mlx import VaayuLlamaMlx
    except Exception:
        pass

from .auto_model import AutoModel
from .vaayu_base import VaayuBaseModel
from .vaayu_chatglm import VaayuChatGLM
from .vaayu_qwen import VaayuQWen

if not is_on_mac_os:
    from .vaayu import VaayuLlama2
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException
