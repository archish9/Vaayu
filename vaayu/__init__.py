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
    from .vaayu_qwen2 import VaayuQWen2
    from .vaayu_baichuan import VaayuBaichuan
    from .vaayu_internlm import VaayuInternLM
    from .vaayu_mistral import VaayuMistral
    from .vaayu_mixtral import VaayuMixtral
    from .vaayu_base import VaayuBaseModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException

