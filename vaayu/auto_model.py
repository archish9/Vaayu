import importlib
import pathlib
import subprocess
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    try:
        from .vaayu_llama_mlx import VaayuLlamaMlx
    except Exception:
        VaayuLlamaMlx = None

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
            pathlib.Path("/sys/module/nvme_core/parameters/default_ps_max_latency_us"),  # kernel 6.x
            pathlib.Path(f"/sys/class/nvme/{nvme_char}/power/ps_max_latency_us"),        # kernel 4.x/5.x
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
        if 'hf_token' in kwargs:
            print(f"using hf_token")
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token'])
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        if "Qwen2ForCausalLM" in config.architectures[0]:
            return "vaayu", "VaayuQWen2"
        elif "Qwen2_5ForCausalLM" in config.architectures[0]:
            return "vaayu", "VaayuQWen2"
        elif "Qwen3ForCausalLM" in config.architectures[0]:
            return "vaayu", "VaayuQWen2"
        elif "QWen" in config.architectures[0]:
            return "vaayu", "VaayuQWen"
        elif "Baichuan" in config.architectures[0]:
            return "vaayu", "VaayuBaichuan"
        elif "ChatGLM" in config.architectures[0]:
            return "vaayu", "VaayuChatGLM"
        elif "InternLM" in config.architectures[0]:
            return "vaayu", "VaayuInternLM"
        elif "Mistral" in config.architectures[0]:
            return "vaayu", "VaayuMistral"
        elif "Mixtral" in config.architectures[0]:
            return "vaayu", "VaayuMixtral"
        elif "Gemma" in config.architectures[0]:
            return "vaayu", "VaayuLlama2"
        elif "Llama" in config.architectures[0]:
            return "vaayu", "VaayuLlama2"
        else:
            print(f"unknown artichitecture: {config.architectures[0]}, try to use Llama2...")
            return "vaayu", "VaayuLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if not is_on_mac_os:
            _check_nvme_scheduler(pretrained_model_name_or_path)

        if is_on_mac_os:
            return VaayuLlamaMlx(pretrained_model_name_or_path, *inputs, ** kwargs)

        module, cls = AutoModel.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, cls)
        return class_(pretrained_model_name_or_path, *inputs, ** kwargs)