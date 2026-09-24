import pathlib
import subprocess
from transformers import AutoConfig


def nvme_apst_tolerance(block_dev: str, sys_root: str = "/sys"):
    """Live APST latency tolerance of the controller behind `block_dev` ("nvme0n1").

    This is the knob the kernel honours at runtime: the controller's PM-QoS
    latency tolerance (nvme_set_latency_tolerance -> nvme_configure_apst).
    The module parameter nvme_core.default_ps_max_latency_us is read only when
    a controller initialises, so after boot it says nothing about the drive --
    writing it changes nothing, and reading it can report "disabled" while the
    drive still has APST on. The file is exposed only for controllers that
    support APST.

    Returns the raw value ("0" = APST off; an integer, "auto" or "any" = APST
    allowed), or None when the controller exposes no APST control.
    """
    f = (pathlib.Path(sys_root) / "block" / block_dev / "device" / "power"
         / "pm_qos_latency_tolerance_us")
    return f.read_text().strip() if f.is_file() else None


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

        scheduler = pathlib.Path(f"/sys/block/{nvme}/queue/scheduler").read_text()
        fix_cmd = f"sudo bash scripts/vaayu_optimize_os.sh {model_path}  # one-time per machine"

        if "[none]" not in scheduler:
            print(f"[Vaayu Warning] {nvme} I/O scheduler is not optimized "
                  f"(current: {scheduler.strip()}).")
            print(f"  You are losing ~5% NVMe throughput.")
            print(f"  Fix (once): {fix_cmd}")
        else:
            print(f"[Vaayu] {nvme} I/O scheduler: optimized (none)")

        apst_val = nvme_apst_tolerance(nvme)
        if apst_val is not None:
            if apst_val != "0":
                # Info, not a warning: measured with no decode/TTFT cost -- the
                # drive's idle gaps during generation are shorter than APST's
                # sleep timeout (todo.md row 25, 2026-09-11).
                print(f"[Vaayu] {nvme} APST: active (ps_max_latency_us={apst_val}) — "
                      f"no measured decode/TTFT cost; optional to disable (until "
                      f"reboot): sudo bash scripts/vaayu_optimize_os.sh {model_path}")
            else:
                print(f"[Vaayu] {nvme} APST: optimized (disabled)")
    except Exception:
        pass


def warn_if_gpu_is_shared() -> None:
    """Say what another process is costing this load. Never refuses.

    The server refuses instead (ServerConfig.require_exclusive_gpu); a refusal
    here would break every existing script and notebook, and the library caller
    may have good reason to share.
    """
    from . import gpu_audit
    report = gpu_audit.compose_report(gpu_audit.audit())
    if report:
        print(f"[Vaayu Warning] {report}")


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
        from .vaayu_deepseek_v4 import VaayuDeepseekV4

        if 'hf_token' in kwargs:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True, token=kwargs['hf_token']
            )
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )

        architectures = getattr(config, "architectures", None) or []
        if not architectures:
            print(f"no architectures field in config for {pretrained_model_name_or_path}, using VaayuBaseModel with auto-detect...")
            return VaayuBaseModel, {}
        arch = architectures[0]

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
        elif "DeepseekV4" in arch:
            return VaayuDeepseekV4, {}
        else:
            print(f"unknown architecture: {arch}, using VaayuBaseModel with auto-detect...")
            return VaayuBaseModel, {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        _check_nvme_scheduler(pretrained_model_name_or_path)
        warn_if_gpu_is_shared()

        model_cls, extra_kwargs = AutoModel.get_module_class(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        # user-supplied kwargs win over architecture defaults
        return model_cls(pretrained_model_name_or_path, *inputs, **{**extra_kwargs, **kwargs})
