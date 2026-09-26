"""DeepSeek-V4's prompt format: the checkpoint's OWN encoder, read in place.

The checkpoint ships no Jinja `chat_template`, on purpose -- DeepSeek's README
points at `encoding/encoding_dsv4.py` instead, which is part of every full
download. Vaayu reads it from there and never copies it: each download brings
the encoder that matches its own weights, and nothing is written into the model
directory (S5).
"""
import types
from pathlib import Path

ENCODER = Path("encoding") / "encoding_dsv4.py"


class DeepseekV4PromptRenderer:
    """`render(messages, enable_thinking) -> str`, the contract
    `VaayuBaseModel.get_prompt_renderer` documents."""

    def __init__(self, encode_messages):
        self._encode = encode_messages

    def render(self, messages, enable_thinking):
        # DeepSeek's two modes: "thinking" ends the prompt with `<think>`;
        # "chat" ends it with `</think>`, so the model answers directly.
        return self._encode(messages,
                            thinking_mode="thinking" if enable_thinking else "chat")


class MissingEncoder:
    """Stands in when the download's encoder is absent or unusable, so every
    request is refused WITH the reason -- never prompted in a guessed format."""

    def __init__(self, reason):
        self.reason = reason

    def render(self, messages, enable_thinking):
        raise FileNotFoundError(self.reason)


def load_prompt_renderer(checkpoint_root):
    """The renderer for the checkpoint at `checkpoint_root` -- the ROOT, beside
    `inference/`, not Vaayu's `splitted_model/`.

    Loaded under a private name and never registered in `sys.modules`: a module
    called `encoding_dsv4` left lying around is a collision waiting to happen
    (the rule `tests/deepseek/parity.py` applies to the checkpoint's `kernel`).

    Compiled and executed directly rather than imported: an import caches
    bytecode as `encoding/__pycache__/*.pyc` BESIDE the source, which is a
    write into the user's download -- the one thing S5 promises never happens.
    """
    path = Path(checkpoint_root) / ENCODER
    if not path.is_file():
        return _refuse(f"{path} is missing. DeepSeek-V4 ships no chat_template; its "
                       f"prompt format lives in that file, part of the full "
                       f"checkpoint download.")
    module = types.ModuleType("_vaayu_deepseek_v4_encoding")
    module.__file__ = str(path)
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    except Exception as failure:
        # A revision with a SyntaxError, or an import this venv lacks, is the
        # same situation as a missing file -- not a reason to stop the load.
        return _refuse(f"{path} could not be loaded ({type(failure).__name__}: "
                       f"{failure}). Refusing rather than guessing the prompt "
                       f"format.")
    encode = getattr(module, "encode_messages", None)
    if encode is None:
        return _refuse(f"{path} defines no encode_messages, the entry point "
                       f"DeepSeek's encoding/README.md documents. Refusing rather "
                       f"than guessing the prompt format.")
    return DeepseekV4PromptRenderer(encode)


def _refuse(reason):
    print(f"[Vaayu Warning] {reason} /v1/chat/completions will refuse every "
          f"request with this reason.")
    return MissingEncoder(reason)
