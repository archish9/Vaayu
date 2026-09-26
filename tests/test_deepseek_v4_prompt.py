"""S5 -- DeepSeek-V4 renders prompts with the checkpoint's OWN encoder.

The unit tests build a fake `encoding/encoding_dsv4.py` in tmp_path -- no
checkpoint, no GPU. The golden tests read DeepSeek's own fixtures off the real
checkpoint and SKIP without it.
"""
import json
import sys

import pytest

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4 import prompt
from vaayu.vaayu_deepseek_v4 import VaayuDeepseekV4

FAKE_ENCODER = '''
def encode_messages(messages, thinking_mode, **kwargs):
    return thinking_mode + "|" + "|".join(m["content"] for m in messages)
'''

MESSAGES = [{"role": "user", "content": "hi"}]


def _checkpoint(tmp_path, source=FAKE_ENCODER):
    (tmp_path / "encoding").mkdir()
    (tmp_path / "encoding" / "encoding_dsv4.py").write_text(source)
    return tmp_path


@pytest.mark.parametrize("thinking, mode", [(True, "thinking"), (False, "chat")])
def test_enable_thinking_picks_deepseeks_mode(tmp_path, thinking, mode):
    renderer = prompt.load_prompt_renderer(_checkpoint(tmp_path))
    assert renderer.render(MESSAGES, thinking) == f"{mode}|hi"


def test_the_encoder_is_not_left_in_sys_modules(tmp_path):
    before = set(sys.modules)
    prompt.load_prompt_renderer(_checkpoint(tmp_path))
    leaked = {name for name in set(sys.modules) - before
              if "dsv4" in name or name.startswith("_vaayu_deepseek_v4")}
    assert not leaked, f"loader left {leaked} in sys.modules"


def test_a_download_without_the_encoder_warns_and_refuses_naming_the_file(tmp_path, capsys):
    renderer = prompt.load_prompt_renderer(tmp_path)

    assert "encoding_dsv4.py" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="encoding_dsv4.py"):
        renderer.render(MESSAGES, True)


def test_an_encoder_without_encode_messages_is_refused_not_guessed(tmp_path, capsys):
    # Review Focus 4: a future DeepSeek rename must not become a guessed prompt.
    renderer = prompt.load_prompt_renderer(_checkpoint(tmp_path, "def other(): pass\n"))

    assert "encode_messages" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="encode_messages"):
        renderer.render(MESSAGES, True)


def test_init_model_builds_the_renderer_from_the_checkpoint_root(monkeypatch, tmp_path):
    """Same real-directory + fake-model-class pattern as
    tests/test_deepseek_v4_checkpoint.py's init_model tests."""
    class Config:
        fp4_group_size = 32

    monkeypatch.setattr(modeling, "KERNEL_DIR", None)
    (tmp_path / "inference").mkdir()
    (tmp_path / "inference" / "kernel.py").write_text("# stub")
    _checkpoint(tmp_path)

    class FakeLoadedModel:
        pass

    FakeLoadedModel.__module__ = modeling.__name__
    monkeypatch.setattr("vaayu.vaayu_base.VaayuBaseModel.init_model",
                        lambda self: setattr(self, "model", FakeLoadedModel()))

    model = VaayuDeepseekV4.__new__(VaayuDeepseekV4)
    model.config = Config()
    model.checkpoint_path = "/some/split"
    model.model_local_path = tmp_path
    model.layer_names_dict = {"layer_prefix": "model.layers"}
    model.init_model()

    assert model.get_prompt_renderer().render(MESSAGES, False) == "chat|hi"


# ── the real checkpoint ─────────────────────────────────────────────────────

ENCODING = parity.CKPT / "encoding"
needs_encoder = pytest.mark.skipif(
    not (ENCODING / "encoding_dsv4.py").is_file(),
    reason=f"DeepSeek-V4's encoder not at {ENCODING}")


@needs_encoder
@pytest.mark.parametrize("case, thinking", [(1, True), (2, True), (3, True), (4, False)])
def test_deepseeks_own_goldens_reproduce_through_vaayus_renderer(case, thinking):
    """`encoding/test_encoding_dsv4.py`'s four cases, through `render()`.

    They pass ONLY `thinking_mode` ("thinking" for 1-3, "chat" for 4), so
    `enable_thinking` alone reproduces all four. Case 1 attaches its tools to
    the first message, exactly as DeepSeek's own test does.
    """
    loaded = json.loads((ENCODING / "tests" / f"test_input_{case}.json").read_text())
    if case == 1:
        messages = loaded["messages"]
        messages[0]["tools"] = loaded["tools"]
    else:
        messages = loaded
    gold = (ENCODING / "tests" / f"test_output_{case}.txt").read_text()

    assert prompt.load_prompt_renderer(parity.CKPT).render(messages, thinking) == gold


@needs_encoder
def test_the_servers_default_prompt_is_one_bos_and_ends_in_a_closed_think():
    """Review Focus 3 + 5: the server's default ENABLE_THINKING=False renders
    "chat" mode, whose prompt ends with the ONE `</think>` token; tokenized the
    way `render_prompt` does it, BOS appears exactly once. The tokenizer loads
    through the S5 registration, so this also holds on a plain download."""
    import vaayu.vaayu_deepseek_v4  # noqa: F401 -- registers deepseek_v4
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(parity.CKPT), trust_remote_code=True)
    text = prompt.load_prompt_renderer(parity.CKPT).render(
        [{"role": "system", "content": "You are helpful."},
         {"role": "user", "content": "Hi"}], False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    assert ids[0] == tokenizer.bos_token_id
    assert ids.count(tokenizer.bos_token_id) == 1
    assert tokenizer.convert_ids_to_tokens(ids[-1]) == "</think>"
