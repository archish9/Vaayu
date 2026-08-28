"""`DELETE_ORIGINAL` reclaims the source shards a completed split replaced.

Irreversible by nature -- recovery is a re-download of 39-66 GB -- so two things
are gates, not niceties: the flag must actually reach the loader (a flag that
silently does nothing is how disk quietly fills), and the server must SAY it is
on before the load that acts on it.
"""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from vaayu.server.app import ServerConfig, create_app


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen.update(kwargs)
            return type("M", (), {
                "max_seq_len": 8192, "tokenizer": type("T", (), {"chat_template": None})(),
                "speculator": None, "_vram_plan": object(), "running_device": "cpu",
            })()

    import vaayu.auto_model as am
    import vaayu.gpu_audit as ga
    monkeypatch.setattr(am, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(ga, "audit", lambda: [])
    return seen


def test_delete_original_is_on_by_default():
    assert ServerConfig(model_path="/x").delete_original is True


def test_it_reaches_the_loader(captured):
    with TestClient(create_app(ServerConfig(model_path="/ckpt"))):
        pass
    assert captured["delete_original"] is True


def test_it_can_be_turned_off(captured):
    cfg = ServerConfig(model_path="/ckpt", delete_original=False)
    with TestClient(create_app(cfg)):
        pass
    assert captured["delete_original"] is False


def test_the_server_announces_an_irreversible_default_before_it_loads(captured, capsys):
    with TestClient(create_app(ServerConfig(model_path="/ckpt"))):
        pass
    out = capsys.readouterr().out
    assert "DELETE_ORIGINAL" in out
    assert "irreversible" in out.lower(), "name what cannot be undone"


def test_no_announcement_when_it_is_off(captured, capsys):
    with TestClient(create_app(ServerConfig(model_path="/ckpt", delete_original=False))):
        pass
    assert "DELETE_ORIGINAL" not in capsys.readouterr().out


def test_the_config_module_exposes_it():
    from vaayu.server import config
    assert hasattr(config, "DELETE_ORIGINAL")
    names = {f.name for f in dataclasses.fields(ServerConfig)}
    assert "delete_original" in names


# ---------------------------------------------------------------------------
# The splitter half: WHEN the originals are allowed to go.
#
# Two paths reach a deletion, and only one of them used to be verified. An
# already-split checkpoint is gated on `all(found_layers.values())` -- every
# layer has its .safetensors AND its .done marker. A FRESH split deleted source
# shard N-1 the moment the loop advanced to shard N, with four more shards still
# unwritten, and skipped `check_space` entirely because the flag was on. A crash
# at shard 5 of 9 left the originals for 1-4 gone and the split unusable, and
# `remove_real_and_linked_file` follows symlinks, so an HF-cache blob went with
# them. Recovery is a 39-66 GB re-download.
#
# These pin the contract: deletion happens ONCE, AFTER the split verifies.
# ---------------------------------------------------------------------------
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from vaayu import utils
from vaayu.persist.model_persister import ModelPersister

SHARD_1 = "model-00001-of-00002.safetensors"
SHARD_2 = "model-00002-of-00002.safetensors"


def _sharded_checkpoint(root: Path) -> Path:
    """A two-shard checkpoint, small enough to split in milliseconds.

    Layer 0 lives in shard 1 and layer 1 in shard 2, so the write loop must
    cross a shard boundary -- that crossing is where the old code deleted.
    """
    root.mkdir(parents=True, exist_ok=True)
    first = {"model.embed_tokens.weight": torch.zeros(2, 2),
             "model.layers.0.w": torch.zeros(2, 2)}
    second = {"model.layers.1.w": torch.zeros(2, 2),
              "model.norm.weight": torch.zeros(2),
              "lm_head.weight": torch.zeros(2, 2)}
    save_file(first, str(root / SHARD_1))
    save_file(second, str(root / SHARD_2))
    index = {k: SHARD_1 for k in first} | {k: SHARD_2 for k in second}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return root


def test_a_crash_mid_split_leaves_every_original_shard(tmp_path, monkeypatch):
    """The failure that costs a re-download: die after shard 1, before the end."""
    ckpt = _sharded_checkpoint(tmp_path / "ckpt")
    persister = ModelPersister.get_model_persister()
    real_persist = persister.persist_model

    def die_on_the_second_layer(state_dict, layer_name, saving_path):
        if layer_name == "model.layers.1":
            raise RuntimeError("disk full")
        return real_persist(state_dict, layer_name, saving_path)

    monkeypatch.setattr(persister, "persist_model", die_on_the_second_layer)

    with pytest.raises(RuntimeError, match="disk full"):
        utils.split_and_save_layers(ckpt, delete_original=True)

    assert (ckpt / SHARD_1).exists(), "shard 1 was deleted before the split finished"
    assert (ckpt / SHARD_2).exists()


def test_an_incomplete_split_keeps_the_originals(tmp_path, monkeypatch):
    """No exception, but a layer never landed. Silence is not verification."""
    ckpt = _sharded_checkpoint(tmp_path / "ckpt")
    persister = ModelPersister.get_model_persister()
    real_persist = persister.persist_model

    def skip_the_norm(state_dict, layer_name, saving_path):
        if layer_name == "model.norm":
            return None
        return real_persist(state_dict, layer_name, saving_path)

    monkeypatch.setattr(persister, "persist_model", skip_the_norm)
    utils.split_and_save_layers(ckpt, delete_original=True)

    assert (ckpt / SHARD_1).exists()
    assert (ckpt / SHARD_2).exists()


def test_a_verified_split_reclaims_every_shard_including_the_last(tmp_path):
    """The point of the flag. The LAST shard counts -- nothing advances past it."""
    ckpt = _sharded_checkpoint(tmp_path / "ckpt")
    utils.split_and_save_layers(ckpt, delete_original=True)

    assert not (ckpt / SHARD_1).exists()
    assert not (ckpt / SHARD_2).exists()
    assert (ckpt / "model.safetensors.index.json").exists(), "the index still gets read"
    assert (ckpt / "splitted_model" / "model.layers.1.safetensors").exists()


def test_the_originals_survive_when_the_flag_is_off(tmp_path):
    ckpt = _sharded_checkpoint(tmp_path / "ckpt")
    utils.split_and_save_layers(ckpt, delete_original=False)

    assert (ckpt / SHARD_1).exists()
    assert (ckpt / SHARD_2).exists()


def test_free_space_is_checked_even_when_the_flag_is_on(tmp_path, monkeypatch):
    """Deleting as you go was the reason to skip the pre-flight. It no longer does."""
    ckpt = _sharded_checkpoint(tmp_path / "ckpt")
    calls = []
    monkeypatch.setattr(utils, "check_space", lambda *a, **k: calls.append(a))

    utils.split_and_save_layers(ckpt, delete_original=True)
    assert calls, "check_space was skipped in exactly the mode with the least margin"


# ---------------------------------------------------------------------------
# B14: a SINGLE-FILE checkpoint must survive its own deletion.
#
# A multi-shard checkpoint stays loadable after the delete because
# `model.safetensors.index.json` is not a weight file and is left behind --
# discovery finds it, `split_and_save_layers` reads its weight_map, sees a
# complete split and returns it. A single-file checkpoint has no index: the
# deleted `model.safetensors` was the only one of the four names
# `find_or_create_local_splitted_path` recognises, so the NEXT load printed
# "didn't find downloaded model", treated the absolute path as a HF repo id and
# died in validate_repo_id -- with a complete splitted_model/ sitting right
# there. The damage is invisible in the run that causes it.
#
# The fix gives the single-file case the same survivor: an index synthesised
# from the map the splitter already built. Nothing downstream is new.
# ---------------------------------------------------------------------------

def _single_file_checkpoint(root: Path, name: str = "model.safetensors") -> Path:
    """One weight file, no index -- the shape that had no survivor."""
    root.mkdir(parents=True, exist_ok=True)
    weights = {"model.embed_tokens.weight": torch.zeros(2, 2),
               "model.layers.0.w": torch.zeros(2, 2),
               "model.layers.1.w": torch.zeros(2, 2),
               "model.norm.weight": torch.zeros(2),
               "lm_head.weight": torch.zeros(2, 2)}
    if name.endswith(".safetensors"):
        save_file(weights, str(root / name))
    else:
        torch.save(weights, str(root / name))
    return root


def test_a_single_file_checkpoint_is_still_discoverable_after_the_delete(tmp_path):
    """THE regression: load, then load again. The second one used to be fatal."""
    ckpt = _single_file_checkpoint(tmp_path / "ckpt")
    saved = utils.split_and_save_layers(ckpt, delete_original=True)
    assert not (ckpt / "model.safetensors").exists(), "the flag reclaimed nothing"

    # Same call the next process makes. Before the fix this raised
    # FileNotFoundError("No model files found") from split_and_save_layers,
    # and find_or_create_local_splitted_path never even got here.
    again = utils.split_and_save_layers(ckpt, delete_original=True)
    assert Path(again) == Path(saved)
    assert (ckpt / "splitted_model" / "model.layers.1.safetensors").exists()


def test_the_delete_leaves_an_index_behind(tmp_path):
    ckpt = _single_file_checkpoint(tmp_path / "ckpt")
    utils.split_and_save_layers(ckpt, delete_original=True)

    left = ckpt / "model.safetensors.index.json"
    assert left.exists(), "nothing survived to make the directory discoverable"
    weight_map = json.loads(left.read_text())["weight_map"]
    assert weight_map["model.layers.1.w"] == "model.safetensors"


def test_a_single_file_torch_checkpoint_gets_the_torch_index(tmp_path):
    """The other single-file shape. Its index has the other name."""
    ckpt = _single_file_checkpoint(tmp_path / "ckpt", name="pytorch_model.bin")
    utils.split_and_save_layers(ckpt, delete_original=True)

    assert not (ckpt / "pytorch_model.bin").exists()
    assert (ckpt / "pytorch_model.bin.index.json").exists()
    assert not (ckpt / "model.safetensors.index.json").exists()
    assert Path(utils.split_and_save_layers(ckpt, delete_original=True)).name == "splitted_model"


def test_an_existing_index_is_never_overwritten(tmp_path):
    """The sharded case already had a survivor. Do not touch the real one."""
    ckpt = _sharded_checkpoint(tmp_path / "ckpt")
    before = (ckpt / "model.safetensors.index.json").read_bytes()

    utils.split_and_save_layers(ckpt, delete_original=True)

    assert (ckpt / "model.safetensors.index.json").read_bytes() == before


def test_the_reclaim_is_given_up_rather_than_stranding_the_checkpoint(tmp_path, monkeypatch):
    """If the index cannot be written, the weight file must NOT go."""
    ckpt = _single_file_checkpoint(tmp_path / "ckpt")
    monkeypatch.setattr(Path, "write_text",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("read-only")))

    utils.split_and_save_layers(ckpt, delete_original=True)

    assert (ckpt / "model.safetensors").exists(), "deleted the only discoverable file anyway"


def test_discovery_itself_still_finds_the_directory_after_the_delete(tmp_path):
    """The exact call the next process makes -- and the one that used to fall
    through to `validate_repo_id` on an absolute path."""
    ckpt = _single_file_checkpoint(tmp_path / "ckpt")
    utils.find_or_create_local_splitted_path(str(ckpt), delete_original=True)

    local, split = utils.find_or_create_local_splitted_path(str(ckpt), delete_original=True)
    assert Path(local) == ckpt
    assert Path(split) == ckpt / "splitted_model"
