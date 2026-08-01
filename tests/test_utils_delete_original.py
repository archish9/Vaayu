"""`delete_original` on the ALREADY-SPLIT path.

The mid-split deletion (delete each source shard as the loop moves past it)
already existed, but it is unreachable once a model is split: the function
returns early at "saved layers already found" before `delete_original` is ever
read. That left every already-split model holding both copies forever — the
exact disk cost the flag exists to avoid.

Deletion here is gated on the split being VERIFIED COMPLETE (every expected
layer present), which is a stronger precondition than the mid-loop path has.
"""
import json
import os
from pathlib import Path

import pytest

from vaayu.utils import split_and_save_layers


def _fake_model_dir(tmp_path, n_layers=2, n_shards=2):
    """A checkpoint dir with an index, original shards, and nothing split yet."""
    weight_map = {}
    for i in range(n_layers):
        shard = f"model-0000{i % n_shards + 1}-of-0000{n_shards}.safetensors"
        weight_map[f"model.layers.{i}.w"] = shard
    weight_map["model.embed_tokens.weight"] = "model-00001-of-00002.safetensors"
    weight_map["model.norm.weight"] = "model-00002-of-00002.safetensors"
    weight_map["lm_head.weight"] = "model-00002-of-00002.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    for s in range(1, n_shards + 1):
        (tmp_path / f"model-0000{s}-of-0000{n_shards}.safetensors").write_bytes(b"x" * 16)
    return sorted(tmp_path.glob("model-0000*.safetensors"))


def _fake_split_output(tmp_path, n_layers=2):
    """Mark the split as already complete, the way a prior run would leave it."""
    out = tmp_path / "splitted_model"
    out.mkdir()
    for name in (["model.embed_tokens"]
                 + [f"model.layers.{i}" for i in range(n_layers)]
                 + ["model.norm", "lm_head"]):
        (out / f"{name}.safetensors").write_bytes(b"y" * 8)
        (out / f"{name}.safetensors.done").touch()      # the completeness marker
    return out


def test_already_split_with_delete_original_removes_the_originals(tmp_path):
    originals = _fake_model_dir(tmp_path)
    _fake_split_output(tmp_path)

    saved = split_and_save_layers(str(tmp_path), delete_original=True)

    assert Path(saved) == tmp_path / "splitted_model"
    for f in originals:
        assert not f.exists(), f"{f.name} should have been reclaimed"
    # the index and the split output must survive — the index is read on every
    # subsequent load, and the split IS the model now
    assert (tmp_path / "model.safetensors.index.json").exists()
    assert list((tmp_path / "splitted_model").glob("*.safetensors"))


def test_already_split_without_the_flag_keeps_everything(tmp_path):
    """Default is non-destructive. This is the regression guard that matters."""
    originals = _fake_model_dir(tmp_path)
    _fake_split_output(tmp_path)

    split_and_save_layers(str(tmp_path), delete_original=False)

    for f in originals:
        assert f.exists(), f"{f.name} must NOT be touched when the flag is off"


def test_incomplete_split_never_deletes_originals(tmp_path):
    """A partial split re-runs the whole split, and that re-run READS the
    originals. Deleting them here would destroy the only complete copy."""
    originals = _fake_model_dir(tmp_path)
    out = _fake_split_output(tmp_path)
    os.remove(out / "model.layers.1.safetensors.done")   # split is now incomplete

    with pytest.raises(Exception):
        # the re-split reads the fake shards, which hold no real tensors --
        # what matters is what is still on disk afterwards
        split_and_save_layers(str(tmp_path), delete_original=True)

    for f in originals:
        assert f.exists(), (
            f"{f.name} deleted while the split was incomplete — unrecoverable")


def test_remove_real_and_linked_file_handles_a_plain_path(tmp_path):
    """os.path.realpath() returns str, so `realpath(p) != p` is ALWAYS True for a
    Path argument — the helper then removed the same file twice and the second
    unlink raised ENOENT. Every caller passes a Path, so this fired every time:
    handled (and silently mis-accounted) in delete_original_shards, UNHANDLED in
    the mid-split loop, where it would abort a fresh conversion."""
    from vaayu.utils import remove_real_and_linked_file

    f = tmp_path / "shard.safetensors"
    f.write_bytes(b"z" * 8)
    remove_real_and_linked_file(f)          # must not raise
    assert not f.exists()


def test_remove_real_and_linked_file_follows_a_symlink(tmp_path):
    """The linked-target deletion is the helper's whole reason to exist: an HF
    cache blob reached through a symlink must go too, or nothing is reclaimed."""
    from vaayu.utils import remove_real_and_linked_file

    real = tmp_path / "blob"
    real.write_bytes(b"z" * 8)
    link = tmp_path / "shard.safetensors"
    link.symlink_to(real)

    remove_real_and_linked_file(link)
    assert not link.exists() and not real.exists()


def test_delete_original_shards_reports_what_it_freed(tmp_path, capsys):
    """The byte total is the user-visible confirmation; the double-unlink bug
    suppressed it while still deleting, which looks exactly like a no-op."""
    from vaayu.utils import delete_original_shards

    originals = _fake_model_dir(tmp_path)
    index = {"a": originals[0].name, "b": originals[1].name}
    freed = delete_original_shards(tmp_path, index, True)

    assert freed == sum(16 for _ in originals)
    out = capsys.readouterr().out
    assert "reclaimed" in out
    assert "could not delete" not in out
