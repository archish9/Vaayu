"""Coverage for the local-checkpoint detection gate in
`find_or_create_local_splitted_path` (vaayu/utils.py).

Task 9 follow-up: closes a review gap flagged for the fix that added
`model.safetensors` / `pytorch_model.bin` (no index) as additional
acceptance paths in the "is this already a valid local checkpoint" gate.
These tests assert the fix is additive-only:
  - single-file local checkpoints now take the local branch
    (huggingface_hub.snapshot_download is NOT called)
  - the pre-existing index.json branch still works unchanged
  - a directory that is neither still falls through to the HF-repo-id /
    download path (snapshot_download IS called)
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from vaayu.utils import find_or_create_local_splitted_path


@pytest.fixture
def fake_split(monkeypatch):
    """Stub out split_and_save_layers so tests don't touch real tensors."""
    mock = MagicMock(return_value="fake_splitted_path")
    monkeypatch.setattr("vaayu.utils.split_and_save_layers", mock)
    return mock


def test_local_single_file_safetensors_takes_local_branch(tmp_path, fake_split):
    (tmp_path / "model.safetensors").write_bytes(b"")

    with patch("huggingface_hub.snapshot_download") as mock_download:
        model_path, splitted_path = find_or_create_local_splitted_path(str(tmp_path))

    mock_download.assert_not_called()
    fake_split.assert_called_once()
    assert model_path == Path(str(tmp_path))
    assert splitted_path == "fake_splitted_path"


def test_local_single_file_pytorch_bin_takes_local_branch(tmp_path, fake_split):
    (tmp_path / "pytorch_model.bin").write_bytes(b"")

    with patch("huggingface_hub.snapshot_download") as mock_download:
        model_path, splitted_path = find_or_create_local_splitted_path(str(tmp_path))

    mock_download.assert_not_called()
    fake_split.assert_called_once()
    assert model_path == Path(str(tmp_path))
    assert splitted_path == "fake_splitted_path"


def test_local_dir_with_index_json_still_takes_index_branch(tmp_path, fake_split):
    (tmp_path / "model.safetensors.index.json").write_text("{}")

    with patch("huggingface_hub.snapshot_download") as mock_download:
        model_path, splitted_path = find_or_create_local_splitted_path(str(tmp_path))

    mock_download.assert_not_called()
    fake_split.assert_called_once()
    assert model_path == Path(str(tmp_path))
    assert splitted_path == "fake_splitted_path"


def test_local_dir_with_pytorch_index_json_still_takes_index_branch(tmp_path, fake_split):
    (tmp_path / "pytorch_model.bin.index.json").write_text("{}")

    with patch("huggingface_hub.snapshot_download") as mock_download:
        model_path, splitted_path = find_or_create_local_splitted_path(str(tmp_path))

    mock_download.assert_not_called()
    fake_split.assert_called_once()
    assert model_path == Path(str(tmp_path))
    assert splitted_path == "fake_splitted_path"


def test_empty_local_dir_falls_through_to_hf_download(tmp_path, fake_split):
    # tmp_path exists locally but has none of the recognized checkpoint files,
    # so it must be treated as a (locally-shadowed) HF repo id and downloaded.
    with patch("huggingface_hub.snapshot_download") as mock_download:
        mock_download.return_value = str(tmp_path)
        find_or_create_local_splitted_path(str(tmp_path))

    mock_download.assert_called()


def test_nonexistent_path_falls_through_to_hf_download(tmp_path, fake_split):
    nonexistent = str(tmp_path / "does-not-exist-locally/some/repo-id")

    with patch("huggingface_hub.snapshot_download") as mock_download:
        mock_download.return_value = str(tmp_path)
        find_or_create_local_splitted_path(nonexistent)

    mock_download.assert_called()

