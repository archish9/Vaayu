"""S51: gpt-oss (MXFP4 whole-layer forward) is the only on-device model that
needs the `triton_kernels` MXFP4 kernels, and transformers fetches them from the
HF Hub on first load. `vaayu/offline_kernels.py` points the `kernels` lib at a
repo-vendored copy via its LOCAL_KERNELS override, so the load needs no network.
These tests guard the vendored copy and the override helper; the end-to-end
offline load is the on-device gate (gpt-oss server 9/9 with HF_HUB_OFFLINE=1)."""

import os
from pathlib import Path

import pytest

from vaayu.offline_kernels import (
    GPT_OSS_KERNEL_REPO_ID,
    VENDORED_KERNEL_DIR,
    ensure_gpt_oss_kernel_offline,
)


@pytest.fixture(autouse=True)
def _restore_local_kernels():
    before = os.environ.get("LOCAL_KERNELS")
    yield
    if before is None:
        os.environ.pop("LOCAL_KERNELS", None)
    else:
        os.environ["LOCAL_KERNELS"] = before


def test_vendored_kernel_imports_via_get_local_kernel():
    """The committed vendored kernel must load without the network -- this is
    the file that makes the whole feature work; if it rots, this fails."""
    from kernels.utils import get_local_kernel, package_name_from_repo_id

    pkg = package_name_from_repo_id(GPT_OSS_KERNEL_REPO_ID)
    module = get_local_kernel(VENDORED_KERNEL_DIR, pkg)
    assert module is not None


def test_ensure_sets_local_kernels_to_vendored_absolute_path():
    os.environ.pop("LOCAL_KERNELS", None)
    assert ensure_gpt_oss_kernel_offline() is True
    entry = f"{GPT_OSS_KERNEL_REPO_ID}={VENDORED_KERNEL_DIR.resolve()}"
    assert os.environ["LOCAL_KERNELS"] == entry


def test_ensure_is_idempotent():
    os.environ.pop("LOCAL_KERNELS", None)
    ensure_gpt_oss_kernel_offline()
    first = os.environ["LOCAL_KERNELS"]
    ensure_gpt_oss_kernel_offline()
    assert os.environ["LOCAL_KERNELS"] == first


def test_ensure_preserves_other_repo_overrides():
    os.environ["LOCAL_KERNELS"] = "some-org/other-kernel=/opt/other"
    ensure_gpt_oss_kernel_offline()
    entries = os.environ["LOCAL_KERNELS"].split(":")
    assert "some-org/other-kernel=/opt/other" in entries
    assert f"{GPT_OSS_KERNEL_REPO_ID}={VENDORED_KERNEL_DIR.resolve()}" in entries


def test_ensure_replaces_a_stale_entry_for_the_same_repo():
    os.environ["LOCAL_KERNELS"] = f"{GPT_OSS_KERNEL_REPO_ID}=/stale/path"
    ensure_gpt_oss_kernel_offline()
    entries = os.environ["LOCAL_KERNELS"].split(":")
    assert f"{GPT_OSS_KERNEL_REPO_ID}=/stale/path" not in entries
    assert f"{GPT_OSS_KERNEL_REPO_ID}={VENDORED_KERNEL_DIR.resolve()}" in entries


def test_ensure_warns_and_noops_when_vendored_dir_missing(tmp_path):
    os.environ.pop("LOCAL_KERNELS", None)
    missing = tmp_path / "not_here"
    with pytest.warns(UserWarning):
        assert ensure_gpt_oss_kernel_offline(vendor_dir=missing) is False
    assert "LOCAL_KERNELS" not in os.environ
