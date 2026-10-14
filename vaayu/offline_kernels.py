"""Make the gpt-oss (MXFP4) load work offline.

gpt-oss-20b runs the whole-layer MXFP4 forward, the only on-device model that
needs the `triton_kernels` MXFP4 kernels. transformers fetches them from the HF
Hub on first load (`transformers/integrations/mxfp4.py` ->
`get_kernel("kernels-community/gpt-oss-triton-kernels")`), so a first load -- or
any load after the HF cache is cleared -- needs the network and fails offline.

The `kernels` lib (>=0.13) checks its LOCAL_KERNELS override *before* any network
call, so pointing that at a repo-vendored copy (`vaayu/vendor/...`) makes the
load need no network and no HF cache. This is gpt-oss-only; no other model
requests this repo id, so the override is inert for them.
"""

import os
import warnings
from pathlib import Path

# The kernel's own repo id on the Hub (the model's string, not an arch literal).
GPT_OSS_KERNEL_REPO_ID = "kernels-community/gpt-oss-triton-kernels"

# The committed copy of the one `torch-cuda` build variant (pure-Python triton,
# torch-version-agnostic). `kernels.get_local_kernel` reads `<dir>/build/...`.
VENDORED_KERNEL_DIR = Path(__file__).parent / "vendor" / "gpt_oss_triton_kernels"


def ensure_gpt_oss_kernel_offline(vendor_dir: Path | None = None) -> bool:
    """Point the kernels lib at the vendored gpt-oss MXFP4 kernel via LOCAL_KERNELS.

    Returns True if the override was set. If the vendored copy is missing, warns
    and leaves LOCAL_KERNELS untouched so the stock network fetch still runs.
    """
    vendor_dir = Path(vendor_dir) if vendor_dir is not None else VENDORED_KERNEL_DIR
    if not vendor_dir.is_dir():
        warnings.warn(
            f"[Vaayu] vendored gpt-oss kernel not found at {vendor_dir}; "
            f"the MXFP4 load will fall back to fetching "
            f"{GPT_OSS_KERNEL_REPO_ID} from the HF Hub (needs the network).",
            UserWarning,
            stacklevel=2,
        )
        return False

    entry = f"{GPT_OSS_KERNEL_REPO_ID}={vendor_dir.resolve()}"
    existing = os.environ.get("LOCAL_KERNELS", "")
    others = [
        e
        for e in existing.split(":")
        if e and not e.startswith(f"{GPT_OSS_KERNEL_REPO_ID}=")
    ]
    os.environ["LOCAL_KERNELS"] = ":".join([entry] + others)
    return True
