"""Fetch persistent buffers a local checkpoint is missing from the original model repo.

Some converted checkpoints drop tensors the model was trained with. The case this
was written for: `NeuroSenko--MiniMax-M2.7-exl3` has no `e_score_correction_bias`
(MiniMax-M2's per-expert routing bias), so it routes with zeros. Vaayu prints a
WARNING at load naming what is missing; this script fixes it.

It builds the model skeleton on the meta device (no weights), finds the
persistent buffers the split checkpoint lacks, then downloads ONLY those tensors
from the Hugging Face repo you name -- the shard header plus each tensor's bytes,
by HTTP Range request -- and writes them to `vaayu_missing_buffers.safetensors`
in the split dir. Vaayu applies that file on the next load. Delete the file to go
back. The tensors are part of the upstream model's weights and carry ITS license.

Usage:
    PYTHONPATH=/home/a/code/ai-ml/Vaayu ./venv/bin/python scripts/fetch_missing_buffers.py \
        --repo MiniMaxAI/MiniMax-M2.7 \
        --checkpoint /home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3
Optional: --revision <branch|tag|sha> (default: main, pinned to its commit sha);
          HF_TOKEN=<token> for gated repos.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vaayu.missing_buffers import (  # noqa: E402
    SIDECAR_FILENAME, missing_persistent_buffers, read_remote_tensors,
)

HF = "https://huggingface.co"


def _open(url, headers=None):
    h = dict(headers or {})
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120)


def split_dir(checkpoint):
    """The split dir Vaayu loads from: `<checkpoint>/splitted_model`, or the
    checkpoint itself when it already holds per-layer files."""
    p = Path(checkpoint)
    for cand in (p / "splitted_model", p):
        if (cand / "model.layers.0.safetensors").exists() or (cand / "lm_head.safetensors").exists():
            return cand
    sys.exit(f"no split checkpoint under {p} -- load it with Vaayu once first")


def build_skeleton(checkpoint):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from vaayu.vaayu_base import VaayuBaseModel

    VaayuBaseModel._apply_transformers_compat_patches()
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="original model repo, e.g. MiniMaxAI/MiniMax-M2.7")
    ap.add_argument("--checkpoint", required=True, help="the local model dir Vaayu loads")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    split = split_dir(args.checkpoint)
    layer_names = sorted(p.stem for p in split.glob("*.safetensors") if p.name != SIDECAR_FILENAME)
    missing = missing_persistent_buffers(build_skeleton(args.checkpoint), split, layer_names)
    if not missing:
        print("No persistent buffers are missing -- nothing to fetch.")
        return
    print(f"{len(missing)} missing buffer(s), e.g. {missing[0]}")

    sha = json.load(_open(f"{HF}/api/models/{args.repo}/revision/{args.revision}"))["sha"]
    print(f"{args.repo} @ {sha}")
    try:
        weight_map = json.load(_open(f"{HF}/{args.repo}/resolve/{sha}/model.safetensors.index.json"))[
            "weight_map"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        weight_map = None  # single-file repo

    def get(filename, start, end):
        data = _open(f"{HF}/{args.repo}/resolve/{sha}/{filename}",
                     {"Range": f"bytes={start}-{end}"}).read()
        if len(data) != end - start + 1:
            sys.exit(f"{filename}: asked for {end - start + 1} bytes, got {len(data)} -- "
                     f"the server ignored the Range request; stopping rather than downloading a shard")
        return data

    if weight_map is None:
        weight_map = {name: "model.safetensors" for name in missing}
    absent = [n for n in missing if n not in weight_map]
    if absent:
        sys.exit(f"{args.repo} does not ship {len(absent)} of them either (e.g. {absent[0]}) -- "
                 f"is this the original model repo?")

    tensors = read_remote_tensors(get, weight_map, missing)
    from safetensors.torch import save_file
    out = split / SIDECAR_FILENAME
    save_file(tensors, str(out), metadata={"source_repo": args.repo, "source_sha": sha})
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({out.stat().st_size} bytes, sha256 {digest})")
    print(f"These tensors come from {args.repo} and carry its license. "
          f"Reload the model to apply them; delete the file to go back.")


if __name__ == "__main__":
    main()
