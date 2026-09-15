"""Persistent buffers a checkpoint has no tensor for: find them, fill them from a
fetched sidecar file, and warn about the rest (CLAUDE.md §5: warn, don't
silently degrade).

A module can register a PERSISTENT buffer -- part of its state_dict, so part of
what the model was trained with -- that a checkpoint does not ship. Nothing on
the load path restores it, so it silently keeps its initial value. The case this
was written from: `NeuroSenko--MiniMax-M2.7-exl3` has no
`e_score_correction_bias` (MiniMax-M2's per-expert routing bias), so every
layer routed with zeros. Non-persistent buffers (rotary `inv_freq`) are derived,
not trained, and are never reported.

The sidecar lives in the SPLIT dir on purpose: `decode_top_k_auto.fingerprint`
globs that dir's `*.safetensors`, so adding or removing the sidecar makes a
cached DECODE_TOP_K="auto" pick re-measure. scripts/fetch_missing_buffers.py
writes it. See documentation/missing_buffers.md.
"""
import json
import struct
from pathlib import Path

import torch

SIDECAR_FILENAME = "vaayu_missing_buffers.safetensors"
FETCH_SCRIPT = "scripts/fetch_missing_buffers.py"

_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
           "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
           "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def _persistent_buffers(model):
    """(full name, owning module, leaf name) for every persistent buffer."""
    for mod_name, mod in model.named_modules():
        for leaf, buf in mod._buffers.items():
            if buf is not None and leaf not in mod._non_persistent_buffers_set:
                yield (f"{mod_name}.{leaf}" if mod_name else leaf), mod, leaf


def _header_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return set(json.loads(f.read(n)))


def missing_persistent_buffers(model, checkpoint_path, layer_names):
    """Persistent buffers whose owning layer file does not carry them.

    A buffer's owner is the LONGEST layer name it starts with plus a dot, so
    `model.layers.1` never claims `model.layers.10`. Only the owners' headers are
    read, once each -- a model with no persistent buffers reads no file. Buffers
    outside every layer have no owning file and are not checked."""
    by_length = sorted(layer_names, key=len, reverse=True)
    headers, missing = {}, []
    for name, _, _ in _persistent_buffers(model):
        owner = next((layer for layer in by_length if name.startswith(layer + ".")), None)
        if owner is None:
            continue
        if owner not in headers:
            path = Path(checkpoint_path) / f"{owner}.safetensors"
            headers[owner] = _header_keys(path) if path.exists() else set()
        if name not in headers[owner]:
            missing.append(name)
    return missing


def apply_sidecar(model, checkpoint_path, missing):
    """Fill `missing` buffers from the sidecar in `checkpoint_path`; returns the
    names applied.

    Keeps the FILE's dtype: MiniMax adds its routing bias to float32 scores, and
    a bf16 cast would round away ~a third of the selection-relevant spread. A
    wrong shape raises -- a sidecar fetched for another model must not load
    half-silently."""
    path = Path(checkpoint_path) / SIDECAR_FILENAME
    if not path.exists():
        return []
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    wanted = set(missing)
    targets = {name: (mod, leaf) for name, mod, leaf in _persistent_buffers(model) if name in wanted}
    applied = []
    for name in missing:
        if name not in tensors:
            continue
        mod, leaf = targets[name]
        old, new = mod._buffers[leaf], tensors[name]
        if tuple(new.shape) != tuple(old.shape):
            raise ValueError(
                f"{path}: {name} has shape {tuple(new.shape)} but the model's buffer is "
                f"{tuple(old.shape)} -- was this sidecar fetched for a different model?")
        mod._buffers[leaf] = new.to(device=old.device)
        applied.append(name)

    ignored = sorted(k for k in tensors if k not in targets)
    if ignored:
        more = " ..." if len(ignored) > 3 else ""
        print(f"[Vaayu] WARNING: {path.name}: ignored {len(ignored)} tensor(s) that are not "
              f"missing buffers of this model: {', '.join(ignored[:3])}{more}")
    if applied:
        print(f"[Vaayu] Loaded {len(applied)} missing buffer(s) from {path}")
    return applied


def warn_missing(names):
    """One WARNING naming what is missing and how to fetch it; silent when none."""
    if not names:
        return
    leaves = ", ".join(sorted({n.rsplit(".", 1)[-1] for n in names}))
    print(f"[Vaayu] WARNING: {len(names)} persistent buffer(s) have no tensor in this checkpoint "
          f"and keep their initial values ({leaves}; e.g. {names[0]}). The model runs, but not "
          f"as trained. Fetch them from the original model repo with:\n"
          f"    ./venv/bin/python {FETCH_SCRIPT} --repo <org/original-model> --checkpoint <model dir>\n"
          f"  then reload. Why this matters: documentation/missing_buffers.md")


def read_remote_tensors(get, weight_map, names):
    """Read only the named tensors from remote safetensors shards.

    `get(filename, start, end)` returns the bytes [start, end] inclusive -- one
    HTTP Range request. Each shard's header is read once; no weight bytes other
    than the named tensors are fetched."""
    headers, out = {}, {}
    for name in names:
        fn = weight_map[name]
        if fn not in headers:
            n = struct.unpack("<Q", get(fn, 0, 7))[0]
            headers[fn] = (8 + n, json.loads(get(fn, 8, 8 + n - 1)))
        base, header = headers[fn]
        meta = header[name]
        start, end = meta["data_offsets"]
        raw = get(fn, base + start, base + end - 1)
        out[name] = torch.frombuffer(bytearray(raw), dtype=_DTYPES[meta["dtype"]]).reshape(
            meta["shape"]).clone()
    return out
