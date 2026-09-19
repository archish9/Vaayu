import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path


_EXPERT_RE = re.compile(r'\.experts\.(\d+)\.')


@dataclass
class ExpertTensorMeta:
    file_offset: int   # absolute byte offset from start of file
    byte_length: int   # byte length of tensor data
    shape: list        # tensor shape e.g. [4096, 4096]
    dtype: str         # safetensors dtype string e.g. "BF16"


class ExpertOffsetIndex:
    """
    Scans safetensors shard headers to build a byte-offset map for every expert
    tensor in every MoE layer. Persisted as JSON alongside the checkpoint to
    avoid re-scanning on every startup.

    index layout:
        {layer_name: {expert_idx: {tensor_key: ExpertTensorMeta}}}

    Detection: any tensor key matching r'\\.experts\\.(\\d+)\\.' is an expert tensor.
    This covers Mixtral, DeepSeek, Kimi, Qwen MoE, and any future architecture
    that follows the `.experts.N.` naming convention.
    """

    CACHE_FILENAME = "vaayu_expert_offsets.json"

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.index: dict[str, dict[int, dict[str, ExpertTensorMeta]]] = {}

    def build(self, write_cache: bool = True) -> None:
        self.index = {}
        for shard_path in sorted(self.checkpoint_path.glob("*.safetensors")):
            layer_name = shard_path.stem
            with open(shard_path, "rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hdr_len))
            data_start = 8 + hdr_len

            for key, meta in hdr.items():
                if key == "__metadata__":
                    continue
                m = _EXPERT_RE.search(key)
                if m is None:
                    continue
                expert_idx = int(m.group(1))
                start, end = meta["data_offsets"]
                self.index.setdefault(layer_name, {}).setdefault(expert_idx, {})[key] = (
                    ExpertTensorMeta(
                        file_offset=data_start + start,
                        byte_length=end - start,
                        shape=meta["shape"],
                        dtype=meta["dtype"],
                    )
                )

        if write_cache:
            self._write_cache()

    def _write_cache(self) -> None:
        raw = {}
        for layer_name, experts in self.index.items():
            raw[layer_name] = {}
            for eidx, tensors in experts.items():
                raw[layer_name][str(eidx)] = {
                    k: {
                        "file_offset": v.file_offset,
                        "byte_length": v.byte_length,
                        "shape": v.shape,
                        "dtype": v.dtype,
                    }
                    for k, v in tensors.items()
                }
        with open(self.checkpoint_path / self.CACHE_FILENAME, "w") as f:
            json.dump(raw, f)

    def load_cached(self) -> bool:
        cache_path = self.checkpoint_path / self.CACHE_FILENAME
        if not cache_path.exists():
            return False
        shards = list(self.checkpoint_path.glob("*.safetensors"))
        if not shards:
            return False
        newest_shard_mtime = max(s.stat().st_mtime for s in shards)
        if cache_path.stat().st_mtime < newest_shard_mtime:
            return False
        with open(cache_path) as f:
            raw = json.load(f)
        self.index = {}
        for layer_name, experts in raw.items():
            for eidx_str, tensors in experts.items():
                eidx = int(eidx_str)
                for k, v in tensors.items():
                    self.index.setdefault(layer_name, {}).setdefault(eidx, {})[k] = (
                        ExpertTensorMeta(
                            file_offset=v["file_offset"],
                            byte_length=v["byte_length"],
                            shape=v["shape"],
                            dtype=v["dtype"],
                        )
                    )
        return True

    def build_or_load(self) -> None:
        if not self.load_cached():
            print("[Vaayu-MoE] Building expert offset index...")
            self.build()
            n = sum(len(e) for e in self.index.values())
            print(f"[Vaayu-MoE] Expert index built: {n} expert entries across "
                  f"{len(self.index)} MoE layers")
        else:
            print("[Vaayu-MoE] Expert offset index loaded from cache")

    def get_expert_offsets(
        self, layer_name: str, expert_idx: int
    ) -> dict[str, ExpertTensorMeta] | None:
        return self.index.get(layer_name, {}).get(expert_idx)

    def moe_layer_names(self) -> list[str]:
        return list(self.index.keys())
