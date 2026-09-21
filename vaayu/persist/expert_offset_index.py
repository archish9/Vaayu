import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path


_EXPERT_RE = re.compile(r'\.experts\.(\d+)\.')

# A tensor that sits under an `experts.` prefix with NO numeric expert component --
# the fused layout, where one tensor holds every expert on a leading axis.
_FUSED_EXPERTS_RE = re.compile(r'\.experts\.[A-Za-z_][A-Za-z0-9_]*$')


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
    This covers Mixtral, DeepSeek, Kimi and Qwen MoE. It does NOT cover
    checkpoints that store all experts in one fused tensor per layer (leading
    expert axis, e.g. `...experts.gate_up_proj_blocks` of shape [E, ...]) --
    those match nothing here. `_assert_experts_found` turns that into an error
    rather than an empty index; supporting the layout is a separate change.
    """

    CACHE_FILENAME = "vaayu_expert_offsets.json"

    def __init__(self, checkpoint_path: str, config_path: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        # The config does NOT always sit beside the shards. Vaayu indexes its own
        # splitted_model/ directory, which carries no config.json, while the expert
        # count lives in the ORIGINAL checkpoint dir -- so without this the guard
        # reads no config, expects nothing, and never fires. Defaults to the
        # checkpoint dir, which is the single-directory case the tests construct.
        self.config_path = Path(config_path) if config_path is not None else self.checkpoint_path
        self.index: dict[str, dict[int, dict[str, ExpertTensorMeta]]] = {}

    # Total routed-expert count, by config field name. ORDERED and EXPLICIT, not a
    # substring scan: `num_experts_per_tok` is the top-k (10 on Qwen3-Next, not 512)
    # and `shared_expert_intermediate_size` is a width -- a scan for "expert" grabs
    # both and silently believes the wrong number. Verified 2026-09-20 against the
    # four checkpoints on this box.
    _EXPERT_COUNT_FIELDS = ("num_experts", "num_local_experts", "n_routed_experts")

    def expected_expert_count(self) -> int | None:
        """How many routed experts this checkpoint's config implies, or None.

        None means "this config does not describe a routed-MoE model" -- a dense
        checkpoint, or one with no config at all. Callers must treat None as
        "no expectation", never as zero: a dense model legitimately indexes no
        experts and must not be made to raise.
        """
        config_path = self.config_path / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (OSError, ValueError):
            return None
        for field in self._EXPERT_COUNT_FIELDS:
            value = config.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    @staticmethod
    def _fused_expert_axis(key: str, shape: list, expected: int | None) -> bool:
        """True iff `key` holds EVERY expert on a leading axis.

        Two conditions, both required. The key must sit under an `experts.`
        prefix with no numeric component (`...experts.gate_up_proj_blocks`, not
        `...experts.0.w1.weight`), and its leading dimension must equal the
        expert count the config declares. Shape alone is not enough -- an
        attention tensor can legitimately lead with the same number -- and the
        prefix alone is not enough either, since a fused checkpoint also stores
        per-layer scalars under it.

        Needs at least 2 dimensions: a 1-D tensor under an experts prefix has
        no axis left to slice once the expert axis is consumed.
        """
        if expected is None or len(shape) < 2 or shape[0] != expected:
            return False
        return _FUSED_EXPERTS_RE.search(key) is not None

    def build(self, write_cache: bool = True) -> None:
        from .checkpoint_integrity import assert_checkpoint_complete
        assert_checkpoint_complete(str(self.checkpoint_path))

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

        self._assert_experts_found()

        if write_cache:
            self._write_cache()

    def _assert_experts_found(self) -> None:
        """Refuse an empty index when the config says there should be experts.

        The index matched nothing, so the model would load, stream no experts,
        and produce wrong output with no error anywhere -- the failure this
        guard exists to convert into a message. Deliberately layout-agnostic:
        it asks "config says E, index holds zero", so it stays correct for the
        next layout nobody has seen yet, not just the ones known today.
        """
        expected = self.expected_expert_count()
        if expected is None or self.index:
            return
        seen = []
        for shard_path in sorted(self.checkpoint_path.glob("*.safetensors")):
            with open(shard_path, "rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hdr_len))
            seen = [k for k in hdr if k != "__metadata__" and "expert" in k.lower()]
            if seen:
                break
        raise RuntimeError(
            f"{self.checkpoint_path}: config implies {expected} routed experts, but the "
            f"expert offset index found NONE. Vaayu would stream no experts and produce "
            f"wrong output silently. Expert-ish tensor keys actually present: "
            f"{seen[:4] if seen else '(none)'}. If this checkpoint stores experts as one "
            f"fused tensor per layer rather than as '.experts.<N>.' keys, that layout is "
            f"not indexed here."
        )

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
            # A cache written before this guard existed can hold an empty index.
            # build() cannot catch that -- it never runs on this branch.
            self._assert_experts_found()

    def get_expert_offsets(
        self, layer_name: str, expert_idx: int
    ) -> dict[str, ExpertTensorMeta] | None:
        return self.index.get(layer_name, {}).get(expert_idx)

    def moe_layer_names(self) -> list[str]:
        return list(self.index.keys())
