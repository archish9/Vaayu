import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path


_EXPERT_RE = re.compile(r'\.experts\.(\d+)\.')

# A tensor that sits under an `experts.` prefix with NO numeric expert component --
# the fused layout, where one tensor holds every expert on a leading axis.
_FUSED_EXPERTS_RE = re.compile(r'\.experts\.[A-Za-z_][A-Za-z0-9_]*$')


def _meta_to_raw(meta: "ExpertTensorMeta") -> dict:
    return {
        "file_offset": meta.file_offset,
        "byte_length": meta.byte_length,
        "shape": meta.shape,
        "dtype": meta.dtype,
    }


def _raw_to_meta(raw: dict) -> "ExpertTensorMeta":
    return ExpertTensorMeta(
        file_offset=raw["file_offset"],
        byte_length=raw["byte_length"],
        shape=raw["shape"],
        dtype=raw["dtype"],
    )


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

    TWO layouts are detected, both by capability, never by architecture name:

      per-expert -- any key matching r'\\.experts\\.(\\d+)\\.', one tensor per
        expert (Mixtral, DeepSeek, Kimi, Qwen MoE, MiniMax).
      fused -- one tensor per layer carrying EVERY expert on a leading axis
        (`...experts.gate_up_proj_blocks` of shape [E, ...], gpt-oss). Expert
        `i` is a contiguous byte sub-range of it, so `index` describes it the
        same way it describes a per-expert checkpoint; `fused_layer_tensors`
        additionally keeps the whole-tensor meta for kernels that route
        internally and want the unsplit blob.

    A layout neither of those covers still indexes nothing, and
    `_assert_experts_found` turns that into an error rather than an empty index.
    """

    # Cache format. 1 = the bare layer map, which is what every cache already on
    # disk holds; 2 wraps it so a fused checkpoint's whole-tensor metas survive
    # the reload. `load_cached` reads both.
    CACHE_VERSION = 2

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
        # Whole-tensor metas for fused layouts, keyed the same way as `index`.
        # `index` holds per-expert sub-ranges; consumers that must hand a kernel
        # the entire layer (a grouped GEMM routes internally) need the unsplit
        # tensor, and deriving it twice would be two places to get wrong.
        self.fused_layer_tensors: dict[str, dict[str, ExpertTensorMeta]] = {}

    # Total routed-expert count, by config field name. ORDERED and EXPLICIT, not a
    # substring scan: `num_experts_per_tok` is the top-k (10 on Qwen3-Next, not 512)
    # and `shared_expert_intermediate_size` is a width -- a scan for "expert" grabs
    # both and silently believes the wrong number. Verified 2026-09-20 against the
    # four checkpoints on this box.
    _EXPERT_COUNT_FIELDS = ("num_experts", "num_local_experts", "n_routed_experts")

    @classmethod
    def expert_count_in(cls, config) -> int | None:
        """The routed-expert count `config` names, or None. `config` is a parsed
        config.json (a dict) or a transformers config object -- the load's own MoE
        gate (`VaayuBaseModel._detect_moe`) asks this same question of the latter,
        so there is one field list, not two that can disagree (S19 iv)."""
        get = config.get if isinstance(config, dict) else (lambda f: getattr(config, f, None))
        for field in cls._EXPERT_COUNT_FIELDS:
            value = get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

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
        return self.expert_count_in(config)

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
        self.fused_layer_tensors = {}
        expected_experts = self.expected_expert_count()
        for shard_path in sorted(self.checkpoint_path.glob("*.safetensors")):
            layer_name = shard_path.stem
            with open(shard_path, "rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hdr_len))
            data_start = 8 + hdr_len

            for key, meta in hdr.items():
                if key == "__metadata__":
                    continue
                start, end = meta["data_offsets"]

                if self._fused_expert_axis(key, meta["shape"], expected_experts):
                    self._index_fused_tensor(
                        layer_name, key, meta,
                        file_offset=data_start + start,
                        byte_length=end - start,
                        num_experts=expected_experts,
                    )
                    continue

                m = _EXPERT_RE.search(key)
                if m is None:
                    continue
                expert_idx = int(m.group(1))
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

    def _index_fused_tensor(self, layer_name: str, key: str, meta: dict,
                            file_offset: int, byte_length: int,
                            num_experts: int) -> None:
        """Record one fused tensor: the whole blob, plus a range per expert.

        safetensors is row-major and the expert axis leads, so expert `i` owns
        exactly the bytes [i*stride, (i+1)*stride) of this tensor -- a single
        contiguous range needing no new reader capability. `load_expert_sparse`
        already reads arbitrary ranges, and `_group_contiguous_ranges` already
        treats weight-type-grouped checkpoints as the expected case.

        Both views are kept because they serve different consumers: per-expert
        ranges for a forward that streams the routed experts only, the whole
        tensor for a kernel that takes every expert and routes internally.
        """
        self.fused_layer_tensors.setdefault(layer_name, {})[key] = ExpertTensorMeta(
            file_offset=file_offset,
            byte_length=byte_length,
            shape=meta["shape"],
            dtype=meta["dtype"],
        )
        stride, remainder = divmod(byte_length, num_experts)
        if remainder:
            raise RuntimeError(
                f"{self.checkpoint_path}: fused expert tensor {key!r} is "
                f"{byte_length} bytes, which does not divide evenly by "
                f"{num_experts} experts. The leading axis is not the expert axis, "
                f"or the config's expert count is wrong."
            )
        per_expert_shape = list(meta["shape"][1:])
        for expert_idx in range(num_experts):
            self.index.setdefault(layer_name, {}).setdefault(expert_idx, {})[key] = (
                ExpertTensorMeta(
                    file_offset=file_offset + expert_idx * stride,
                    byte_length=stride,
                    shape=per_expert_shape,
                    dtype=meta["dtype"],
                )
            )

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
                    k: _meta_to_raw(v) for k, v in tensors.items()
                }
        payload = {
            "__vaayu_index_version__": self.CACHE_VERSION,
            "index": raw,
            "fused_layer_tensors": {
                layer_name: {k: _meta_to_raw(v) for k, v in tensors.items()}
                for layer_name, tensors in self.fused_layer_tensors.items()
            },
        }
        with open(self.checkpoint_path / self.CACHE_FILENAME, "w") as f:
            json.dump(payload, f)

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
        # Version 1 is the bare layer map. It is still read rather than
        # invalidated: every cache on this box is in it, and rebuilding one
        # means re-scanning the headers of a 24576-expert checkpoint for a
        # layout that checkpoint does not have. A v1 cache has no fused
        # section, which is the correct answer for it.
        if raw.get("__vaayu_index_version__") == self.CACHE_VERSION:
            fused_raw = raw.get("fused_layer_tensors", {})
            raw = raw["index"]
        else:
            fused_raw = {}
        self.index = {}
        self.fused_layer_tensors = {}
        for layer_name, experts in raw.items():
            for eidx_str, tensors in experts.items():
                eidx = int(eidx_str)
                for k, v in tensors.items():
                    self.index.setdefault(layer_name, {}).setdefault(eidx, {})[k] = (
                        _raw_to_meta(v)
                    )
        for layer_name, tensors in fused_raw.items():
            for k, v in tensors.items():
                self.fused_layer_tensors.setdefault(layer_name, {})[k] = _raw_to_meta(v)
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
