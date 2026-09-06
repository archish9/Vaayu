import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


@dataclass
class LayerTiming:
    layer_name: str
    file_bytes: int
    header_parse_us: int = 0
    io_submit_us: int = 0
    io_wait_us: int = 0
    tensor_reconstruct_us: int = 0
    gpu_transfer_us: int = 0
    compute_us: int = 0
    cache_hit: bool = False

    @property
    def total_us(self) -> int:
        return (self.header_parse_us + self.io_submit_us + self.io_wait_us +
                self.tensor_reconstruct_us + self.gpu_transfer_us + self.compute_us)

    @property
    def nvme_throughput_gbps(self) -> float:
        if self.cache_hit or self.io_wait_us == 0:
            return 0.0
        return (self.file_bytes / 1e9) / (self.io_wait_us / 1e6)


@dataclass
class TokenTiming:
    token_index: int
    timestamp: float = field(default_factory=time.time)
    layers: List[LayerTiming] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(l.total_us for l in self.layers) / 1e6

    @property
    def avg_nvme_throughput(self) -> float:
        disk_layers = [l for l in self.layers if not l.cache_hit and l.io_wait_us > 0]
        if not disk_layers:
            return 0.0
        total_bytes = sum(l.file_bytes for l in disk_layers)
        total_us = sum(l.io_wait_us for l in disk_layers)
        return (total_bytes / 1e9) / (total_us / 1e6)


class VaayuTelemetry:
    MAX_TOKENS = 4096  # ring cap: ~97 LayerTiming objects/token; unbounded growth
                       # on long-running sessions is pure leak. 4096 tokens covers
                       # any realistic dump_json() analysis window.

    def __init__(self, output_dir: Optional[str] = None):
        self.tokens: List[TokenTiming] = []
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self._current_token: Optional[TokenTiming] = None

    def begin_token(self, token_index: int):
        self._current_token = TokenTiming(token_index=token_index)

    def record_layer(self, timing: LayerTiming):
        if self._current_token:
            self._current_token.layers.append(timing)

    def end_token(self):
        if self._current_token:
            self.tokens.append(self._current_token)
            if len(self.tokens) > self.MAX_TOKENS:
                del self.tokens[0]
            self._current_token = None

    def record_batch_overlap(self, layer_idx, union_size, max_possible):
        if not hasattr(self, "_batch_overlap"):
            self._batch_overlap = []
        self._batch_overlap.append((layer_idx, union_size, max_possible))

    def reset_batch_overlap(self):
        self._batch_overlap = []

    def batch_overlap_summary(self):
        rows = getattr(self, "_batch_overlap", [])
        ratios = [u / m for _, u, m in rows if m]
        return {"layers": len(rows),
                "mean_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0}

    def dump_json(self, path: str):
        filepath = self.output_dir / path
        with open(filepath, 'w') as f:
            json.dump([asdict(t) for t in self.tokens], f, indent=2)
