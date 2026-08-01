"""Expert usage histogram (measurement only).

Counts which MoE experts DECODE routes to, so the hot-but-evicted population
can be measured before any hot-expert pin is designed. See
docs/superpowers/specs/2026-08-01-expert-usage-histogram-measurement-design.md.

Ships dark: nothing here is constructed unless enable_expert_usage is set.
"""
import torch


class ExpertUsageTracker:
    """Device-resident [n_rows, n_experts] route counter.

    Rows are indexed by the model's OWN layer_idx (Qwen's MoE layers start at 3),
    not by a zero-based MoE ordinal, so callers can pass layer_idx straight
    through. The few non-MoE rows cost ~48 KB at Qwen geometry and buy simpler,
    less error-prone indexing.
    """

    def __init__(self, n_rows: int, n_experts: int, device):
        self.n_rows = int(n_rows)
        self.n_experts = int(n_experts)
        self.counts = torch.zeros((self.n_rows, self.n_experts),
                                  dtype=torch.int32, device=device)

    def record(self, layer_idx: int, expert_indices: torch.Tensor) -> None:
        """Add one route per entry of expert_indices to row layer_idx.

        Hot path: this must stay a pure device op. No .item()/.cpu()/.tolist()
        and no synchronize() — the histogram is only read in snapshot().
        """
        flat = expert_indices.reshape(-1).to(torch.int64)
        ones = torch.ones_like(flat, dtype=self.counts.dtype)
        self.counts[layer_idx].index_add_(0, flat, ones)

    def snapshot(self) -> torch.Tensor:
        """int32 CPU copy of the counters. The ONLY device->host sync in this
        module; call it after generation, never inside a forward."""
        return self.counts.detach().to("cpu").clone()
