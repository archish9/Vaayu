"""S14 E -- the embedding table lives on the CPU; only its rows cross PCIe.

A forward reads one row of the embedding per token, yet the table is the size of
`lm_head` (1.158 GB on gpt-oss-20b). Held in VRAM it spends that much of the pin
budget on every model; streamed (gpt-oss, whose eager reserve floor leaves no
budget for it) it is re-read and re-uploaded whole on every decode token -- 163 of
482 ms, measured in S14 steps 1-2. The HF module keeps running its OWN forward on
the CPU copy, so a scaled subclass still scales, and a caller that post-processes
the output (DeepSeek-V4's `run_embed` repeat) receives the same tensor on the GPU.
"""
import torch


def cpu_embedding_eligibility(module, tensors: dict) -> tuple[bool, str]:
    """Can `module` run on the CPU from `tensors` (its checkpoint shard)?

    Refuses rather than guesses: anything but one floating-point weight for a
    plain `nn.Embedding` (subclasses included) keeps today's VRAM-pin / stream
    path. `max_norm` is refused because F.embedding renormalises the weight IN
    PLACE when it is set.
    """
    if not isinstance(module, torch.nn.Embedding):
        return False, f"module is {type(module).__name__}, not an nn.Embedding"
    if getattr(module, "max_norm", None) is not None:
        return False, "max_norm is set: F.embedding would renormalise the weight in place"
    if len(tensors) != 1:
        return False, f"the shard holds {len(tensors)} tensors, expected one weight"
    (t,) = tensors.values()
    if not t.is_floating_point():
        return False, f"weight dtype {t.dtype} is not floating point"
    if tuple(t.shape) != tuple(module.weight.shape):
        return False, (f"weight shape {tuple(t.shape)} does not match the module's "
                       f"{tuple(module.weight.shape)}")
    return True, "eligible"


class CpuResidentEmbedding(torch.nn.Module):
    """Runs `inner` (an nn.Embedding whose weight is on the CPU) and returns its
    output on `target_device`. Replaces the embedding in VaayuBaseModel.layers."""

    def __init__(self, inner: torch.nn.Embedding, target_device):
        super().__init__()
        self.inner = inner
        self.target_device = torch.device(target_device)

    def forward(self, input_ids):
        return self.inner(input_ids.to("cpu")).to(self.target_device)
