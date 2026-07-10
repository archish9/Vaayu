# vaayu/speculative/standard.py
import torch
from transformers import AutoModelForCausalLM
from .base import SpeculativeDecoder


def drafter_rollback_target(seen_before_draft: int, num_drafted: int, num_accepted: int) -> int:
    """Tokens of drafter KV cache that remain valid after verification (Fix #9).

    During draft(), the drafter feeds its own predictions one at a time, so after
    drafting k tokens its cache covers seen_before_draft + (k - 1) positions (the
    last draft is predicted, not fed). The verifier accepts `a` tokens: a-1 drafts
    plus one token the drafter did not produce (correction or bonus). The drafted
    prefix that matches the accepted sequence is a-1 tokens long, but the cache
    cannot keep more than it fed: min(a - 1, k - 1) positions past the base.
    """
    kept_drafts = max(0, min(num_accepted - 1, num_drafted - 1))
    return seen_before_draft + kept_drafts


class StandardDrafterDecoder(SpeculativeDecoder):
    """
    Speculative decoder using a resident small HuggingFace model as drafter.

    The drafter lives permanently in VRAM. Must share the same tokenizer
    as the Vaayu verifier model.

    Recommended drafters for 16 GB GPU:
      Qwen2.5-0.5B (q4):        ~300 MB VRAM
      Qwen2.5-Coder-1.5B (q4):  ~1.0 GB VRAM
    """

    def __init__(self, drafter_path: str, device: str, dtype=torch.float16):
        self.device = device
        print(f"[Vaayu-Drafter] Loading: {drafter_path}")
        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.drafter.eval()
        self._vram_bytes = sum(p.nbytes for p in self.drafter.parameters())
        n_params = sum(p.numel() for p in self.drafter.parameters()) / 1e6
        print(f"[Vaayu-Drafter] {n_params:.0f}M params, {self._vram_bytes / 1e9:.2f} GB VRAM")
        self._drafter_past = None
        self._drafter_seen = 0

    def draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        num_tokens: int = 5,
    ):
        drafts = []
        with torch.no_grad():
            # Feed everything the drafter has not seen yet (prompt on first call,
            # correction/bonus tokens on later calls). Fix #9: replaces the
            # reset-to-None + full re-prefill pattern.
            unseen = input_ids if self._drafter_past is None else input_ids[:, self._drafter_seen:]
            if unseen.shape[1] == 0:
                unseen = input_ids[:, -1:]
            out = self.drafter(input_ids=unseen, use_cache=True,
                               past_key_values=self._drafter_past)
            self._drafter_past = out.past_key_values
            self._drafter_seen = input_ids.shape[1]
            next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            drafts.append(next_token)

            for _ in range(num_tokens - 1):
                out = self.drafter(input_ids=drafts[-1],
                                   past_key_values=self._drafter_past, use_cache=True)
                self._drafter_past = out.past_key_values
                self._drafter_seen += 1
                drafts.append(out.logits[:, -1].argmax(dim=-1, keepdim=True))

        draft_ids = torch.cat(drafts, dim=1)
        return draft_ids, {"num_drafted": len(drafts),
                           "seen_before_draft": input_ids.shape[1]}

    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        num_draft = draft_ids.shape[1]
        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() != draft_ids[:, i].item():
                correction = verifier_token.unsqueeze(1)
                return torch.cat([draft_ids[:, :i], correction], dim=1), i + 1
        bonus = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        return torch.cat([draft_ids, bonus], dim=1), num_draft + 1

    def vram_usage_bytes(self) -> int:
        return self._vram_bytes

    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        """Crop the drafter cache to the accepted prefix (Fix #9). Never resets
        to None — the old full re-prefill was O(seq) per rejection."""
        if self._drafter_past is None:
            return
        target = drafter_rollback_target(
            draft_state.get("seen_before_draft", 0),
            draft_state.get("num_drafted", 0),
            num_accepted,
        )
        if hasattr(self._drafter_past, "crop"):
            self._drafter_past.crop(target)
            self._drafter_seen = target
        else:
            # Cache type without crop support: fall back to the old reset.
            self._drafter_past = None
            self._drafter_seen = 0
