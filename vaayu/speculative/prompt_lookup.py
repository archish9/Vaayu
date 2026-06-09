import torch
from typing import Tuple
from .base import SpeculativeDecoder


class PromptLookupDecoder(SpeculativeDecoder):
    """
    Speculative decoder using Prompt Lookup Decoding (PLD).

    Searches the generation context backwards for n-gram matches of the last
    generated tokens, then drafts the tokens that followed those n-grams in the
    input. Zero VRAM cost; works with any vocabulary.

    Best for: code completion, template continuation, repetitive boilerplate.
    Avoid for: instruction→response tasks — when the model generates novel tokens
      not present in the prompt, match rate drops to 0% and PLD becomes equivalent
      to greedy single-token decoding with negligible extra CPU overhead.

    Accept rates on good-fit workloads: 100–120% (all drafts accepted + bonus token).
    """

    def __init__(self, eos_token_id=None, max_ngram_size: int = 3):
        self.eos_token_ids = (
            eos_token_id if isinstance(eos_token_id, list)
            else ([eos_token_id] if eos_token_id is not None else [])
        )
        self.max_ngram_size = max_ngram_size

    def draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        num_tokens: int = 5,
    ) -> Tuple[torch.LongTensor, dict]:
        input_length = input_ids.size(1)
        if input_length < 2:
            return torch.empty((input_ids.size(0), 0), dtype=input_ids.dtype, device=input_ids.device), {}

        chosen_ids = None
        match_found = False

        for ngram_size in range(min(self.max_ngram_size, input_length - 1), 0, -1):
            windows = input_ids[0, :-1].unfold(dimension=0, size=ngram_size, step=1)
            ngram_tensor = input_ids[0, -ngram_size:]

            matches = (windows == ngram_tensor).all(dim=-1)
            match_indices = matches.nonzero(as_tuple=True)[0]

            for idx in reversed(match_indices):
                start_idx = idx.item() + ngram_size
                end_idx = min(start_idx + num_tokens, input_length)

                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]

                    if self.eos_token_ids:
                        eos_mask = torch.isin(
                            chosen_ids,
                            torch.tensor(self.eos_token_ids, device=chosen_ids.device)
                        )
                        eos_matches = eos_mask.nonzero(as_tuple=True)[0]
                        if eos_matches.numel() > 0:
                            chosen_ids = chosen_ids[:eos_matches[0].item()]

                    match_found = True
                    break

            if match_found:
                break

        if chosen_ids is None or len(chosen_ids) == 0:
            draft_ids = torch.empty((input_ids.size(0), 0), dtype=input_ids.dtype, device=input_ids.device)
        else:
            draft_ids = chosen_ids.unsqueeze(0)

        return draft_ids, {"num_drafted": draft_ids.shape[1]}

    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        num_draft = draft_ids.shape[1]
        if num_draft == 0:
            next_token = verifier_logits[:, 0].argmax(dim=-1, keepdim=True)
            return next_token, 1

        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() != draft_ids[:, i].item():
                correction = verifier_token.unsqueeze(1)
                return torch.cat([draft_ids[:, :i], correction], dim=1), i + 1

        bonus = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        return torch.cat([draft_ids, bonus], dim=1), num_draft + 1

    def vram_usage_bytes(self) -> int:
        return 0

    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        pass
