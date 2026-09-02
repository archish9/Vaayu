from abc import ABC, abstractmethod
from typing import Tuple
import torch


class SpeculativeDecoder(ABC):

    @abstractmethod
    def draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        num_tokens: int = 5,
    ) -> Tuple[torch.LongTensor, object]:
        """
        Generate num_tokens draft candidates.

        Returns:
            draft_ids: (batch, num_tokens) — drafted token IDs
            draft_state: opaque state needed for cache rollback on rejection
        """

    @abstractmethod
    def verify_and_accept(
        self,
        input_ids: torch.LongTensor,
        draft_ids: torch.LongTensor,
        verifier_logits: torch.Tensor,
        draft_state: object,
    ) -> Tuple[torch.LongTensor, int]:
        """
        Accept the longest prefix where verifier agrees with drafter.

        verifier_logits shape: (batch, 1+num_draft, vocab_size)
          - logits[:, i] is the verifier's prediction at draft position i
          - logits[:, num_draft] is a bonus token if all drafts accepted

        Returns:
            accepted_ids: (batch, accepted_len) — tokens to append to sequence
            num_accepted: 1 to num_draft+1 (includes bonus or correction)
        """

    @abstractmethod
    def vram_usage_bytes(self) -> int:
        """VRAM consumed by this decoder (bytes)."""

    @abstractmethod
    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        """Trim drafter KV cache so it aligns with accepted token count."""

    def reset(self) -> None:
        """Drop all state carried between generations.

        rollback_cache aligns the drafter WITHIN one generation loop; this
        clears what must not survive INTO the next one, where input_ids is a
        different sequence. Default no-op: a stateless decoder (prompt lookup)
        has nothing to drop and must still answer the call.
        """
