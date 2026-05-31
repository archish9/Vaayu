# vaayu/speculative/eagle.py
import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from .base import SpeculativeDecoder


class EAGLEDecoder(SpeculativeDecoder):
    """
    EAGLE-3 self-speculative decoder.

    Path 1 (preferred): delegates to exllamav3.speculative.EAGLEDraftModel.
    Path 2 (fallback): manual weight loader + single transformer layer from safetensors.

    Compatible checkpoints for Llama-3.3 70B:
      yuhuili/EAGLE3-LLaMA3.3-Instruct-70B
      RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3
      nebius/EAGLE3-Llama-3.3-70B-Instruct
    """

    def __init__(self, eagle_head_path: str, base_model, device: str):
        self.device = device
        self.base_model = base_model
        self._exl3_eagle = None
        self._eagle_layer = None
        self._fc = None
        self._weights = {}

        config_path = os.path.join(eagle_head_path, "config.json")
        with open(config_path) as f:
            self.eagle_config = json.load(f)

        # Path 1: ExLlamaV3 native EAGLE support
        try:
            from exllamav3.speculative import EAGLEDraftModel
            self._exl3_eagle = EAGLEDraftModel(eagle_head_path, base_model, device)
            print("[Vaayu-EAGLE] Using ExLlamaV3 native EAGLE support")
            return
        except (ImportError, Exception) as e:
            print(f"[Vaayu-EAGLE] ExLlamaV3 EAGLE unavailable ({e}), using manual loader")

        # Path 2: Manual weight loader
        weight_files = sorted(f for f in os.listdir(eagle_head_path) if f.endswith(".safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No .safetensors files in {eagle_head_path}")

        for wf in weight_files:
            self._weights.update(load_file(os.path.join(eagle_head_path, wf), device=device))

        self._build_manual_layer(base_model)

        total_params = sum(t.numel() for t in self._weights.values())
        print(f"[Vaayu-EAGLE] Manual loader: {total_params / 1e6:.1f}M params, "
              f"{self.vram_usage_bytes() / 1e6:.0f} MB VRAM")

    def _build_manual_layer(self, base_model):
        """
        Reconstruct EAGLE decoder layer from safetensors weights.

        EAGLE-3 checkpoint layout (LLaMA family):
          model.layers.0.*  — one LlamaDecoderLayer
          fc.*              — optional input fusion projection
          embed_tokens.*    — input embeddings (may be absent if tied to base)
        """
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        # Build a 1-layer skeleton using the base model's config
        cfg = base_model.config.__class__(**base_model.config.to_dict())
        cfg.num_hidden_layers = 1

        try:
            with init_empty_weights():
                mini = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"[Vaayu-EAGLE] Could not build EAGLE layer skeleton: {e}. "
                "ExLlamaV3 native EAGLE support is required for this checkpoint."
            ) from e

        # Inject layer 0 weights
        layer_weights = {
            k.replace("model.layers.0.", ""): v
            for k, v in self._weights.items()
            if k.startswith("model.layers.0.")
        }
        if layer_weights:
            try:
                mini.model.layers[0].load_state_dict(layer_weights, strict=False)
                mini.model.layers[0].to(
                    device=self.device, dtype=base_model.running_dtype
                )
                self._eagle_layer = mini.model.layers[0]
            except Exception as e:
                print(f"[Vaayu-EAGLE] Warning: partial layer load: {e}")

        # Optional input fusion layer (fuses token embed + hidden state)
        fc_weight = self._weights.get("fc.weight")
        fc_bias = self._weights.get("fc.bias")
        if fc_weight is not None:
            self._fc = nn.Linear(
                fc_weight.shape[1], fc_weight.shape[0], bias=fc_bias is not None
            )
            self._fc.weight = nn.Parameter(fc_weight)
            if fc_bias is not None:
                self._fc.bias = nn.Parameter(fc_bias)
            self._fc = self._fc.to(device=self.device, dtype=base_model.running_dtype)

    def draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        num_tokens: int = 5,
    ):
        if self._exl3_eagle is not None:
            return self._exl3_eagle.draft(input_ids, past_key_values, num_tokens)

        if self._eagle_layer is None:
            raise RuntimeError(
                "[Vaayu-EAGLE] Layer not initialized. "
                "Verify the eagle_head_path contains valid .safetensors files."
            )

        hidden = self.base_model._last_hidden_state
        if hidden is None:
            raise RuntimeError(
                "[Vaayu-EAGLE] No hidden state captured yet. "
                "Call model.forward() at least once before speculative_generate()."
            )

        drafts = []
        with torch.inference_mode():
            for _ in range(num_tokens):
                if self._fc is not None:
                    # Fuse last hidden state with token embedding
                    token_embed = self.base_model.layers[0](input_ids[:, -1:])
                    fused = self._fc(torch.cat([hidden[:, -1:, :], token_embed], dim=-1))
                    layer_input = fused
                else:
                    layer_input = hidden[:, -1:, :]

                layer_out = self._eagle_layer(layer_input)
                next_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                logits = self.base_model.run_lm_head(
                    self.base_model._lm_head_layer, next_hidden
                )
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                drafts.append(next_token)
                hidden = next_hidden

        draft_ids = torch.cat(drafts, dim=1)
        return draft_ids, {"last_hidden": hidden}

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
        if self._exl3_eagle is not None:
            return getattr(self._exl3_eagle, "vram_usage_bytes", lambda: 0)()
        return sum(t.nbytes for t in self._weights.values())

    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        # EAGLE shares the base model KV cache; VaayuBaseModel._trim_kv_cache handles it.
        if self._exl3_eagle is not None and hasattr(self._exl3_eagle, "rollback_cache"):
            self._exl3_eagle.rollback_cache(draft_state, num_accepted)
