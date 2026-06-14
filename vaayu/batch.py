import torch


def batch_generate(
    model,
    prompts: list[str],
    max_new_tokens: int = 128,
) -> list[str]:
    """
    Process multiple prompts sharing the same NVMe reads.

    Each layer is loaded from disk once and applied to all N prompts'
    hidden states sequentially, so total time ≈ single-query time +
    (N-1) × compute_overhead instead of N × single-query time.

    Requires sdpa or flash_attention_2 for correct causal masking with
    left-padded batches. With the default eager attention backend, padding
    tokens are not masked and may corrupt output quality.

    Usage:
        results = batch_generate(model, ["prompt A", "prompt B", "prompt C"])
        for text in results:
            print(text)
    """
    tokenizer = model.tokenizer

    # Left padding required: all sequences must align on the right (last real token
    # at the same position) so the model sees consistent position IDs.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
