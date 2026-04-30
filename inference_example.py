import torch
from vaayu import VaayuLlama2

# Device selection: GPU by default, falls back to CPU if no GPU available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MAX_LENGTH = 128
# could use hugging face model repo id:
model = VaayuLlama2("garage-bAInd/Platypus2-70B-instruct", device=DEVICE)

# or use model's local path...
#model = VaayuLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f", device=DEVICE)

input_text = [
        'What is the capital of United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=True)

generation_output = model.generate(
    input_tokens['input_ids'].to(DEVICE),
    max_new_tokens=2,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
