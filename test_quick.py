"""
Quick test without installation
This adds the local path to Python's import system
"""

import sys
import os
import time
import torch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print(f"Using vaayu from: {project_root}")

# Device selection: GPU by default, falls back to CPU if no GPU available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Now import and test
from vaayu import AutoModel

MAX_LENGTH = 128

print("Loading model...")
model = AutoModel.from_pretrained("/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct", device=DEVICE)

messages = [
    {"role": "user", "content": "What is the capital of United States?"}
]
formatted_prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

input_tokens = model.tokenizer(
    formatted_prompt,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False
)

print("Generating output...")
print(input_tokens['input_ids'].to(DEVICE))
start_time = time.time()
generation_output = model.generate(
    input_tokens['input_ids'].to(DEVICE),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)
end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60.0

output = model.tokenizer.decode(generation_output.sequences[0])

print(f"\nOutput: {output}")
print(f"Total generation time: {elapsed_minutes:.2f} minutes")
