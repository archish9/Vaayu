"""
Quick test without installation
This adds the local path to Python's import system
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print(f"Using vaayu from: {project_root}")

# Now import and test
from vaayu import AutoModel

MAX_LENGTH = 128

print("Loading model...")
model = AutoModel.from_pretrained("/media/a/MyPassport/AI-ML/LLM-models/gpt-oss-20b")

input_text = ['What is the capital of United States?']

input_tokens = model.tokenizer(
    "What is the capital of United States?",
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False
)

print("Generating output...")
print(input_tokens['input_ids'].cuda())
generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])

print(f"\nOutput: {output}")
