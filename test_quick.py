"""
Quick test without installation
This adds the local path to Python's import system
"""

import sys
import os

# Add local airllm to path
project_root = os.path.dirname(os.path.abspath(__file__))
airllm_path = os.path.join(project_root, 'air_llm')
sys.path.insert(0, airllm_path)

print(f"Using airllm from: {airllm_path}")

# Now import and test
from airllm import AutoModel

MAX_LENGTH = 128

print("Loading model...")
model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B")

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
