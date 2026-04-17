"""
Test script with smaller model first
"""

import sys
import os

# Add local airllm to path
project_root = os.path.dirname(os.path.abspath(__file__))
airllm_path = os.path.join(project_root, 'air_llm')
sys.path.insert(0, airllm_path)

print(f"Using airllm from: {airllm_path}")

from airllm import AutoModel

MAX_LENGTH = 128

# Start with a very small model to test
print("Testing with Qwen2.5-0.5B (smallest model)...")
try:
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    input_text = ['What is the capital of United States?']
    
    input_tokens = model.tokenizer(
        input_text,
        return_tensors="pt", 
        return_attention_mask=False, 
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding=False
    )
               
    print("Generating output...")
    generation_output = model.generate(
        input_tokens['input_ids'].cuda(), 
        max_new_tokens=20,
        use_cache=True,
        return_dict_in_generate=True
    )
    
    output = model.tokenizer.decode(generation_output.sequences[0])
    
    print(f"\n✅ SUCCESS!")
    print(f"Output: {output}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
