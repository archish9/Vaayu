# Update test_quick.py for Chat Template and Timing

This plan outlines the changes needed to fix the prompt hallucination by applying the correct Llama-3.3 Instruct chat template, and adds a timer to track exactly how many minutes it takes to generate the answer.

## Proposed Changes

### `test_quick.py`

#### [MODIFY] [test_quick.py](file:///home/a/code/ai-ml/Vaayu/test_quick.py)

**1. Import the Time Module:**
Add `import time` to the top of the script so we can measure the elapsed time.

**2. Apply the Chat Template:**
Replace the raw string `input_text` with a structured messages list and use the tokenizer's `apply_chat_template` method:
```python
messages = [
    {"role": "user", "content": "What is the capital of United States?"}
]
formatted_prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# Pass formatted_prompt into the tokenizer call instead of the raw string.
```

**3. Add a Generation Timer:**
Wrap the `model.generate()` call (or the whole generation block) with a timer. After the output string is decoded, calculate the total minutes and print it.
```python
print("Generating output...")
start_time = time.time()

generation_output = model.generate(
    # ... args ...
)

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60.0

output = model.tokenizer.decode(generation_output.sequences[0])

print(f"\nOutput: {output}")
print(f"Total generation time: {elapsed_minutes:.2f} minutes")
```

## Verification Plan

### Automated Tests
- Run `python3 test_quick.py`.
- Verify that the model outputs a factual answer (e.g., "Washington, D.C.") rather than hallucinating more questions.
- Verify that the terminal logs the exact minutes taken at the end of the script.
