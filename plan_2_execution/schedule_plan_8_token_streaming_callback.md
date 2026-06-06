# Schedule Plan 8: Token Streaming Callback

**Phase:** 3 (UX & Stability)
**Priority:** High
**Effort:** Low (3 hours)
**Risk:** Low
**Impact:** Converts a 5-minute blank-screen wait into a responsive typewriter effect, vastly improving perceived performance.

## Objective
Vaayu's `model.generate()` blocks execution until all `max_new_tokens` are fully generated. Given the layer-streaming approach, 20 tokens can take minutes. We need to implement Hugging Face's `TextIteratorStreamer` so the API/CLI can print tokens instantly as they clear the GPU.

## Context for the Coding Agent
The `generate()` function in `vaayu_base.py` wraps the standard transformers generation loop. The simplest way to expose streaming without rewriting the entire generation logic is to provide a helper function that spins up a background thread for `generate()` while yielding from a `TextIteratorStreamer` on the main thread.

## Implementation Steps

### Step 1: Create `vaayu/streaming.py`

Create a new utility module for streaming interactions:

```python
from transformers import TextIteratorStreamer
from threading import Thread
from typing import Generator

def stream_generate(
    model,
    input_ids,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs
) -> Generator[str, None, None]:
    """
    Yields decoded text chunks as the model generates them.
    
    Usage:
        for chunk in stream_generate(model, input_ids):
            print(chunk, end="", flush=True)
    """
    streamer = TextIteratorStreamer(
        model.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        use_cache=True,
        **kwargs
    )
    
    # Run the generation in a background thread so the main thread can yield from the streamer
    if getattr(model, 'speculator', None) is not None:
        # If using speculative decoding, ensure your speculative_generate supports the streamer kwarg
        thread = Thread(target=model.speculative_generate, kwargs=generation_kwargs)
    else:
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        
    thread.start()
    
    for text_chunk in streamer:
        yield text_chunk
        
    thread.join()
```

### Step 2: Support Streamer in `speculative_generate`

**File to modify:** `vaayu/vaayu_base.py`

If `speculative_generate` is called, ensure it processes the `streamer` object. 
In `speculative_generate`, near where tokens are appended to the sequence:

```python
def speculative_generate(self, input_ids, max_new_tokens=128, streamer=None, **kwargs):
    # ... inside the loop where accepted_ids are added:
    if streamer is not None:
        # streamer.put() accepts a 1D tensor of newly generated tokens
        # Since speculative decoding can accept multiple tokens, pass them all
        streamer.put(accepted_ids[0].cpu())
        
    # ... at the very end of the function:
    if streamer is not None:
        streamer.end()
```

*(You must ensure the same `streamer.put()` logic exists if overriding the standard `generate` loop, though `transformers` handles it natively if calling `super().generate`)*.

### Step 3: Implement in `test_quick.py`

**File to modify:** `test_quick.py`

Replace the blocking `output = model.tokenizer.decode(generation_output.sequences[0])` block with:

```python
from vaayu.streaming import stream_generate

print("\nOutput: ", end="", flush=True)
for chunk in stream_generate(
    model, 
    input_tokens['input_ids'].to(DEVICE), 
    max_new_tokens=MAX_NEW_TOKENS
):
    print(chunk, end="", flush=True)
print("\n")
```

## Verification Plan
1. Run `python test_quick.py`.
2. Ensure you see the text outputting one word at a time in the terminal, exactly aligned with the `[VAAYU-IO]` token summary logs.
