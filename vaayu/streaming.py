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

    if getattr(model, 'speculator', None) is not None:
        thread = Thread(target=model.speculative_generate, kwargs=generation_kwargs)
    else:
        thread = Thread(target=model.generate, kwargs=generation_kwargs)

    thread.start()

    for text_chunk in streamer:
        yield text_chunk

    thread.join()
