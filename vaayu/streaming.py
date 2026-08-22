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

    target = (model.speculative_generate
              if getattr(model, 'speculator', None) is not None
              else model.generate)

    # A background-thread exception (e.g. an io_uring read error) otherwise
    # only prints via threading's default excepthook and leaves this generator
    # blocked forever: TextIteratorStreamer's queue.get() has no timeout, and
    # the stop signal that unblocks it is only pushed by generate()'s own
    # normal-completion path (on_finalized_text(stream_end=True)), which a
    # crash never reaches. Catch it, push the stop signal ourselves so the
    # loop below terminates, then re-raise after join() so the caller sees it.
    caught: list[BaseException] = []

    def _run():
        try:
            target(**generation_kwargs)
        except BaseException as e:
            caught.append(e)
            streamer.text_queue.put(streamer.stop_signal)

    # daemon: a non-daemon worker keeps the interpreter alive after SIGTERM
    # until generation finishes -- which is how a killed server went on holding
    # the GPU for the rest of an uncapped run. The caller joins it below on the
    # normal path, so nothing is lost by letting the process win at exit.
    thread = Thread(target=_run, daemon=True)
    thread.start()

    for text_chunk in streamer:
        yield text_chunk

    thread.join()
    if caught:
        raise caught[0]
