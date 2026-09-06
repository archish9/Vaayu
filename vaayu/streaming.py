from transformers import TextIteratorStreamer
from threading import Thread
from typing import Generator


class _IdCountingStreamer(TextIteratorStreamer):
    """A TextIteratorStreamer that also records the real generated token ids.

    Callers that need a token COUNT cannot get one by re-tokenizing the streamed
    text: it is decoded with skip_special_tokens=True, so EOS is gone, and the
    merges differ across a decode boundary. generate() already hands every
    generated id to put(), so recording them here is exact and costs nothing.
    """

    def __init__(self, tokenizer, sink: list, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self._sink = sink

    def put(self, value):
        # Read the prompt flag BEFORE super().put(), which clears it and returns
        # early. Recording after would count the whole prompt as generated.
        is_prompt = self.skip_prompt and self.next_tokens_are_prompt
        super().put(value)
        if is_prompt:
            return
        # Same normalisation TextStreamer.put does: a (1, N) batch -- what
        # speculative_generate puts for a whole accepted run -- becomes (N,).
        ids = value[0] if getattr(value, "ndim", 1) > 1 else value
        self._sink.extend(ids.tolist())


def stream_generate(
    model,
    input_ids,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    token_ids_out: list | None = None,
    **kwargs
) -> Generator[str, None, None]:
    """
    Yields decoded text chunks as the model generates them.

    Usage:
        for chunk in stream_generate(model, input_ids):
            print(chunk, end="", flush=True)

    Speculation is dispatched PER CALL: a loaded speculator is used only when
    this call asks for nothing its greedy, stop-string-less loop would drop.
    Note the sampling defaults below -- a caller that wants speculation must
    pass `temperature=0.0` explicitly.

    Pass `token_ids_out` to also collect the real generated token ids into it --
    the only honest source for a token count, since the yielded text cannot be
    re-tokenized back into one.
    """
    if token_ids_out is None:
        streamer = TextIteratorStreamer(
            model.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
    else:
        streamer = _IdCountingStreamer(
            model.tokenizer,
            token_ids_out,
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

    # Per-request dispatch, not per-process. `speculative_generate` is greedy
    # at every decision point and has no stop-string machinery, so a loaded
    # speculator used to silently change EVERY request: a sampled one came back
    # greedy, a tool-enabled one generated straight past its own stop string.
    # Imported here, not at module scope: `vaayu_base` pulls the whole model
    # stack in, and this module is imported for its streamer alone. Same
    # local-import pattern as `engine.notice_if_speculation_skipped`.
    from .vaayu_base import choose_speculative_target
    target = choose_speculative_target(model, temperature, generation_kwargs)

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
