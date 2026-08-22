"""EngineHandle -- the one loaded model, the admission gate, and generation.

Concurrency is 1 by construction. The engine is not thread-safe, and N>=2 at
long context is a settled hardware refusal on this box (peak 14.17-15.57 GiB of
15.47), so this is a depth-bounded FIFO gate, not a scheduler. `batched_generate`
is deliberately not wired in: it is static-batch v1 and would admit nothing here.
"""
import asyncio
import queue as _queue
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .chat import render_prompt
from .stopping import finish_reason, resolve_max_tokens
from .tools import detect_tool_format, parse_tool_calls


# Keyword arguments vaayu.streaming.stream_generate constructs itself. Anything
# here that also arrives via **kwargs is a duplicate-keyword TypeError. Kept as a
# named set so a change to stream_generate's signature has one place to land.
STREAM_GENERATE_OWNS = ("input_ids", "streamer", "max_new_tokens", "temperature",
                        "top_p", "do_sample", "use_cache")


class QueueFull(RuntimeError):
    """More requests are waiting than the configured depth allows."""


class CancelToken:
    """Set from the event loop, read from the generation thread.

    A blocking generate() in an executor thread cannot be killed from outside,
    so cancellation is cooperative: this flag is checked once per token. Without
    it a client that walks away leaves generation running -- measured holding
    the GPU and the single slot with no connection left on the port.
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


def cancel_criteria(token: CancelToken):
    """A transformers StoppingCriteriaList that stops when `token` is cancelled.

    Imports live here rather than at module scope so `import vaayu.server` stays
    as cheap as it was. The criteria must answer per batch ROW -- StoppingCriteria
    List ORs the results into a (batch,) tensor, and a bare bool works right up
    until something batches.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Cancelled(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), token.cancelled,
                              dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([_Cancelled()])


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    tool_calls: list = field(default_factory=list)


class EngineHandle:
    def __init__(self, model, queue_depth: int = 8):
        self.model = model
        self.queue_depth = queue_depth
        self._lock = asyncio.Lock()
        self._inflight = 0
        # The token of the request currently generating, so shutdown can stop
        # it. Concurrency is 1 by construction, so one slot is the whole set.
        self._live_cancel = None

        # vaayu_base falls back to max_seq_len=512 when it built no VramPlan
        # (CPU / no CUDA). This server defaults the generation cap to "the
        # remaining window", so that fallback would silently reinstate exactly
        # the fixed 512-token cap the server exists to remove. Name it loudly.
        if getattr(model, "_vram_plan", None) is None:
            print(f"[Vaayu-Server] WARNING: this model has no VRAM plan (CPU or "
                  f"no CUDA), so max_seq_len is vaayu_base's fallback of "
                  f"{self.max_seq_len}. Every request's default token cap is "
                  f"bounded by it. Load on CUDA for a derived window.")

    # ── properties read off the model, never cached ──────────────────────────

    @property
    def max_seq_len(self) -> int:
        """Derived at load by plan_vram; never a constant in this package."""
        return self.model.max_seq_len

    @property
    def tool_format(self):
        return detect_tool_format(getattr(self.model.tokenizer, "chat_template", None))

    @property
    def depth(self) -> int:
        return self._inflight

    @property
    def live_cancel(self):
        """The running request's CancelToken, or None when idle."""
        return self._live_cancel

    def cancel_all(self) -> None:
        """Stop whatever is generating. Idempotent, and a no-op when idle."""
        token = self._live_cancel
        if token is not None:
            token.cancel()

    def prefix_stats(self) -> str:
        """One grep-able line from the prefix pool, or a note that it is off.

        Hit rate here is the difference between ~1 s and ~98 s TTFT, and there
        is otherwise no way to observe it from outside the process.
        """
        pc = getattr(self.model, "_prefix_cache", None)
        if pc is None:
            return "disabled"
        try:
            return pc.stats()
        except Exception as e:
            return f"unavailable ({type(e).__name__}: {e})"

    # ── admission ────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def slot(self):
        """Admit one request. Raises QueueFull past `queue_depth`.

        The counter is incremented BEFORE awaiting the lock, so it counts
        waiting requests as well as the running one -- a client's queue position
        is `depth` at entry.
        """
        if self._inflight >= self.queue_depth:
            raise QueueFull(
                f"{self._inflight} requests are already queued (depth "
                f"{self.queue_depth}). This engine runs one request at a time.")
        self._inflight += 1
        try:
            async with self._lock:
                yield
        finally:
            self._inflight -= 1

    # ── request mapping ──────────────────────────────────────────────────────

    def prepare(self, req):
        """(prompt_ids, resolved_cap). Raises ContextTooLong / TemplateLacksToolSupport."""
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        tools = [t.model_dump(exclude_none=True) for t in req.tools] if req.tools else None
        prompt_ids = render_prompt(self.model.tokenizer, messages, tools=tools)
        cap = resolve_max_tokens(req.max_tokens, len(prompt_ids), self.max_seq_len)
        return prompt_ids, cap

    def warn_if_speculative_sampling_ignored(self, req):
        """speculative_generate honours `temperature` only; top_p/top_k/stop fall
        into **kwargs and are silently discarded. Return a message when this
        request asks for something that path will drop, else None."""
        if getattr(self.model, "speculator", None) is None:
            return None
        ignored = []
        if req.top_p != 1.0:
            ignored.append("top_p")
        if req.top_k != 0:
            ignored.append("top_k")
        if req.stop:
            ignored.append("stop")
        if not ignored:
            return None
        return (f"speculative decoding is active and its loop honours temperature "
                f"only, so {', '.join(ignored)} will be IGNORED for this request. "
                f"Load without drafter_model_path to use them.")

    def build_generate_kwargs(self, req, prompt_ids, cap=None, cancel=None) -> dict:
        import torch
        kwargs = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long,
                                      device=self.model.running_device),
            "max_new_tokens": cap if cap is not None else 1,
            "use_cache": True,
            # session_id, else OpenAI's `user`, else None -> auto-prefix-match.
            "session_id": req.session_key,
        }
        if req.temperature > 0.0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = req.temperature
            kwargs["top_p"] = req.top_p
            kwargs["top_k"] = req.top_k
        else:
            kwargs["do_sample"] = False

        stops = list(req.stop)
        fmt = self.tool_format
        if req.tools and fmt is not None:
            # Halt AT the call rather than running on to hallucinate its result.
            for s in fmt.stop_strings():
                if s not in stops:
                    stops.append(s)
        if stops:
            kwargs["stop_strings"] = stops
            kwargs["tokenizer"] = self.model.tokenizer   # HF requires it with stop_strings
        if cancel is not None:
            # Honoured by transformers' loop AND by speculative_generate, which
            # has no criteria machinery of its own and checks this per round.
            kwargs["stopping_criteria"] = cancel_criteria(cancel)
        return kwargs

    @asynccontextmanager
    async def _running(self, cancel):
        """Publish this request's token for the lifetime of the generation."""
        self._live_cancel = cancel
        try:
            yield
        finally:
            self._live_cancel = None

    # ── generation ───────────────────────────────────────────────────────────

    def _decode_new(self, out_ids, prompt_len):
        new_ids = out_ids[0][prompt_len:].tolist()
        text = self.model.tokenizer.decode(new_ids, skip_special_tokens=True)
        return new_ids, text

    def _finish(self, new_ids, cap, tool_calls):
        eos = self.model.config.eos_token_id
        eos_set = set(eos) if isinstance(eos, (list, tuple, set)) else {eos}
        return finish_reason(
            hit_eos=bool(new_ids) and new_ids[-1] in eos_set,
            # HF trims a matched stop string before returning, so the returned
            # ids cannot be inspected for one. `produced < cap` below covers it:
            # the only distinction a client acts on is length vs stop.
            hit_stop_string=len(new_ids) < cap,
            hit_tool_call=bool(tool_calls),
            produced=len(new_ids), cap=cap)

    async def generate(self, req, cancel=None) -> GenerationResult:
        prompt_ids, cap = self.prepare(req)
        if req.max_tokens is None:
            # Non-streaming sends NOTHING until the last token, so this request
            # can sit silent for as long as `cap` tokens take -- measured at 15+
            # minutes and counting on MiniMax-M2.7 before a client gave up and
            # the run looked hung. It is not hung; it is uncapped.
            print(f"[Vaayu-Server] WARNING: non-streaming request with no "
                  f"max_tokens: up to {cap} tokens will be generated before ANY "
                  f"bytes are sent. Set max_tokens, or use stream=true.")
        kwargs = self.build_generate_kwargs(req, prompt_ids, cap, cancel=cancel)
        loop = asyncio.get_running_loop()
        async with self._running(cancel):
            out = await loop.run_in_executor(
                None, lambda: self.model.generate(**kwargs))
        new_ids, text = self._decode_new(out, len(prompt_ids))
        cleaned, calls = parse_tool_calls(text, self.tool_format if req.tools else None)
        return GenerationResult(
            text=cleaned, tool_calls=calls,
            prompt_tokens=len(prompt_ids), completion_tokens=len(new_ids),
            finish_reason=self._finish(new_ids, cap, calls))

    async def stream(self, req, cancel=None):
        """Async generator yielding ("delta", text) then ("final", GenerationResult).

        Bridges vaayu.streaming's worker thread onto the event loop through a
        thread-safe queue.
        """
        from ..streaming import stream_generate

        prompt_ids, cap = self.prepare(req)
        kwargs = self.build_generate_kwargs(req, prompt_ids, cap, cancel=cancel)
        input_ids = kwargs.pop("input_ids")
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 1.0)
        # stream_generate builds these itself; leaving any of them in **kwargs is
        # a duplicate-keyword TypeError, which is how `use_cache` shipped broken
        # past every fake-backed test and was caught only on-device.
        for k in STREAM_GENERATE_OWNS:
            kwargs.pop(k, None)

        loop = asyncio.get_running_loop()
        bridge: _queue.Queue = _queue.Queue()
        DONE = object()

        def _pump():
            try:
                for chunk in stream_generate(self.model, input_ids,
                                             max_new_tokens=cap,
                                             temperature=temperature,
                                             top_p=top_p, **kwargs):
                    bridge.put(("delta", chunk))
            except BaseException as e:              # noqa: BLE001 -- re-raised below
                bridge.put(("error", e))
            finally:
                bridge.put((DONE, None))

        thread = threading.Thread(target=_pump, daemon=True)
        self._live_cancel = cancel
        thread.start()

        parts, error = [], None
        try:
            while True:
                kind, payload = await loop.run_in_executor(None, bridge.get)
                if kind is DONE:
                    break
                if kind == "error":
                    error = payload
                    break
                parts.append(payload)
                yield ("delta", payload)
        finally:
            self._live_cancel = None
        thread.join()
        if error is not None:
            raise error

        full = "".join(parts)
        cleaned, calls = parse_tool_calls(full, self.tool_format if req.tools else None)
        produced = len(self.model.tokenizer(full, add_special_tokens=False)["input_ids"])
        yield ("final", GenerationResult(
            text=cleaned, tool_calls=calls,
            prompt_tokens=len(prompt_ids), completion_tokens=produced,
            finish_reason=finish_reason(
                hit_eos=produced < cap, hit_stop_string=False,
                hit_tool_call=bool(calls), produced=produced, cap=cap)))
