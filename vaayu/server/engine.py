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


class QueueFull(RuntimeError):
    """More requests are waiting than the configured depth allows."""


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

    def build_generate_kwargs(self, req, prompt_ids, cap=None) -> dict:
        import torch
        kwargs = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long,
                                      device=self.model.running_device),
            "max_new_tokens": cap if cap is not None else 1,
            "use_cache": True,
            "session_id": req.session_id,     # None -> auto-prefix-match
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
        return kwargs

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

    async def generate(self, req) -> GenerationResult:
        prompt_ids, cap = self.prepare(req)
        kwargs = self.build_generate_kwargs(req, prompt_ids, cap)
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, lambda: self.model.generate(**kwargs))
        new_ids, text = self._decode_new(out, len(prompt_ids))
        cleaned, calls = parse_tool_calls(text, self.tool_format if req.tools else None)
        return GenerationResult(
            text=cleaned, tool_calls=calls,
            prompt_tokens=len(prompt_ids), completion_tokens=len(new_ids),
            finish_reason=self._finish(new_ids, cap, calls))

    async def stream(self, req):
        """Async generator yielding ("delta", text) then ("final", GenerationResult).

        Bridges vaayu.streaming's worker thread onto the event loop through a
        thread-safe queue.
        """
        from ..streaming import stream_generate

        prompt_ids, cap = self.prepare(req)
        kwargs = self.build_generate_kwargs(req, prompt_ids, cap)
        input_ids = kwargs.pop("input_ids")
        kwargs.pop("max_new_tokens", None)
        kwargs.pop("do_sample", None)               # stream_generate derives it
        temperature = kwargs.pop("temperature", 0.0)
        top_p = kwargs.pop("top_p", 1.0)

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
        thread.start()

        parts, error = [], None
        while True:
            kind, payload = await loop.run_in_executor(None, bridge.get)
            if kind is DONE:
                break
            if kind == "error":
                error = payload
                break
            parts.append(payload)
            yield ("delta", payload)
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
