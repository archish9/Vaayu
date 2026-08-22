"""OpenAI-compatible request/response models, plus Vaayu's own optional extras.

Pure pydantic: no torch, no model, no I/O.

`extra="ignore"` throughout is deliberate. Off-the-shelf agent clients send
fields this server does not implement (logit_bias, presence_penalty, user, ...);
rejecting those with a 422 would break every one of them. Fields Vaayu DOES
honour are declared explicitly, so an ignored field is genuinely unimplemented
rather than merely unnamed.
"""
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def make_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


# ── request ──────────────────────────────────────────────────────────────────

class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    arguments: str          # JSON-encoded string, per the OpenAI wire format


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class FunctionDef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["function"] = "function"
    function: FunctionDef


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)

    # None means "omitted" -- resolved later against the model's DERIVED context
    # window by stopping.resolve_max_tokens. A numeric default here would be
    # exactly the literal cap this server exists to remove.
    max_tokens: int | None = Field(default=None, gt=0)

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stop: list[str] = Field(default_factory=list)
    n: int = 1
    seed: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None

    tools: list[ToolDef] | None = None
    tool_choice: Any = None

    # OpenAI's own end-user tag. Declared, not ignored, because it is the only
    # per-stream id an off-the-shelf client sends -- see `session_key`.
    user: str | None = None

    # ── Vaayu extras. Optional; OpenAI clients never send them. ──
    session_id: str | None = None
    kv_cache_mode: str | None = None
    drafter_model_path: str | None = None

    @property
    def session_key(self) -> str | None:
        """The prefix-cache session for this turn, or None to auto-match.

        `session_id` wins: it names one conversation. `user` is OpenAI's stable
        END-USER id, so two concurrent conversations from one user share a pool
        entry and evict each other -- a real cost, but a smaller one than the
        silent drop it replaces, which cost exact matching with no way to tell.
        """
        return self.session_id or self.user or None

    @field_validator("stop", mode="before")
    @classmethod
    def _normalise_stop(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("n")
    @classmethod
    def _only_one_choice(cls, v):
        if v != 1:
            raise ValueError(
                "this server generates one choice per request (n=1). Concurrency "
                "is 1 on this hardware -- N>=2 at long context is a settled "
                "refusal, not a missing feature.")
        return v


# ── response ─────────────────────────────────────────────────────────────────

class ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: int = 0
    message: ResponseMessage
    finish_reason: str


class Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None


class Delta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChunkChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None
