from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import gc
import os
import time
import torch
from typing import Optional, List, Dict, Any

# ExLlamaV3 imports
from exllamav3 import (
    ExLlamaV3Config, ExLlamaV3Model, ExLlamaV3Cache,
    ExLlamaV3Tokenizer, ExLlamaV3Generator, ExLlamaV3Sampler
)

# ========================= CONFIG =========================
MODEL_DIR = "models/exl3"
MODELS = {
    "node":   os.path.join(MODEL_DIR, "node"),
    "python": os.path.join(MODEL_DIR, "python"),
    "rust":   os.path.join(MODEL_DIR, "rust"),
}

MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 16384))   # ← you can safely increase this

# ======================= GLOBAL STATE =======================
current_specialization: Optional[str] = None
model:     Optional[ExLlamaV3Model]     = None
cache:     Optional[ExLlamaV3Cache]     = None
tokenizer: Optional[ExLlamaV3Tokenizer] = None
generator: Optional[ExLlamaV3Generator] = None

model_lock = asyncio.Lock()
app = FastAPI(title="Qwen3.5-9B ExLlamaV3 Code Specialist")


# ======================= MODEL LIFECYCLE =======================
def _unload_model():
    global model, cache, tokenizer, generator, current_specialization
    if model is None:
        return
    print(f"🗑️  Unloading {current_specialization.upper()} …")
    del generator, cache, tokenizer, model
    generator = cache = tokenizer = model = None
    current_specialization = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("✅ GPU VRAM fully cleared")


def _load_model(specialization: str):
    global model, cache, tokenizer, generator, current_specialization
    if specialization not in MODELS:
        raise ValueError(f"Unknown specialization: {specialization}")

    if current_specialization == specialization and model is not None:
        return

    _unload_model()

    t0 = time.time()
    config = ExLlamaV3Config(MODELS[specialization])
    config.max_seq_len = MAX_SEQ_LEN

    model = ExLlamaV3Model(config)
    tokenizer = ExLlamaV3Tokenizer(config)
    cache = ExLlamaV3Cache(model, max_seq_len=MAX_SEQ_LEN)
    generator = ExLlamaV3Generator(model, cache, tokenizer)

    current_specialization = specialization
    print(f"✅ {specialization.upper()} loaded in {time.time()-t0:.1f}s")


# ======================= CHAT TEMPLATE =======================
def apply_qwen3_chat_template(messages: List[Dict[str, str]]) -> str:
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


# ======================= REQUEST MODEL =======================
class ChatRequest(BaseModel):
    specialization: str
    messages: List[Dict[str, str]]
    max_tokens: int = Field(2048, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stream: bool = False


# ======================= ENDPOINTS =======================
@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.specialization not in MODELS:
        raise HTTPException(400, f"specialization must be one of {list(MODELS.keys())}")

    async with model_lock:
        await asyncio.get_event_loop().run_in_executor(None, _load_model, req.specialization)

        prompt = apply_qwen3_chat_template(req.messages)

        settings = ExLlamaV3Sampler.Settings()
        settings.temperature = req.temperature
        settings.top_p = req.top_p
        settings.repetition_penalty = req.repetition_penalty

        if req.stream:
            async def token_stream():
                input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
                generator.begin_stream(input_ids, settings)
                generated = 0
                while True:
                    chunk, eos = await asyncio.get_event_loop().run_in_executor(None, generator.stream)
                    if chunk:
                        yield f'data: {{"choices":[{{"delta":{{"content":{chunk!r}}}}}]}}\n\n'
                    generated += 1
                    if eos or generated >= req.max_tokens:
                        break
                yield "data: [DONE]\n\n"

            return StreamingResponse(token_stream(), media_type="text/event-stream")

        # Non-streaming
        output = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: tokenizer.decode(
                generator.generate(
                    tokenizer.encode(prompt, encode_special_tokens=True),
                    gen_settings=settings,
                    max_new_tokens=req.max_tokens,
                    add_bos=False,
                    encode_special_tokens=True,
                    decode_special_tokens=False
                )[:, -1:][0]   # only new tokens
            )[0]
        )

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.specialization,
        "choices": [{"message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
    }


@app.get("/health")
def health():
    return {
        "status": "ready",
        "current_model": current_specialization,
        "backend": "ExLlamaV3",
        "gpu": "RTX 5060 Ti 16GB",
        "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "max_seq_len": MAX_SEQ_LEN,
    }


@app.post("/unload")
async def unload():
    async with model_lock:
        await asyncio.get_event_loop().run_in_executor(None, _unload_model)
    return {"status": "unloaded"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Qwen3.5-9B ExLlamaV3 Multi-Specialist Server starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)