"""
FastAPI Production Inference Server.
Exposes standard OpenAI-compatible endpoints:
- GET  /health
- GET  /v1/models
- POST /v1/chat/completions (supports Server-Sent Events SSE streaming)
- POST /v1/completions
- POST /generate
"""

import json
import time
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from inference.generator import CodeGenerator

app = FastAPI(
    title="Universal Coding AI Model API Server",
    description="High-performance programming foundation model API for code generation, debugging, and agentic workflows.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance
generator: Optional[CodeGenerator] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "universal-coding-model-v1"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str = "universal-coding-model-v1"
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    suffix: Optional[str] = None  # FIM support


def get_generator() -> CodeGenerator:
    global generator
    if generator is None:
        generator = CodeGenerator()
    return generator


@app.get("/health")
def health():
    gen = get_generator()
    return {
        "status": "healthy",
        "service": "Universal Coding AI Model",
        "checkpoint": gen.checkpoint_dir,
        "device": gen.device,
        "vocab_size": gen.config.vocab_size,
        "hidden_size": gen.config.hidden_size,
        "layers": gen.config.num_hidden_layers,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "universal-coding-model-v1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "universal-coding-ai",
                "capabilities": ["code_generation", "fim", "chat", "debugging", "agentic_tools"],
            }
        ],
    }


@app.post("/generate")
def direct_generate(req: GenerateRequest):
    gen = get_generator()
    if req.stream:
        def event_stream():
            for chunk in gen.stream_generate(
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = gen.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    return {"generated_text": text}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    gen = get_generator()
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        # Formulate ChatML prompt
        input_ids, _ = gen.tokenizer.encode_chat(msgs, add_generation_prompt=True)
        prompt_text = gen.tokenizer.decode(input_ids, skip_special_tokens=False)

        def chat_stream():
            for chunk in gen.stream_generate(
                prompt=prompt_text,
                max_new_tokens=req.max_tokens or 256,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.9,
            ):
                payload = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"delta": {"content": chunk}, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(chat_stream(), media_type="text/event-stream")

    answer = gen.chat(
        messages=msgs,
        max_new_tokens=req.max_tokens or 256,
        temperature=req.temperature or 0.7,
    )

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "message": {"role": "assistant", "content": answer},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    gen = get_generator()
    if req.suffix:
        # FIM mode
        result = gen.complete_fim(prefix=req.prompt, suffix=req.suffix, max_new_tokens=req.max_tokens or 128)
    else:
        result = gen.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens or 256,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
        )

    return {
        "id": f"cmpl-{int(time.time()*1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": result, "index": 0, "finish_reason": "stop"}],
    }
