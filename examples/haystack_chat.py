"""
Haystack example: serve a Haystack chat pipeline as an OpenAI-compatible API.

This example builds a simple Haystack chat pipeline using OpenAIChatGenerator
and exposes it through fastapi-openai-compat with streaming support.

Prerequisites:
    pip install fastapi-openai-compat[haystack] "fastapi[standard]"
    export OPENAI_API_KEY="sk-..."

Run:
    fastapi dev examples/haystack_chat.py
    # or
    python examples/haystack_chat.py

Test:
    # Non-streaming
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is Haystack?"}]}'

    # Streaming
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is Haystack?"}], "stream": true}'

    # With the OpenAI Python client:
    #   pip install openai
    #   python -c "
    #   from openai import OpenAI
    #   client = OpenAI(base_url='http://localhost:8000/v1', api_key='unused')
    #   for chunk in client.chat.completions.create(
    #       model='gpt-4o-mini',
    #       messages=[{'role': 'user', 'content': 'What is Haystack?'}],
    #       stream=True,
    #   ):
    #       print(chunk.choices[0].delta.content or '', end='', flush=True)
    #   print()
    #   "
"""

import uvicorn
from collections.abc import Generator

from fastapi import FastAPI
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk

from fastapi_openai_compat import CompletionResult, create_openai_router

# Available models -- each one will use OpenAI under the hood,
# but you could map different model names to different Haystack pipelines.
MODELS = ["gpt-4o-mini", "gpt-4o"]


def list_models() -> list[str]:
    return MODELS


def run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
    """
    Run a Haystack chat completion.

    When streaming is requested, we pass a callback to OpenAIChatGenerator
    that yields StreamingChunks. The library's default chunk mapper handles
    StreamingChunk objects automatically (via the .content attribute).

    For non-streaming requests, we return the reply as a plain string.
    """
    chat_messages = [ChatMessage.from_dict(m) for m in messages]
    stream = body.get("stream", False)

    if stream:
        return _stream_completion(model, chat_messages, body)

    generator = OpenAIChatGenerator(model=model)
    result = generator.run(messages=chat_messages, generation_kwargs=_generation_kwargs(body))
    return result["replies"][0].text


def _stream_completion(
    model: str,
    messages: list[ChatMessage],
    body: dict,
) -> Generator[StreamingChunk, None, None]:
    """Run a streaming chat completion, yielding Haystack StreamingChunks."""
    chunks: list[StreamingChunk] = []

    def on_chunk(chunk: StreamingChunk) -> None:
        chunks.append(chunk)

    generator = OpenAIChatGenerator(model=model, streaming_callback=on_chunk)
    generator.run(messages=messages, generation_kwargs=_generation_kwargs(body))

    yield from chunks


def _generation_kwargs(body: dict) -> dict:
    """Extract generation parameters from the request body."""
    kwargs = {}
    for key in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"):
        if key in body:
            kwargs[key] = body[key]
    return kwargs


app = FastAPI(title="Haystack OpenAI-Compatible Server")
router = create_openai_router(
    list_models=list_models,
    run_completion=run_completion,
)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
