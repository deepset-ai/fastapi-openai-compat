"""
Basic example: OpenAI-compatible chat completion server.

This example shows how to use fastapi-openai-compat to create a simple
chat completion API that echoes back user messages.

Run:
    pip install fastapi-openai-compat "fastapi[standard]"  # includes uvicorn
    fastapi dev examples/basic.py
    # or
    python examples/basic.py

Test:
    # List models
    curl http://localhost:8000/v1/models | jq

    # Non-streaming completion
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "echo", "messages": [{"role": "user", "content": "Hello!"}]}'

    # Streaming completion
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "echo-stream", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'

    # Metadata in request and response
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "echo-metadata", "messages": [{"role": "user", "content": "Hello!"}], "metadata": {"request_id": "abc-123"}}'

    # Streaming with custom SSE events (e.g. Open WebUI status updates)
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "echo-events", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'

    # Works with the OpenAI Python client too:
    #   pip install openai
    #   python -c "
    #   from openai import OpenAI
    #   client = OpenAI(base_url='http://localhost:8000/v1', api_key='unused')
    #   r = client.chat.completions.create(model='echo', messages=[{'role': 'user', 'content': 'Hi!'}])
    #   print(r.choices[0].message.content)
    #   "
"""

import time
from collections.abc import Generator

import uvicorn
from fastapi import FastAPI

from fastapi_openai_compat import (
    ChatCompletion,
    Choice,
    CompletionResult,
    Message,
    MessageParam,
    create_chat_completion_router,
)


class StatusEvent:
    """Custom SSE event for sending status updates to clients like Open WebUI."""

    def __init__(self, description: str, done: bool = False):
        self.description = description
        self.done = done

    def to_event_dict(self) -> dict:
        return {"type": "status", "data": {"description": self.description, "done": self.done}}


def list_models() -> list[str]:
    """Return the list of available models."""
    return ["echo", "echo-stream", "echo-metadata", "echo-events"]


def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    """
    Run a chat completion.

    - "echo" model: returns the last user message as a plain string.
    - "echo-stream" model: streams the last user message word by word.
    - "echo-metadata" model: reads metadata from the request and includes it in the response.
    - "echo-events" model: streams with custom SSE events interspersed between text chunks.
    """
    last_message = messages[-1]["content"] if messages else "No messages provided."

    if model == "echo-stream":
        return _stream_words(last_message)

    if model == "echo-metadata":
        return _echo_with_metadata(model, last_message, body)

    if model == "echo-events":
        return _stream_with_events(last_message)

    return f"You said: {last_message}"


def _stream_words(text: str) -> Generator[str, None, None]:
    """Yield text word by word to demonstrate streaming."""
    words = text.split()
    for i, word in enumerate(words):
        suffix = "" if i == len(words) - 1 else " "
        yield word + suffix


def _echo_with_metadata(model: str, last_message: str, body: dict) -> ChatCompletion:
    """Return a ChatCompletion with request metadata echoed back in the response."""
    metadata = body.get("metadata", {})
    request_id = metadata.get("request_id", "unknown")

    return ChatCompletion(
        id=f"resp-{request_id}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=f"You said: {last_message}"),
                finish_reason="stop",
            )
        ],
        metadata={"request_id": request_id, "echo": True},
    )


def _stream_with_events(text: str) -> Generator[str | StatusEvent, None, None]:
    """Yield custom status events alongside text chunks."""
    yield StatusEvent("Processing your request...")
    words = text.split()
    for i, word in enumerate(words):
        suffix = "" if i == len(words) - 1 else " "
        yield word + suffix
    yield StatusEvent("Done", done=True)


app = FastAPI(title="Basic OpenAI-Compatible Server")
router = create_chat_completion_router(
    list_models=list_models,
    run_completion=run_completion,
)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
