"""
Basic example: OpenAI-compatible chat completion server.

This example shows how to use fastapi-openai-compat to create a simple
chat completion API that echoes back user messages.

Run:
    pip install fastapi-openai-compat "fastapi[standard]"
    fastapi dev examples/basic.py
    # or
    python examples/basic.py

Test:
    # List models
    curl http://localhost:8000/v1/models | python -m json.tool

    # Non-streaming completion
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "echo", "messages": [{"role": "user", "content": "Hello!"}]}'

    # Streaming completion
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "echo", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'

    # Works with the OpenAI Python client too:
    #   pip install openai
    #   python -c "
    #   from openai import OpenAI
    #   client = OpenAI(base_url='http://localhost:8000/v1', api_key='unused')
    #   r = client.chat.completions.create(model='echo', messages=[{'role': 'user', 'content': 'Hi!'}])
    #   print(r.choices[0].message.content)
    #   "
"""

import uvicorn
from collections.abc import Generator

from fastapi import FastAPI

from fastapi_openai_compat import CompletionResult, create_openai_router


def list_models() -> list[str]:
    """Return the list of available models."""
    return ["echo", "echo-stream"]


def run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
    """
    Run a chat completion.

    - "echo" model: returns the last user message as a plain string.
    - "echo-stream" model: streams the last user message word by word.
    """
    last_message = messages[-1]["content"] if messages else "No messages provided."

    if model == "echo-stream":
        return _stream_words(last_message)

    return f"You said: {last_message}"


def _stream_words(text: str) -> Generator[str, None, None]:
    """Yield text word by word to demonstrate streaming."""
    words = text.split()
    for i, word in enumerate(words):
        suffix = "" if i == len(words) - 1 else " "
        yield word + suffix


app = FastAPI(title="Basic OpenAI-Compatible Server")
router = create_openai_router(
    list_models=list_models,
    run_completion=run_completion,
)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
