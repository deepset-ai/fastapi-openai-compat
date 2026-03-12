"""
Basic example: OpenAI-compatible Responses API server.

This example shows how to use fastapi-openai-compat to expose /v1/responses
with non-streaming text, streaming text, and function call output.

Run the server:
    pip install fastapi-openai-compat "fastapi[standard]"  # includes uvicorn
    python examples/responses_basic.py

Run the e2e client (in a second terminal):
    pip install openai
    python examples/responses_basic_client.py

Or test with curl:
    curl http://localhost:8000/v1/models | jq

    curl http://localhost:8000/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "responses-echo", "input": "Hello!"}' | jq

    curl http://localhost:8000/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "responses-stream", "input": "Hello from streaming", "stream": true}'

    curl http://localhost:8000/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "responses-function", "input": "Weather in Paris?", "stream": true}'

    curl http://localhost:8000/v1/responses \
      -H "Content-Type: application/json" \
      -d '{"model": "responses-function", "input": "Weather in Paris?"}' | jq
"""

import json
import time
import uuid
from collections.abc import Generator

import uvicorn
from fastapi import FastAPI

from fastapi_openai_compat import InputItem, Response, ResponseResult, create_responses_router

MODELS = ["responses-echo", "responses-stream", "responses-function"]


class FunctionCallChunk:
    """Minimal chunk shape recognized by responses_streaming duck typing."""

    def __init__(self, *, call_id: str, name: str | None, arguments: str | None):
        self.function_call_id = call_id
        self.function_call_name = name
        self.function_call_arguments = arguments


def list_models() -> list[str]:
    return MODELS


def run_response(model: str, input_items: list[InputItem], body: dict) -> ResponseResult:
    user_text = _extract_user_text(input_items)
    stream = bool(body.get("stream", False))

    if model == "responses-stream" and stream:
        return _stream_words(user_text or "No input provided.")

    if model == "responses-function":
        if stream:
            return _stream_function_call(user_text)
        return _function_call_response(model, user_text)

    return f"You said: {user_text or 'No input provided.'}"


def _extract_user_text(input_items: list[InputItem]) -> str:
    """Extract user text from normalized input items."""
    collected: list[str] = []
    for item in input_items:
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            collected.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "input_text":
                    collected.append(part.get("text", ""))
    return " ".join(collected).strip()


def _stream_words(text: str) -> Generator[str, None, None]:
    words = text.split()
    if not words:
        yield text
        return
    for idx, word in enumerate(words):
        suffix = "" if idx == len(words) - 1 else " "
        yield word + suffix


def _stream_function_call(user_text: str) -> Generator[FunctionCallChunk, None, None]:
    location = user_text or "Paris"
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    prefix = '{"location": "'
    suffix = location + '"}'
    yield FunctionCallChunk(call_id=call_id, name="get_weather", arguments=prefix)
    yield FunctionCallChunk(call_id=call_id, name=None, arguments=suffix)


def _function_call_response(model: str, user_text: str) -> Response:
    location = user_text or "Paris"
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    arguments = json.dumps({"location": location})
    return Response(
        id=f"resp_{uuid.uuid4().hex}",
        created_at=int(time.time()),
        model=model,
        output=[
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": "get_weather",
                "arguments": arguments,
            }
        ],
    )


app = FastAPI(title="Responses API Example")
app.include_router(
    create_responses_router(
        list_models=list_models,
        run_response=run_response,
        include_models_endpoints=True,
    )
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
