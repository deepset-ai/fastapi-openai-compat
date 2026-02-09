# fastapi-openai-compat

[![PyPI - Version](https://img.shields.io/pypi/v/fastapi-openai-compat.svg)](https://pypi.org/project/fastapi-openai-compat)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastapi-openai-compat.svg)](https://pypi.org/project/fastapi-openai-compat)
[![Tests](https://github.com/deepset-ai/fastapi-openai-compat/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/fastapi-openai-compat/actions/workflows/tests.yml)

FastAPI router factory for OpenAI-compatible [Chat Completions](https://platform.openai.com/docs/api-reference/chat) endpoints.

Provides a configurable `APIRouter` that exposes `/v1/chat/completions` and `/v1/models` endpoints,
following the [OpenAI API specification](https://platform.openai.com/docs/api-reference/chat),
with support for streaming (SSE), non-streaming responses, tool calling, configurable hooks, and custom chunk mapping.

## Installation

```bash
pip install fastapi-openai-compat
```

With Haystack `StreamingChunk` support:

```bash
pip install fastapi-openai-compat[haystack]
```

## Quick start

Create an OpenAI-compatible Chat Completions server in a few lines. Both sync and async
callables are supported -- sync callables are automatically executed in a thread pool
so they never block the async event loop.

```python
from fastapi import FastAPI
from fastapi_openai_compat import create_openai_router, CompletionResult

def list_models() -> list[str]:
    return ["my-pipeline"]

def run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
    # Your (potentially blocking) pipeline execution logic here
    return "Hello from Haystack!"

app = FastAPI()
router = create_openai_router(
    list_models=list_models,
    run_completion=run_completion,
)
app.include_router(router)
```

Async callables work the same way:

```python
async def list_models() -> list[str]:
    return ["my-pipeline"]

async def run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
    return "Hello from Haystack!"
```

## Tool calling

### Returning ChatCompletion directly

For tool calls and other advanced responses, return a `ChatCompletion` directly
from `run_completion` for full control over the response structure:

```python
import time
from fastapi_openai_compat import ChatCompletion, Choice, Message, CompletionResult

def run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
    return ChatCompletion(
        id="resp-1",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }],
                ),
                finish_reason="tool_calls",
            )
        ],
    )
```

Streaming tool calls work the same way -- yield `ChatCompletion` chunk objects
from your generator and the library serializes them directly as SSE:

```python
def run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
    def stream():
        yield ChatCompletion(
            id="resp-1", object="chat.completion.chunk",
            created=int(time.time()), model=model,
            choices=[Choice(index=0, delta=Message(
                role="assistant",
                tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                             "function": {"name": "get_weather", "arguments": ""}}],
            ))],
        )
        yield ChatCompletion(
            id="resp-1", object="chat.completion.chunk",
            created=int(time.time()), model=model,
            choices=[Choice(index=0, delta=Message(
                role="assistant",
                tool_calls=[{"index": 0, "function": {"arguments": '{"city": "Paris"}'}}],
            ))],
        )
        yield ChatCompletion(
            id="resp-1", object="chat.completion.chunk",
            created=int(time.time()), model=model,
            choices=[Choice(index=0, delta=Message(role="assistant"), finish_reason="tool_calls")],
        )
    return stream()
```

### Automatic StreamingChunk support

When using Haystack's `StreamingChunk` (requires `pip install fastapi-openai-compat[haystack]`),
tool call deltas and finish reasons are handled automatically via duck typing:

```python
from haystack.dataclasses import StreamingChunk
from haystack.dataclasses.streaming_chunk import ToolCallDelta

def run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
    def stream():
        yield StreamingChunk(
            content="",
            tool_calls=[ToolCallDelta(
                index=0, id="call_1",
                tool_name="get_weather", arguments='{"city": "Paris"}',
            )],
            index=0,
        )
        yield StreamingChunk(content="", finish_reason="tool_calls")
    return stream()
```

The library automatically:

- Converts `ToolCallDelta` objects to OpenAI wire format (`tool_calls[].function.name/arguments`)
- Propagates `finish_reason` from chunks (e.g. `"stop"`, `"tool_calls"`, `"length"`)
- Only auto-appends `finish_reason="stop"` if no chunk already carried a finish reason
- Works via duck typing -- any object with `tool_calls` and `finish_reason` attributes is supported

## Hooks

You can inject pre/post hooks to modify requests and results (transformer hooks)
or to observe them without modification (observer hooks). Both sync and async
hooks are supported.

### Transformer hooks

Return a modified value to transform the request or result:

```python
from fastapi_openai_compat import ChatRequest, CompletionResult

async def pre_hook(request: ChatRequest) -> ChatRequest:
    # e.g. inject system prompts, validate, rate-limit
    return request

async def post_hook(result: CompletionResult) -> CompletionResult:
    # e.g. transform, filter
    return result

router = create_openai_router(
    list_models=list_models,
    run_completion=run_completion,
    pre_hook=pre_hook,
    post_hook=post_hook,
)
```

### Observer hooks

Return `None` to observe without modifying (useful for logging, metrics, etc.):

```python
def log_request(request: ChatRequest) -> None:
    print(f"Request for model: {request.model}")

def log_result(result: CompletionResult) -> None:
    print(f"Got result type: {type(result).__name__}")

router = create_openai_router(
    list_models=list_models,
    run_completion=run_completion,
    pre_hook=log_request,
    post_hook=log_result,
)
```

## Custom chunk mapping

By default the router handles plain `str` chunks and objects with a `.content`
attribute (e.g. Haystack `StreamingChunk`). If your pipeline streams a different
type, provide a `chunk_mapper` to extract text content:

```python
from dataclasses import dataclass

@dataclass
class MyChunk:
    text: str
    score: float

def my_mapper(chunk: MyChunk) -> str:
    return chunk.text

router = create_openai_router(
    list_models=list_models,
    run_completion=run_completion,
    chunk_mapper=my_mapper,
)
```

This works with any object -- dataclasses, dicts, Pydantic models, etc.:

```python
def dict_mapper(chunk: dict) -> str:
    return chunk["payload"]
```

## Examples

The [`examples/`](examples/) folder contains ready-to-run servers:

- **[`basic.py`](examples/basic.py)** -- Minimal echo server, no external API keys required.
- **[`haystack_chat.py`](examples/haystack_chat.py)** -- Haystack `OpenAIChatGenerator` with streaming support.

See the [examples README](examples/README.md) for setup and usage instructions.

## Reference

This library implements endpoints compatible with the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).
