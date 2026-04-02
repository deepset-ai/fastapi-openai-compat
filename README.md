# fastapi-openai-compat

[![PyPI - Version](https://img.shields.io/pypi/v/fastapi-openai-compat.svg)](https://pypi.org/project/fastapi-openai-compat)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastapi-openai-compat.svg)](https://pypi.org/project/fastapi-openai-compat)
[![Tests](https://github.com/deepset-ai/fastapi-openai-compat/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/fastapi-openai-compat/actions/workflows/tests.yml)

FastAPI router factory for OpenAI-compatible Chat Completions, Responses, and Files upload endpoints.

Provides configurable router factories for OpenAI-style APIs, with support for
streaming (SSE), non-streaming responses, tool calling, reasoning content,
configurable hooks, custom chunk mapping, and callback-driven file upload handling.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [A note on dict-based types](#a-note-on-dict-based-types)
- **Chat Completions API**
  - [The `run_completion` callable](#the-run_completion-callable)
  - [Response types](#response-types) -- [string](#returning-a-string) &#183; [generator](#streaming-with-a-generator) &#183; [ChatCompletion](#returning-a-chatcompletion)
  - [Tool calling](#tool-calling) -- [ChatCompletion](#returning-chatcompletion-directly) &#183; [StreamingChunk](#automatic-streamingchunk-support)
  - [Reasoning content](#reasoning-content)
  - [Custom SSE events](#custom-sse-events)
  - [Hooks](#hooks) -- [transformer](#transformer-hooks) &#183; [observer](#observer-hooks)
  - [Custom chunk mapping](#custom-chunk-mapping)
- **Responses API**
  - [Quick start](#quick-start-1)
  - [The `run_response` callable](#the-run_response-callable)
  - [Streaming text](#streaming-text)
  - [Streaming function calls](#streaming-function-calls)
  - [Streaming reasoning](#streaming-reasoning)
  - [Returning a Response object](#returning-a-response-object)
  - [Combining with chat completions](#combining-with-chat-completions)
  - [Hooks](#hooks-1)
- [Examples](#examples)
- [API reference](#api-reference) -- [`create_chat_completion_router`](#create_chat_completion_router) &#183; [`create_models_router`](#create_models_router) &#183; [`create_responses_router`](#create_responses_router) &#183; [`create_files_router`](#create_files_router-minimal-files-upload-support)

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
from fastapi_openai_compat import CompletionResult, MessageParam, create_chat_completion_router

def list_models() -> list[str]:
    return ["my-pipeline"]

def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    # Your (potentially blocking) pipeline execution logic here
    return "Hello from Haystack!"

app = FastAPI()
router = create_chat_completion_router(
    list_models=list_models,
    run_completion=run_completion,
)
app.include_router(router)
```

Async callables work the same way:

```python
async def list_models() -> list[str]:
    return ["my-pipeline"]

async def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    return "Hello from Haystack!"
```

## A note on dict-based types

The type aliases `MessageParam`, `InputItem`, and `OutputItem` are all
`dict[str, Any]` -- **not** Pydantic models.  This is a deliberate design
choice:

- **Forward-compatibility.** OpenAI regularly adds new message types, content
  part types, and input/output item types.  A strict union of Pydantic models
  would reject unknown types and require a library release for every API
  change.  Plain dicts let your callbacks handle new types immediately.
- **Pass-through design.** This library validates the request *envelope*
  (correct top-level structure) and passes the inner items through to your
  callback unchanged.  Domain-specific validation belongs in your callback or
  a pre-hook, not in the transport layer.
- **Consistency.** Both Chat Completions (`messages`) and Responses API
  (`input_items`, `output`) follow the same pattern, so you have a single
  mental model for both.

The aliases exist to give you IDE hints and self-documenting signatures.
Each alias's docstring lists the common dict shapes you'll encounter -- check
them in your IDE or in the
[source](src/fastapi_openai_compat/chat_completions/models.py).

## The `run_completion` callable

The `run_completion` callable receives three arguments:

| Argument   | Type                  | Description |
|------------|-----------------------|-------------|
| `model`    | `str`                 | The model name from the request (e.g. `"my-pipeline"`). |
| `messages` | `list[MessageParam]`  | The conversation history in OpenAI format (see [A note on dict-based types](#a-note-on-dict-based-types)). |
| `body`     | `dict`                | The full request body, including all extra parameters (e.g. `temperature`, `max_tokens`, `stream`, `metadata`, `tools`). |

The request model accepts any additional fields beyond `model`, `messages`, and `stream`.
These extra parameters are forwarded as-is in the `body` dict, so you can use them
however you need without any library changes.

For example, you can access `metadata` and any other extra field from `body`:

```python
import time
from fastapi_openai_compat import ChatCompletion, Choice, Message, MessageParam, CompletionResult

def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    metadata = body.get("metadata", {})
    temperature = body.get("temperature", 1.0)
    request_id = metadata.get("request_id", "unknown")

    return ChatCompletion(
        id=f"resp-{request_id}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
        metadata={"request_id": request_id, "temperature_used": temperature},
    )
```

A client can then send:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-pipeline",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "metadata": {"request_id": "abc-123", "user_tier": "premium"}
  }'
```

The `metadata` field in the response works because `ChatCompletion` also allows extra fields,
so you can attach any additional data to the response object.

The return type determines how the response is formatted:

| Return type        | Behavior |
|--------------------|----------|
| `str`              | Wrapped automatically into a `ChatCompletion` response. |
| `Generator`        | Each yielded chunk is converted to a `chat.completion.chunk` SSE message. |
| `AsyncGenerator`   | Same as `Generator`, but async. |
| `ChatCompletion`   | Returned as-is for full control over the response. |

## Response types

### Returning a string

The simplest option -- return a plain string and the library wraps it as a
complete `ChatCompletion` response automatically:

```python
def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    last_msg = messages[-1]["content"]
    return f"You said: {last_msg}"
```

### Streaming with a generator

Return a generator to stream responses token by token via SSE.
Each yielded string is automatically wrapped into a `chat.completion.chunk` message --
you only need to yield the text content, the library handles the SSE wire format.
A `finish_reason="stop"` sentinel is appended automatically at the end of the stream.

Your `run_completion` should check `body.get("stream", False)` to decide whether
to return a generator or a plain string:

```python
from collections.abc import Generator

def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    last_msg = messages[-1]["content"]

    if body.get("stream", False):
        def stream() -> Generator[str, None, None]:
            for word in last_msg.split():
                yield word + " "
        return stream()

    return f"You said: {last_msg}"
```

Async generators work the same way:

```python
from collections.abc import AsyncGenerator

async def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    async def stream() -> AsyncGenerator[str, None]:
        for word in ["Hello", " from", " Haystack", "!"]:
            yield word
    return stream()
```

### Returning a ChatCompletion

For full control over the response (e.g. custom `usage`, `finish_reason`, or `system_fingerprint`),
return a `ChatCompletion` object directly:

```python
import time
from fastapi_openai_compat import ChatCompletion, Choice, Message, CompletionResult

def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    return ChatCompletion(
        id="resp-1",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
```

## Tool calling

### Returning ChatCompletion directly

For tool calls and other advanced responses, return a `ChatCompletion` directly
from `run_completion` for full control over the response structure:

```python
import time
from fastapi_openai_compat import ChatCompletion, Choice, Message, CompletionResult

def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
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
def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
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

def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
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

## Reasoning content

When streaming, chunks with a `reasoning` attribute are emitted as
`reasoning_content` on the message delta (the
[DeepSeek convention](https://api-docs.deepseek.com/guides/reasoning_model),
widely adopted by OpenAI-compatible clients). This is detected via duck
typing -- any object with a `.reasoning` attribute works.

With Haystack's `StreamingChunk` and `ReasoningContent`:

```python
from haystack.dataclasses import StreamingChunk
from haystack.dataclasses.streaming_chunk import ReasoningContent

def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    def stream():
        yield StreamingChunk(content="", reasoning=ReasoningContent(reasoning_text="Let me think..."))
        yield StreamingChunk(content="", reasoning=ReasoningContent(reasoning_text=" The answer is 42."))
        yield StreamingChunk(content="The answer is 42.")
    return stream()
```

The resulting SSE stream contains chunks with `reasoning_content` on the delta
for reasoning tokens, followed by regular `content` deltas for the visible
answer:

```json
{"choices": [{"delta": {"reasoning_content": "Let me think...", "content": null}}]}
{"choices": [{"delta": {"reasoning_content": " The answer is 42.", "content": null}}]}
{"choices": [{"delta": {"content": "The answer is 42."}}]}
```

Any custom object works too -- the library checks for `chunk.reasoning` and
extracts text via `.reasoning_text` (falling back to `str()`):

```python
class ReasoningChunk:
    def __init__(self, reasoning_text: str):
        self.reasoning = type("R", (), {"reasoning_text": reasoning_text})()

def stream():
    yield ReasoningChunk("Thinking step 1...")
    yield "Final answer"
```

## Custom SSE events

You can yield custom SSE events alongside regular chat completion chunks. This is useful
for sending side-channel data to clients like [Open WebUI](https://openwebui.com) --
status updates, notifications, source citations, etc.

Any object with a `.to_event_dict()` method is recognized as a custom event and serialized
as `data: {"event": {...}}` in the SSE stream. Custom events don't interfere with
chat completion chunks or the `finish_reason` tracking.

```python
from collections.abc import Generator
from fastapi_openai_compat import CompletionResult

class StatusEvent:
    def __init__(self, description: str, done: bool = False):
        self.description = description
        self.done = done

    def to_event_dict(self) -> dict:
        return {"type": "status", "data": {"description": self.description, "done": self.done}}

def run_completion(model: str, messages: list[MessageParam], body: dict) -> CompletionResult:
    def stream() -> Generator[str | StatusEvent, None, None]:
        yield StatusEvent("Processing your request...")
        for word in ["Hello", " from", " Haystack", "!"]:
            yield word
        yield StatusEvent("Done", done=True)
    return stream()
```

This works via duck typing -- any object implementing `to_event_dict() -> dict` is supported.
The protocol is compatible with [Hayhooks' Open WebUI events](https://deepset-ai.github.io/hayhooks/).

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

router = create_chat_completion_router(
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

router = create_chat_completion_router(
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

router = create_chat_completion_router(
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

## Responses API

The Responses API uses named SSE events (matching the
[OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses))
instead of the `data:`-only format used by chat completions.

### Quick start

```python
from fastapi import FastAPI
from fastapi_openai_compat import InputItem, ResponseResult, create_responses_router

def list_models() -> list[str]:
    return ["my-pipeline"]

def run_response(model: str, input_items: list[InputItem], body: dict) -> ResponseResult:
    return "Hello from the Responses API!"

app = FastAPI()
app.include_router(
    create_responses_router(
        list_models=list_models,
        run_response=run_response,
        include_models_endpoints=True,
    )
)
```

### The `run_response` callable

The `run_response` callable receives three arguments:

| Argument       | Type               | Description |
|----------------|--------------------|-------------|
| `model`        | `str`              | The model name from the request. |
| `input_items`  | `list[InputItem]`  | Normalized input items (see [A note on dict-based types](#a-note-on-dict-based-types)). String shorthand is converted to a message item; `None` becomes `[]`. |
| `body`         | `dict`             | The full request body, including all extra parameters (e.g. `temperature`, `tools`, `instructions`). |

The return type determines how the response is formatted:

| Return type      | Behavior |
|------------------|----------|
| `str`            | Wrapped into a `Response` with a single text output message. |
| `Generator`      | Each yielded chunk is emitted as named SSE events (`response.output_text.delta`, etc.). |
| `AsyncGenerator` | Same as `Generator`, but async. |
| `Response`       | Returned as-is for full control over the response. |

### Streaming text

Return a generator to stream text via named SSE events. Each yielded string
becomes a `response.output_text.delta` event. The library handles all the
surrounding lifecycle events (`response.created`, `response.in_progress`,
`response.output_item.added`, `response.completed`, etc.) automatically.

```python
from collections.abc import Generator

def run_response(model: str, input_items: list[InputItem], body: dict) -> ResponseResult:
    if body.get("stream", False):
        def stream() -> Generator[str, None, None]:
            for word in ["Hello", " from", " streaming", "!"]:
                yield word
        return stream()
    return "Hello!"
```

### Streaming function calls

Yield objects with `function_call_id`, `function_call_name`, and
`function_call_arguments` attributes to stream function call events.
The library emits `response.function_call_arguments.delta` events during
streaming and a `response.output_item.done` event when the call completes.

```python
class FunctionCallChunk:
    def __init__(self, *, call_id: str, name: str | None, arguments: str | None):
        self.function_call_id = call_id
        self.function_call_name = name
        self.function_call_arguments = arguments

def run_response(model: str, input_items: list[InputItem], body: dict) -> ResponseResult:
    def stream():
        yield FunctionCallChunk(call_id="call_1", name="get_weather", arguments='{"city":')
        yield FunctionCallChunk(call_id="call_1", name=None, arguments=' "Paris"}')
    return stream()
```

### Streaming reasoning

Yield chunks with a `reasoning` attribute to stream reasoning output items.
The library emits the proper OpenAI
[reasoning summary events](https://platform.openai.com/docs/api-reference/responses-streaming)
(`response.reasoning_summary_part.added`, `response.reasoning_summary_text.delta`,
`response.reasoning_summary_text.done`, `response.reasoning_summary_part.done`)
and produces a `type: "reasoning"` output item with a `summary` array in the
completed response.

```python
from haystack.dataclasses import StreamingChunk
from haystack.dataclasses.streaming_chunk import ReasoningContent

def run_response(model: str, input_items: list[InputItem], body: dict) -> ResponseResult:
    def stream():
        yield StreamingChunk(content="", reasoning=ReasoningContent(reasoning_text="Step 1: "))
        yield StreamingChunk(content="", reasoning=ReasoningContent(reasoning_text="analyze the input."))
        yield StreamingChunk(content="Here is the answer.")
    return stream()
```

The completed `Response` object includes the reasoning item before the message:

```json
{
  "output": [
    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Step 1: analyze the input."}]},
    {"type": "message", "content": [{"type": "output_text", "text": "Here is the answer."}]}
  ]
}
```

### Returning a Response object

For full control, return a `Response` directly:

```python
import time
import uuid
from fastapi_openai_compat import Response

def run_response(model: str, input_items: list[InputItem], body: dict) -> ResponseResult:
    return Response(
        id=f"resp_{uuid.uuid4().hex}",
        created_at=int(time.time()),
        model=model,
        output=[{
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
        }],
    )
```

### Combining with chat completions

When mounting both routers in the same app, use a dedicated `create_models_router`
to avoid duplicate `/v1/models` endpoints:

```python
from fastapi import FastAPI
from fastapi_openai_compat import (
    create_chat_completion_router,
    create_models_router,
    create_responses_router,
)

app = FastAPI()
app.include_router(create_models_router(list_models=list_models))
app.include_router(
    create_chat_completion_router(
        list_models=list_models,
        run_completion=run_completion,
        include_models_endpoints=False,
    )
)
app.include_router(
    create_responses_router(
        list_models=list_models,
        run_response=run_response,
        include_models_endpoints=False,
    )
)
```

### Hooks

Pre/post hooks work the same way as chat completions. The pre-hook receives
a `ResponseRequest` and the post-hook receives a `ResponseResult`:

```python
from fastapi_openai_compat import ResponseRequest

async def pre_hook(request: ResponseRequest) -> ResponseRequest:
    # e.g. inject instructions, validate, rate-limit
    return request

router = create_responses_router(
    list_models=list_models,
    run_response=run_response,
    pre_hook=pre_hook,
)
```

## Examples

The [`examples/`](examples/) folder contains ready-to-run servers:

- **[`basic.py`](examples/basic.py)** -- Minimal echo server, no external API keys required.
- **[`haystack_chat.py`](examples/haystack_chat.py)** -- Haystack `OpenAIChatGenerator` with streaming support.
- **[`responses_basic.py`](examples/responses_basic.py)** -- Responses API text + streaming + function call demo.
- **[`responses_with_files.py`](examples/responses_with_files.py)** -- Responses API with `/v1/files` upload + `input_file.file_id`.

See the [examples README](examples/README.md) for setup and usage instructions.

## API reference

This library implements endpoints compatible with the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat), the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses), and a minimal Files upload router.

### `create_chat_completion_router`

`create_openai_router` is still available as a backward-compatible alias.

```python
create_chat_completion_router(
    *,
    list_models,
    run_completion,
    pre_hook=None,
    post_hook=None,
    chunk_mapper=default_chunk_mapper,
    owned_by="custom",
    tags=None,
    include_models_endpoints=True,
) -> APIRouter
```

| Parameter        | Type                      | Description |
|------------------|---------------------------|-------------|
| `list_models`    | `Callable -> list[str]`   | Returns available model/pipeline names. |
| `run_completion` | `Callable -> CompletionResult` | Runs a chat completion given `(model, messages: list[MessageParam], body)`. |
| `pre_hook`       | `Callable` or `None`      | Called before `run_completion`. Receives `ChatRequest`, returns modified request (transformer) or `None` (observer). |
| `post_hook`      | `Callable` or `None`      | Called after `run_completion`. Receives `CompletionResult`, returns modified result (transformer) or `None` (observer). |
| `chunk_mapper`   | `Callable[[Any], str]`    | Converts streamed chunks to strings. Default handles `str` and `.content` attribute. |
| `owned_by`       | `str`                     | Value for the `owned_by` field in model objects. Defaults to `"custom"`. |
| `tags`           | `list[str]` or `None`     | OpenAPI tags for the generated endpoints. Defaults to `["openai"]`. |
| `include_models_endpoints` | `bool`          | If true, includes `/v1/models` and `/models` in this router. Defaults to `True`. |

### Endpoints

With `include_models_endpoints=True` (default), the router exposes:

| Method | Path                        | Description |
|--------|-----------------------------|-------------|
| `GET`  | `/v1/models`                | List available models. |
| `POST` | `/v1/chat/completions`      | Create a chat completion (streaming or non-streaming). |
| `GET`  | `/models`                   | Alias for `/v1/models`. |
| `POST` | `/chat/completions`         | Alias for `/v1/chat/completions`. |

### `create_models_router`

```python
create_models_router(
    *,
    list_models,
    owned_by="custom",
    tags=None,
    operation_id_prefix="openai",
) -> APIRouter
```

Use this when composing multiple routers and you want a single owner for
`/v1/models`. See [Combining with chat completions](#combining-with-chat-completions)
for a full example.

### `create_responses_router`

```python
create_responses_router(
    *,
    list_models,
    run_response,
    pre_hook=None,
    post_hook=None,
    chunk_mapper=default_chunk_mapper,
    owned_by="custom",
    tags=None,
    include_models_endpoints=False,
) -> APIRouter
```

| Parameter                 | Type                      | Description |
|---------------------------|---------------------------|-------------|
| `list_models`             | `Callable -> list[str]`   | Returns available model/pipeline names. |
| `run_response`            | `Callable -> ResponseResult` | Runs a Responses request given `(model, input_items: list[InputItem], body)`. |
| `pre_hook`                | `Callable` or `None`      | Called before `run_response`. Receives `ResponseRequest`, returns modified request (transformer) or `None` (observer). |
| `post_hook`               | `Callable` or `None`      | Called after `run_response`. Receives `ResponseResult`, returns modified result (transformer) or `None` (observer). |
| `chunk_mapper`            | `Callable[[Any], str]`    | Converts streamed non-string chunks to strings. |
| `owned_by`                | `str`                     | Value for `owned_by` in model objects when models endpoints are enabled. Defaults to `"custom"`. |
| `tags`                    | `list[str]` or `None`     | OpenAPI tags for generated endpoints. Defaults to `["openai"]`. |
| `include_models_endpoints`| `bool`                    | If true, includes `/v1/models` and `/models` in this router. Defaults to `False` to avoid conflicts when combined with chat or a dedicated models router. |

Responses router endpoints:

| Method | Path            | Description |
|--------|-----------------|-------------|
| `POST` | `/v1/responses` | Create a Responses API response (streaming or non-streaming). |
| `POST` | `/responses`    | Alias for `/v1/responses`. |

### `create_files_router` (minimal Files upload support)

```python
create_files_router(
    *,
    run_file_upload,
    tags=None,
) -> APIRouter
```

The `run_file_upload` callback receives:

- `filename`: uploaded filename, if present
- `content_type`: uploaded content type, if present
- `content`: full uploaded file bytes
- `purpose`: multipart form `purpose` field

It can return either:

- `FileObject`
- `dict` matching the `FileObject` schema

The router exposes:

| Method | Path         | Description |
|--------|--------------|-------------|
| `POST` | `/v1/files`  | Upload a file (`files.create(...)` compatible). |
| `POST` | `/files`     | Alias for `/v1/files`. |
