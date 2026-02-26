# Responses API Support -- Design Document

**Date:** 2026-02-26
**Status:** Approved

## Goal

Add OpenAI Responses API (`POST /v1/responses`) support to fastapi-openai-compat, using the same callback-driven pattern as the existing Chat Completions implementation.

## Scope

### Phase 1 (this design)

- Text input/output (string shorthand and input item lists)
- Function calling (tool definitions + function_call output items)
- Streaming with named SSE events matching the Responses API spec
- Pre/post hooks (transformer or observer pattern)
- `instructions` field support

### Phase 2 (future)

- File and image input forwarding to callbacks
- Structured outputs (`text.format` with JSON schema)
- Reasoning items support

### Out of scope

Built-in hosted tools (web_search, file_search, code_interpreter, image_generation, computer_use), stateful conversation management (`previous_response_id`, `store`, `conversation`), background mode, MCP server tools. These are OpenAI-hosted features that don't map to a "bring your own backend" library.

## Architecture

Parallel module approach -- new files alongside existing Chat Completions code with no changes to the stable chat API.

```
src/fastapi_openai_compat/
  router.py                  # existing (untouched)
  models.py                  # existing (untouched)
  streaming.py               # existing (untouched)
  responses_router.py        # NEW
  responses_models.py        # NEW
  responses_streaming.py     # NEW
  __init__.py                # updated with new exports
```

## Data Models (`responses_models.py`)

### Request

```python
class ResponseRequest(OpenAIBaseModel):
    model: str
    input: str | list[dict]
    instructions: str | None = None
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    metadata: dict | None = None
    # extra="allow" via OpenAIBaseModel for forward compat
```

`input` accepts either a plain string (converted to a user message internally) or a list of input item dicts matching the OpenAI spec.

### Output Item Types

```python
class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict] = []

class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    status: str = "completed"
    role: Literal["assistant"] = "assistant"
    content: list[dict]

class ResponseFunctionCall(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: str = "completed"
```

### Response Object

```python
class Response(OpenAIBaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: str = "completed"
    model: str
    output: list[dict]
    usage: dict | None = None
    error: dict | None = None
    incomplete_details: dict | None = None
    # extra="allow" for fields like instructions, tools, etc.
```

### Return Type

```python
ResponseResult = str | Response | Generator[Any, None, None] | AsyncGenerator[Any, None]
```

- `str` -- wrapped into a Response with a single output message containing output_text
- `Response` -- returned as-is (full control)
- `Generator` / `AsyncGenerator` -- streamed as named SSE events

## Router (`responses_router.py`)

### Factory Function

```python
def create_responses_router(
    *,
    list_models: ListModelsFn,
    run_response: RunResponseFn,         # (model, input, body) -> ResponseResult
    pre_hook: PreHook | None = None,     # (ResponseRequest) -> ResponseRequest | None
    post_hook: PostHook | None = None,   # (ResponseResult) -> ResponseResult | None
    chunk_mapper: ChunkMapper = default_chunk_mapper,
    owned_by: str = "custom",
    tags: list[str] | None = None,
) -> APIRouter
```

### Endpoints

| Method | Path | Operation |
|--------|------|-----------|
| GET | `/v1/models`, `/models` | List models (reuses `list_models` callback) |
| POST | `/v1/responses` | Create response |
| POST | `/responses` | Alias |

### Request Processing Flow

1. Parse request as `ResponseRequest`
2. Normalize `input`: if string, convert to `[{"type": "message", "role": "user", "content": input_str}]`
3. Call `pre_hook(request)` -- returns modified request (transformer) or None (observer)
4. Call `run_response(model, input_items, body)` where `body = request.model_dump()`
5. Call `post_hook(result)` -- returns modified result (transformer) or None (observer)
6. Build HTTP response based on result type

### Non-streaming Response

- `str` -> `response_from_text(text, resp_id, model)` builds a complete Response
- `Response` -> returned as-is

### Streaming Response

- Generator/AsyncGenerator -> wrapped via `create_responses_streaming_response()`

## Streaming (`responses_streaming.py`)

### SSE Event Format

The Responses API uses named events (unlike Chat Completions which uses unnamed `data:` lines):

```
event: response.created
data: {"type":"response.created","response":{...}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"chunk",...}

event: response.completed
data: {"type":"response.completed","response":{...}}
```

### Lifecycle Events (auto-generated envelope)

For simple text streaming (user yields strings or objects with `.content`):

1. `response.created` -- emitted once at start
2. `response.in_progress` -- emitted once
3. `response.output_item.added` -- when output message starts
4. `response.content_part.added` -- when text content part starts
5. `response.output_text.delta` -- for each chunk
6. `response.output_text.done` -- after last text chunk (includes full text)
7. `response.content_part.done` -- content part completed
8. `response.output_item.done` -- output item completed
9. `response.completed` -- final event with complete Response and usage

### Chunk Type Handling

| Chunk from generator | Behavior |
|---|---|
| `str` | Text delta |
| Object with `.content` | Content extracted as text delta |
| `Response` object | Serialized as `response.completed` directly |
| Object with `.to_event_dict()` | Custom SSE event (pass-through) |
| Object with function call attributes | Function call output item events |

### Helper Functions

- `response_from_text(text, resp_id, model)` -- build a non-streaming Response from plain text
- `create_responses_streaming_response()` -- sync generator wrapper
- `create_async_responses_streaming_response()` -- async generator wrapper

## Public API Changes (`__init__.py`)

New exports:

- `create_responses_router`
- `ResponseRequest`, `Response`, `ResponseOutputText`, `ResponseOutputMessage`, `ResponseFunctionCall`
- `ResponseResult`, `RunResponseFn`

All existing exports remain unchanged.

## Callback Comparison

| | Chat Completions | Responses API |
|---|---|---|
| Factory | `create_openai_router()` | `create_responses_router()` |
| Main callback | `run_completion(model, messages, body)` | `run_response(model, input, body)` |
| Input format | `messages: list[dict]` (role/content) | `input: list[dict]` (input items) |
| Return `str` | Wrapped as ChatCompletion | Wrapped as Response |
| Return full object | `ChatCompletion` | `Response` |
| Streaming | Generator yields chunks -> `chat.completion.chunk` SSE | Generator yields chunks -> named event SSE |
| Pre-hook receives | `ChatRequest` | `ResponseRequest` |
| Post-hook receives | `CompletionResult` | `ResponseResult` |

## Usage Example

```python
from fastapi import FastAPI
from fastapi_openai_compat import create_responses_router

app = FastAPI()

def list_models():
    return ["my-model"]

def run_response(model, input_items, body):
    user_text = ""
    for item in input_items:
        if item.get("role") == "user":
            content = item.get("content", "")
            if isinstance(content, str):
                user_text = content
    return f"Echo: {user_text}"

router = create_responses_router(
    list_models=list_models,
    run_response=run_response,
)
app.include_router(router)
```

## Testing Strategy

- Unit tests for Pydantic models (serialization, defaults, extra fields)
- Unit tests for streaming (event envelope generation, chunk mapping)
- Integration tests with httpx (non-streaming, streaming, hooks, function calls)
- OpenAI SDK compatibility test using `openai.responses.create()`
