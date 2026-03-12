# Future Improvements

Post-merge improvements to consider for upcoming releases. None are blockers —
the current implementation is solid, well-tested (160 tests), and production-ready.

## Deduplicate chat completions sync/async streaming

`chat_completions/streaming.py` has parallel `create_sync_streaming_response` and
`create_async_streaming_response` functions with duplicated chunk-dispatch logic
(ChatCompletion passthrough, custom events, tool calls, text chunks, finish reason
tracking).

The `responses/streaming.py` module already solved this with `_ResponseStreamState`.
Apply the same pattern: extract a `_ChatStreamState` class that owns the dispatch
logic, then make the sync/async wrappers thin iterators over it.

**Effort:** Small. The chat streaming logic is simpler (no nested items or
function call state), so the state class would be straightforward.

## Stricter callback type signatures

`RunCompletionFn`, `RunResponseFn`, `RunFileUploadFn`, and `ListModelsFn` are all
typed as `Callable[..., Any]`. This works but gives users zero IDE autocompletion
or type-checking on expected parameters.

Replace with `Protocol` classes:

```python
class RunCompletionFn(Protocol):
    def __call__(
        self, model: str, messages: list[dict], body: dict
    ) -> CompletionResult: ...

class RunResponseFn(Protocol):
    def __call__(
        self, model: str, input_items: list[dict], body: dict
    ) -> ResponseResult: ...
```

This requires defining both sync and async overloads (or using
`Union[Callable[..., T], Callable[..., Awaitable[T]]]`), which adds complexity.
Worth it for developer experience once the API is stable.

**Effort:** Medium. Needs careful handling of sync/async duality.

## PEP 561 `py.typed` marker

Add an empty `src/fastapi_openai_compat/py.typed` file so downstream projects
using mypy or pyright get proper type-checking support from this package.

**Effort:** Trivial.

## Files API: document upload size limits

`files/router.py` calls `await file.read()` which loads the entire upload into
memory. This is acceptable for a library (the app author controls deployment),
but should be documented clearly.

Options:

- Document in `create_files_router` docstring that callbacks should enforce limits.
- Add an optional `max_file_size` parameter that rejects uploads above a threshold
  before reading the full body.
- Recommend FastAPI middleware or reverse proxy (nginx) configuration for
  production deployments.

**Effort:** Small for docs, medium for the `max_file_size` parameter.

## Consolidate `ListModelsFn` location

`ListModelsFn` is defined in `models_router.py` and re-exported through
`chat_completions/__init__.py` and the root `__init__.py`. Since it's used by
both `chat_completions/router.py` and `responses/router.py`, it could live in
`_shared.py` alongside the other cross-cutting types. Low priority since the
current arrangement works.

**Effort:** Trivial.
