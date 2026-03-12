"""Chat Completions router factory."""

import logging
import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from fastapi_openai_compat._async_utils import ensure_async
from fastapi_openai_compat._shared import ChunkMapper, PostHook, PreHook, default_chunk_mapper
from fastapi_openai_compat.chat_completions.models import ChatCompletion, ChatRequest
from fastapi_openai_compat.chat_completions.streaming import (
    chat_completion_response,
    create_async_streaming_response,
    create_sync_streaming_response,
)
from fastapi_openai_compat.models_router import ListModelsFn, create_models_router

logger = logging.getLogger("fastapi_openai_compat.chat_completions")

CompletionResult = str | ChatCompletion | Generator[Any, None, None] | AsyncGenerator[Any, None]
RunCompletionFn = Callable[..., Any]


async def _default_pre_hook(chat_request: ChatRequest) -> ChatRequest:
    """Default pre-hook: passthrough."""
    return chat_request


async def _default_post_hook(result: CompletionResult) -> CompletionResult:
    """Default post-hook: passthrough."""
    return result


def create_chat_completion_router(  # noqa: PLR0913, C901
    *,
    list_models: ListModelsFn,
    run_completion: RunCompletionFn,
    pre_hook: PreHook | None = None,
    post_hook: PostHook | None = None,
    chunk_mapper: ChunkMapper = default_chunk_mapper,
    owned_by: str = "custom",
    tags: list[str] | None = None,
    include_models_endpoints: bool = True,
) -> APIRouter:
    """
    Create a FastAPI APIRouter with OpenAI-compatible chat completion endpoints.

    All callable parameters accept both sync and async functions.
    Sync functions are automatically executed in a thread pool to avoid
    blocking the async event loop.

    Args:
        list_models: Callable returning a list of available model/pipeline names.
        run_completion: Callable that runs a chat completion given
            ``(model, messages: list[MessageParam], body)`` and returns one of:

            - ``str`` -- plain text, wrapped as ``Message(content=str)``.
            - ``ChatCompletion`` -- returned as-is (full control over
              tool_calls, finish_reason, usage, etc.).
            - ``Generator`` / ``AsyncGenerator`` -- streamed as SSE.
              Yields of ``ChatCompletion`` are serialized directly;
              other yields go through ``chunk_mapper``.
        pre_hook: Optional callable invoked **before** ``run_completion``.
            Receives the parsed ``ChatRequest`` and may return a modified
            ``ChatRequest`` (transformer) or ``None`` (observer).
            Raise ``HTTPException`` to abort.
        post_hook: Optional callable invoked **after** ``run_completion``
            but before the response is formatted. Receives the raw
            ``CompletionResult`` and may return a modified ``CompletionResult``
            (transformer) or ``None`` (observer).
        chunk_mapper: Callable that converts each streamed chunk into a ``str``
            for the SSE response. The default handles plain strings and objects
            with a ``.content`` attribute (e.g. Haystack ``StreamingChunk``).
            Override this to support custom chunk types from your pipeline.
        owned_by: Value used in the ``owned_by`` field of model objects.
        tags: OpenAPI tags applied to the generated endpoints.
        include_models_endpoints: If true, includes ``/v1/models`` and
            ``/models`` from this router. Set to false when mounting a shared
            models router separately.

    Returns:
        A configured ``APIRouter`` ready to be included in a FastAPI app.
    """
    _run_completion = ensure_async(run_completion)
    _pre_hook = ensure_async(pre_hook) if pre_hook else _default_pre_hook
    _post_hook = ensure_async(post_hook) if post_hook else _default_post_hook
    _chunk_mapper = chunk_mapper
    _tags = tags or ["openai"]

    router = APIRouter()

    if include_models_endpoints:
        router.include_router(
            create_models_router(
                list_models=list_models,
                owned_by=owned_by,
                tags=_tags,
                operation_id_prefix="openai",
            )
        )

    chat_params: dict = {
        "response_model": ChatCompletion,
        "tags": _tags,
        "summary": "Create chat completion",
        "description": (
            "Generates a chat completion for the given conversation in OpenAI-compatible format.\n\n"
            "**Non-streaming** (`stream: false`, default): returns a single JSON `ChatCompletion` object.\n\n"
            "**Streaming** (`stream: true`): returns a stream of server-sent events (SSE), "
            "each containing a `ChatCompletion` chunk with incremental content in `choices[].delta`.\n\n"
            "Any extra fields in the request body (e.g. `temperature`, `max_tokens`, `top_p`) "
            "are forwarded to the underlying pipeline execution.\n\n"
            "References:\n"
            "- [OpenAI Chat Completions API]"
            "(https://platform.openai.com/docs/api-reference/chat/create)"
        ),
        "responses": {
            200: {
                "description": (
                    "Successful completion. Returns a `ChatCompletion` JSON object "
                    "or an SSE stream depending on the `stream` parameter."
                ),
            },
            500: {
                "description": "Pipeline execution failed.",
                "content": {
                    "application/json": {
                        "example": {"detail": "Pipeline execution failed: <error message>"},
                    }
                },
            },
        },
    }

    @router.post("/v1/chat/completions", **chat_params, operation_id="openai_chat_completions")
    @router.post("/chat/completions", **chat_params, operation_id="openai_chat_completions_alias")
    async def chat_endpoint(chat_req: ChatRequest) -> ChatCompletion | StreamingResponse:
        try:
            pre_result = await _pre_hook(chat_req)
            if pre_result is not None:
                chat_req = pre_result

            result: CompletionResult = await _run_completion(
                chat_req.model,
                chat_req.messages,
                chat_req.model_dump(),
            )

            post_result = await _post_hook(result)
            if post_result is not None:
                result = post_result

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Pipeline execution error")
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc!s}") from exc

        if isinstance(result, ChatCompletion):
            return result

        resp_id = f"{chat_req.model}-{uuid.uuid4()}"

        if isinstance(result, str):
            return chat_completion_response(result, resp_id, chat_req.model)

        if isinstance(result, Generator):
            return create_sync_streaming_response(result, resp_id, chat_req.model, _chunk_mapper)

        if isinstance(result, AsyncGenerator):
            return create_async_streaming_response(result, resp_id, chat_req.model, _chunk_mapper)

        raise HTTPException(status_code=500, detail="Unsupported response type from completion")

    return router


def create_openai_router(  # noqa: PLR0913
    *,
    list_models: ListModelsFn,
    run_completion: RunCompletionFn,
    pre_hook: PreHook | None = None,
    post_hook: PostHook | None = None,
    chunk_mapper: ChunkMapper = default_chunk_mapper,
    owned_by: str = "custom",
    tags: list[str] | None = None,
    include_models_endpoints: bool = True,
) -> APIRouter:
    """
    Backward-compatible alias for `create_chat_completion_router`.

    Existing integrations can keep using `create_openai_router`; new code should
    prefer `create_chat_completion_router` for clarity.
    """
    return create_chat_completion_router(
        list_models=list_models,
        run_completion=run_completion,
        pre_hook=pre_hook,
        post_hook=post_hook,
        chunk_mapper=chunk_mapper,
        owned_by=owned_by,
        tags=tags,
        include_models_endpoints=include_models_endpoints,
    )


__all__ = [
    "CompletionResult",
    "PostHook",
    "PreHook",
    "RunCompletionFn",
    "create_chat_completion_router",
    "create_openai_router",
]
