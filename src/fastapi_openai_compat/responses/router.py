"""Responses API router factory."""

import logging
import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from fastapi_openai_compat._async_utils import ensure_async
from fastapi_openai_compat._shared import ChunkMapper, PostHook, PreHook, default_chunk_mapper
from fastapi_openai_compat.models_router import ListModelsFn, create_models_router
from fastapi_openai_compat.responses.models import InputItem, Response, ResponseRequest
from fastapi_openai_compat.responses.streaming import (
    create_async_responses_streaming_response,
    create_responses_streaming_response,
    response_from_text,
)

logger = logging.getLogger("fastapi_openai_compat.responses")

ResponseResult = str | Response | Generator[Any, None, None] | AsyncGenerator[Any, None]
RunResponseFn = Callable[..., Any]


def _normalize_input(input_value: str | list[InputItem] | None) -> list[InputItem]:
    if input_value is None:
        return []
    if isinstance(input_value, str):
        return [{"type": "message", "role": "user", "content": input_value}]
    return input_value


async def _default_pre_hook(response_request: ResponseRequest) -> ResponseRequest:
    return response_request


async def _default_post_hook(result: ResponseResult) -> ResponseResult:
    return result


def create_responses_router(  # noqa: PLR0913, C901
    *,
    list_models: ListModelsFn,
    run_response: RunResponseFn,
    pre_hook: PreHook | None = None,
    post_hook: PostHook | None = None,
    chunk_mapper: ChunkMapper = default_chunk_mapper,
    owned_by: str = "custom",
    tags: list[str] | None = None,
    include_models_endpoints: bool = False,
) -> APIRouter:
    """
    Create a FastAPI APIRouter with OpenAI-compatible Responses API endpoints.

    All callable parameters accept both sync and async functions.
    Sync functions are automatically executed in a thread pool to avoid
    blocking the async event loop.

    The ``run_response`` callback is invoked as
    ``(model, input_items, body)`` where:

    - ``model`` is the selected model name from the request.
    - ``input_items: list[InputItem]`` is the normalized request input
      (string shorthand converted to a ``message`` input item, or an
      empty list when ``input`` is omitted).
    - ``body`` is the full request body dictionary, including extra
      parameters.

    Args:
        list_models: Callable returning a list of available model names.
        run_response: Callable that runs a response request and returns one of:
            - `str` for simple text output.
            - `Response` for full manual response control.
            - `Generator` or `AsyncGenerator` for streaming named SSE events.
        pre_hook: Optional callable invoked before `run_response`.
            Receives `ResponseRequest` and may return a modified request
            (transformer) or `None` (observer).
        post_hook: Optional callable invoked after `run_response`.
            Receives `ResponseResult` and may return a modified result
            (transformer) or `None` (observer).
        chunk_mapper: Callable used to map streamed non-string chunks to text.
        owned_by: Owner value used in `/v1/models` responses.
        tags: Optional OpenAPI tags for generated endpoints.
        include_models_endpoints: If true, exposes `/v1/models` and `/models`
            from this router. Defaults to false to allow coexistence with
            a chat-completions router in the same app.
    """
    _run_response = ensure_async(run_response)
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
                operation_id_prefix="responses_openai",
            )
        )

    responses_params: dict[str, Any] = {
        "response_model": Response,
        "tags": _tags,
        "summary": "Create response",
        "description": (
            "Generates a response for OpenAI Responses API compatible clients. "
            "Supports non-streaming and named SSE streaming event formats."
        ),
    }

    @router.post("/v1/responses", **responses_params, operation_id="openai_responses")
    @router.post("/responses", **responses_params, operation_id="openai_responses_alias")
    async def responses_endpoint(response_req: ResponseRequest) -> Response | StreamingResponse:
        try:
            pre_result = await _pre_hook(response_req)
            if pre_result is not None:
                response_req = pre_result

            input_items = _normalize_input(response_req.input)
            result: ResponseResult = await _run_response(
                response_req.model,
                input_items,
                response_req.model_dump(),
            )

            post_result = await _post_hook(result)
            if post_result is not None:
                result = post_result

        except HTTPException:
            raise
        except Exception as exc:
            error_msg = f"Pipeline execution failed: {exc!s}"
            logger.exception("Responses API execution error")
            raise HTTPException(status_code=500, detail=error_msg) from exc

        if isinstance(result, Response):
            return result

        resp_id = f"resp_{uuid.uuid4().hex}"
        if isinstance(result, str):
            return response_from_text(result, resp_id, response_req.model)

        if isinstance(result, Generator):
            return create_responses_streaming_response(result, resp_id, response_req.model, _chunk_mapper)

        if isinstance(result, AsyncGenerator):
            return create_async_responses_streaming_response(result, resp_id, response_req.model, _chunk_mapper)

        raise HTTPException(status_code=500, detail="Unsupported response type from run_response")

    return router


__all__ = [
    "ResponseResult",
    "RunResponseFn",
    "create_responses_router",
]
