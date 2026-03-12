"""Shared OpenAI-compatible /models router factory."""

import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter

from fastapi_openai_compat._async_utils import ensure_async
from fastapi_openai_compat.chat_completions.models import ModelObject, ModelsResponse

ListModelsFn = Callable[..., Any]


def _operation_id(prefix: str, suffix: str) -> str:
    normalized = prefix.strip()
    if not normalized:
        return suffix
    return f"{normalized}_{suffix}"


def create_models_router(
    *,
    list_models: ListModelsFn,
    owned_by: str = "custom",
    tags: list[str] | None = None,
    operation_id_prefix: str = "openai",
) -> APIRouter:
    """
    Create a FastAPI APIRouter exposing OpenAI-compatible model listing endpoints.

    Exposes:
    - ``GET /v1/models``
    - ``GET /models`` (alias)
    """
    _list_models = ensure_async(list_models)
    _tags = tags or ["openai"]

    router = APIRouter()

    models_params: dict[str, Any] = {
        "response_model": ModelsResponse,
        "tags": _tags,
        "summary": "List models",
        "description": ("Returns a list of available models (deployed pipelines) in OpenAI-compatible format."),
    }

    @router.get("/v1/models", **models_params, operation_id=_operation_id(operation_id_prefix, "models"))
    @router.get("/models", **models_params, operation_id=_operation_id(operation_id_prefix, "models_alias"))
    async def get_models() -> ModelsResponse:
        raw = await _list_models()
        names: list[str] = list(raw) if raw else []
        return ModelsResponse(
            data=[
                ModelObject(
                    id=str(name),
                    name=str(name),
                    object="model",
                    created=int(time.time()),
                    owned_by=owned_by,
                )
                for name in names
            ],
            object="list",
        )

    return router


__all__ = ["ListModelsFn", "create_models_router"]
