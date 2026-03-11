"""Backward-compatible chat models re-export module."""

from fastapi_openai_compat.chat_completions.models import (
    ChatCompletion,
    ChatRequest,
    Choice,
    Message,
    ModelObject,
    ModelsResponse,
    OpenAIBaseModel,
)

__all__ = [
    "ChatCompletion",
    "ChatRequest",
    "Choice",
    "Message",
    "ModelObject",
    "ModelsResponse",
    "OpenAIBaseModel",
]
