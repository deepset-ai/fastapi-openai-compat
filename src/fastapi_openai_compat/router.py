"""Backward-compatible chat router re-export module."""

from fastapi_openai_compat.chat_completions.router import (
    CompletionResult,
    ListModelsFn,
    PostHook,
    PreHook,
    RunCompletionFn,
    create_chat_completion_router,
    create_openai_router,
)

__all__ = [
    "CompletionResult",
    "ListModelsFn",
    "PostHook",
    "PreHook",
    "RunCompletionFn",
    "create_chat_completion_router",
    "create_openai_router",
]
