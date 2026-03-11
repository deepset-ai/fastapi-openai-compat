"""
Backward-compatible chat router re-export module.

.. deprecated::
    Import from ``fastapi_openai_compat.chat_completions.router`` instead.
"""

import warnings

warnings.warn(
    "Importing from 'fastapi_openai_compat.router' is deprecated. "
    "Use 'fastapi_openai_compat.chat_completions.router' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from fastapi_openai_compat.chat_completions.router import (  # noqa: E402
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
