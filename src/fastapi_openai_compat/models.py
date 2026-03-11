"""
Backward-compatible chat models re-export module.

.. deprecated::
    Import from ``fastapi_openai_compat.chat_completions.models`` instead.
"""

import warnings

warnings.warn(
    "Importing from 'fastapi_openai_compat.models' is deprecated. "
    "Use 'fastapi_openai_compat.chat_completions.models' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from fastapi_openai_compat.chat_completions.models import (  # noqa: E402
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
