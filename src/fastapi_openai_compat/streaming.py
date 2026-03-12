"""
Backward-compatible chat streaming re-export module.

.. deprecated::
    Import from ``fastapi_openai_compat.chat_completions.streaming`` instead.
"""

import warnings

warnings.warn(
    "Importing from 'fastapi_openai_compat.streaming' is deprecated. "
    "Use 'fastapi_openai_compat.chat_completions.streaming' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from fastapi_openai_compat.chat_completions.streaming import (  # noqa: E402
    ChunkMapper,
    chat_completion_response,
    create_async_streaming_response,
    create_sse_data_msg,
    create_sync_streaming_response,
    default_chunk_mapper,
    event_to_sse_msg,
)

__all__ = [
    "ChunkMapper",
    "chat_completion_response",
    "create_async_streaming_response",
    "create_sse_data_msg",
    "create_sync_streaming_response",
    "default_chunk_mapper",
    "event_to_sse_msg",
]
