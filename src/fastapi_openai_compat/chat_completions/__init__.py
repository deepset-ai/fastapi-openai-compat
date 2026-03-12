"""Chat Completions public module."""

from fastapi_openai_compat.chat_completions.models import (
    ChatCompletion,
    ChatRequest,
    Choice,
    Message,
    MessageParam,
    ModelObject,
    ModelsResponse,
    OpenAIBaseModel,
)
from fastapi_openai_compat.chat_completions.router import (
    CompletionResult,
    ListModelsFn,
    PostHook,
    PreHook,
    RunCompletionFn,
    create_chat_completion_router,
    create_openai_router,
)
from fastapi_openai_compat.chat_completions.streaming import (
    ChunkMapper,
    chat_completion_response,
    create_async_streaming_response,
    create_sse_data_msg,
    create_sync_streaming_response,
    default_chunk_mapper,
    event_to_sse_msg,
)

__all__ = [
    "ChatCompletion",
    "ChatRequest",
    "Choice",
    "ChunkMapper",
    "CompletionResult",
    "ListModelsFn",
    "Message",
    "MessageParam",
    "ModelObject",
    "ModelsResponse",
    "OpenAIBaseModel",
    "PostHook",
    "PreHook",
    "RunCompletionFn",
    "chat_completion_response",
    "create_async_streaming_response",
    "create_chat_completion_router",
    "create_openai_router",
    "create_sse_data_msg",
    "create_sync_streaming_response",
    "default_chunk_mapper",
    "event_to_sse_msg",
]
