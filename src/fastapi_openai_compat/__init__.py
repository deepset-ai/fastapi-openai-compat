"""
FastAPI OpenAI-compatible API toolkit.

Re-exports the primary public symbols for convenience. Low-level streaming
helpers and individual SSE event constructors are available from the
``chat_completions``, ``responses``, and ``files`` subpackages.
"""

from fastapi_openai_compat._shared import ChunkMapper, OpenAIBaseModel, PostHook, PreHook
from fastapi_openai_compat.chat_completions import (
    ChatCompletion,
    ChatRequest,
    Choice,
    CompletionResult,
    ListModelsFn,
    Message,
    MessageParam,
    ModelObject,
    ModelsResponse,
    RunCompletionFn,
    create_chat_completion_router,
    create_openai_router,
)
from fastapi_openai_compat.files import FileObject, FileUploadResult, RunFileUploadFn, create_files_router
from fastapi_openai_compat.models_router import create_models_router
from fastapi_openai_compat.responses import (
    InputItem,
    OutputItem,
    Response,
    ResponseFunctionCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseRequest,
    ResponseResult,
    RunResponseFn,
    create_responses_router,
)

__all__ = [
    "ChatCompletion",
    "ChatRequest",
    "Choice",
    "ChunkMapper",
    "CompletionResult",
    "FileObject",
    "FileUploadResult",
    "InputItem",
    "ListModelsFn",
    "Message",
    "MessageParam",
    "ModelObject",
    "ModelsResponse",
    "OpenAIBaseModel",
    "OutputItem",
    "PostHook",
    "PreHook",
    "Response",
    "ResponseFunctionCall",
    "ResponseOutputMessage",
    "ResponseOutputText",
    "ResponseRequest",
    "ResponseResult",
    "RunCompletionFn",
    "RunFileUploadFn",
    "RunResponseFn",
    "create_chat_completion_router",
    "create_files_router",
    "create_models_router",
    "create_openai_router",
    "create_responses_router",
]
