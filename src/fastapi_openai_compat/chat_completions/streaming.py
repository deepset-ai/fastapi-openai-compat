"""Chat Completions streaming helpers."""

import json
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

from fastapi.responses import StreamingResponse

from fastapi_openai_compat._shared import ChunkMapper, _extract_reasoning_text, _is_custom_event, default_chunk_mapper
from fastapi_openai_compat.chat_completions.models import ChatCompletion, Choice, Message


def event_to_sse_msg(data: dict) -> str:
    """
    Format an arbitrary event dict as an SSE message.

    Useful for sending custom event payloads alongside chat completion chunks.
    """
    event_payload = {"event": data}
    return f"data: {json.dumps(event_payload)}\n\n"


def create_sse_data_msg(
    resp_id: str,
    model_name: str,
    chunk_content: str = "",
    finish_reason: str | None = None,
) -> str:
    """Format a single chat completion chunk as an SSE data message."""
    response = ChatCompletion(
        id=resp_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model_name,
        choices=[Choice(index=0, delta=Message(role="assistant", content=chunk_content), finish_reason=finish_reason)],
    )
    return f"data: {response.model_dump_json()}\n\n"


def _completion_to_sse(completion: ChatCompletion) -> str:
    """Serialize a pre-built ChatCompletion chunk to an SSE data message."""
    return f"data: {completion.model_dump_json()}\n\n"


def _has_tool_calls(chunk: Any) -> bool:
    """Check if a chunk carries tool call deltas (duck typing)."""
    tool_calls = getattr(chunk, "tool_calls", None)
    return tool_calls is not None and len(tool_calls) > 0


def _get_finish_reason(chunk: Any) -> str | None:
    """Extract finish_reason from a chunk via duck typing."""
    return getattr(chunk, "finish_reason", None)


def _tool_call_delta_to_openai(tc: Any) -> dict[str, Any]:
    """Convert a ToolCallDelta-like object to OpenAI wire format via duck typing."""
    result: dict[str, Any] = {
        "index": getattr(tc, "index", 0),
        "type": "function",
        "function": {},
    }
    tc_id = getattr(tc, "id", None)
    if tc_id is not None:
        result["id"] = tc_id
    tool_name = getattr(tc, "tool_name", None) or getattr(tc, "name", None)
    if tool_name is not None:
        result["function"]["name"] = tool_name
    arguments = getattr(tc, "arguments", None)
    if arguments is not None:
        result["function"]["arguments"] = arguments
    return result


def _reasoning_chunk_to_sse(reasoning_text: str, resp_id: str, model_name: str) -> str:
    """Build an SSE message with reasoning content on the delta."""
    response = ChatCompletion(
        id=resp_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model_name,
        choices=[
            Choice(
                index=0,
                delta=Message(role="assistant", content=None, reasoning_content=reasoning_text),
            )
        ],
    )
    return f"data: {response.model_dump_json()}\n\n"


def _tool_calls_chunk_to_sse(chunk: Any, resp_id: str, model_name: str) -> str:
    """Build an SSE message from a chunk carrying tool call deltas."""
    tool_calls_openai = [_tool_call_delta_to_openai(tc) for tc in chunk.tool_calls]
    finish_reason = _get_finish_reason(chunk)
    response = ChatCompletion(
        id=resp_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model_name,
        choices=[
            Choice(
                index=0,
                delta=Message(role="assistant", content=None, tool_calls=tool_calls_openai),
                finish_reason=finish_reason,
            )
        ],
    )
    return f"data: {response.model_dump_json()}\n\n"


def create_sync_streaming_response(
    result: Generator[Any, None, None],
    resp_id: str,
    model_name: str,
    chunk_mapper: ChunkMapper = default_chunk_mapper,
) -> StreamingResponse:
    """
    Wrap a synchronous generator of chunks into an SSE StreamingResponse.

    Handles five chunk types automatically:

    * ``ChatCompletion`` objects are serialized directly.
    * Objects with a ``to_event_dict()`` method are serialized as custom SSE
      events (e.g. Open WebUI status/notification events).
    * Chunks with a ``reasoning`` attribute (e.g. Haystack ``StreamingChunk``
      with ``ReasoningContent``) emit ``reasoning_content`` on the delta.
    * Chunks with a ``tool_calls`` attribute (e.g. Haystack ``StreamingChunk``)
      are converted to OpenAI-format tool call deltas.
    * All other chunks are mapped to text via ``chunk_mapper``.

    A ``finish_reason`` attribute on any non-``ChatCompletion`` chunk is
    propagated to the SSE message.  A final ``finish_reason="stop"`` sentinel
    is appended only when no chunk already carried a finish reason.
    Custom events do not affect finish reason tracking.
    """

    def stream_chunks() -> Generator[str, None, None]:
        has_non_completion_chunks = False
        has_explicit_finish = False
        for chunk in result:
            if isinstance(chunk, ChatCompletion):
                yield _completion_to_sse(chunk)
            elif _is_custom_event(chunk):
                yield event_to_sse_msg(chunk.to_event_dict())
            elif (reasoning_text := _extract_reasoning_text(chunk)) is not None:
                yield _reasoning_chunk_to_sse(reasoning_text, resp_id, model_name)
                has_non_completion_chunks = True
            elif _has_tool_calls(chunk):
                yield _tool_calls_chunk_to_sse(chunk, resp_id, model_name)
                has_non_completion_chunks = True
                if _get_finish_reason(chunk) is not None:
                    has_explicit_finish = True
            else:
                finish_reason = _get_finish_reason(chunk)
                yield create_sse_data_msg(
                    resp_id=resp_id,
                    model_name=model_name,
                    chunk_content=chunk_mapper(chunk),
                    finish_reason=finish_reason,
                )
                has_non_completion_chunks = True
                if finish_reason is not None:
                    has_explicit_finish = True
        if has_non_completion_chunks and not has_explicit_finish:
            yield create_sse_data_msg(resp_id=resp_id, model_name=model_name, finish_reason="stop")

    return StreamingResponse(stream_chunks(), media_type="text/event-stream")


def create_async_streaming_response(
    result: AsyncGenerator[Any, None],
    resp_id: str,
    model_name: str,
    chunk_mapper: ChunkMapper = default_chunk_mapper,
) -> StreamingResponse:
    """
    Wrap an asynchronous generator of chunks into an SSE StreamingResponse.

    Handles five chunk types automatically:

    * ``ChatCompletion`` objects are serialized directly.
    * Objects with a ``to_event_dict()`` method are serialized as custom SSE
      events (e.g. Open WebUI status/notification events).
    * Chunks with a ``reasoning`` attribute (e.g. Haystack ``StreamingChunk``
      with ``ReasoningContent``) emit ``reasoning_content`` on the delta.
    * Chunks with a ``tool_calls`` attribute (e.g. Haystack ``StreamingChunk``)
      are converted to OpenAI-format tool call deltas.
    * All other chunks are mapped to text via ``chunk_mapper``.

    A ``finish_reason`` attribute on any non-``ChatCompletion`` chunk is
    propagated to the SSE message.  A final ``finish_reason="stop"`` sentinel
    is appended only when no chunk already carried a finish reason.
    Custom events do not affect finish reason tracking.
    """

    async def stream_chunks_async() -> AsyncGenerator[str, None]:
        has_non_completion_chunks = False
        has_explicit_finish = False
        async for chunk in result:
            if isinstance(chunk, ChatCompletion):
                yield _completion_to_sse(chunk)
            elif _is_custom_event(chunk):
                yield event_to_sse_msg(chunk.to_event_dict())
            elif (reasoning_text := _extract_reasoning_text(chunk)) is not None:
                yield _reasoning_chunk_to_sse(reasoning_text, resp_id, model_name)
                has_non_completion_chunks = True
            elif _has_tool_calls(chunk):
                yield _tool_calls_chunk_to_sse(chunk, resp_id, model_name)
                has_non_completion_chunks = True
                if _get_finish_reason(chunk) is not None:
                    has_explicit_finish = True
            else:
                finish_reason = _get_finish_reason(chunk)
                yield create_sse_data_msg(
                    resp_id=resp_id,
                    model_name=model_name,
                    chunk_content=chunk_mapper(chunk),
                    finish_reason=finish_reason,
                )
                has_non_completion_chunks = True
                if finish_reason is not None:
                    has_explicit_finish = True
        if has_non_completion_chunks and not has_explicit_finish:
            yield create_sse_data_msg(resp_id=resp_id, model_name=model_name, finish_reason="stop")

    return StreamingResponse(stream_chunks_async(), media_type="text/event-stream")


def chat_completion_response(result: str, resp_id: str, model_name: str) -> ChatCompletion:
    """Create a non-streaming chat completion response from a plain string result."""
    return ChatCompletion(
        id=resp_id,
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[Choice(index=0, message=Message(role="assistant", content=result), finish_reason="stop")],
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
