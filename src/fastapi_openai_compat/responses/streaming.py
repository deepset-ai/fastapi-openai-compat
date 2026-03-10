"""Responses API streaming helpers."""

import json
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Any

from fastapi.responses import StreamingResponse

from fastapi_openai_compat.responses.models import Response
from fastapi_openai_compat.streaming import ChunkMapper, default_chunk_mapper


def format_named_sse_event(event_name: str, data_dict: dict[str, Any]) -> str:
    """Format a named SSE event."""
    return f"event: {event_name}\ndata: {json.dumps(data_dict)}\n\n"


def response_from_text(text: str, resp_id: str, model_name: str) -> Response:
    """Build a full Response object from plain text."""
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [part],
    }
    return Response(
        id=resp_id,
        object="response",
        created_at=int(time.time()),
        status="completed",
        model=model_name,
        output=[item],
    )


def create_response_created_event(response: Response) -> str:
    return format_named_sse_event("response.created", {"type": "response.created", "response": response.model_dump()})


def create_response_in_progress_event(response: Response) -> str:
    return format_named_sse_event(
        "response.in_progress",
        {"type": "response.in_progress", "response": response.model_dump()},
    )


def create_output_item_added_event(output_index: int, item: dict[str, Any]) -> str:
    return format_named_sse_event(
        "response.output_item.added",
        {"type": "response.output_item.added", "output_index": output_index, "item": item},
    )


def create_content_part_added_event(
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict[str, Any],
) -> str:
    return format_named_sse_event(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        },
    )


def create_output_text_delta_event(item_id: str, output_index: int, content_index: int, delta: str) -> str:
    return format_named_sse_event(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
        },
    )


def create_output_text_done_event(item_id: str, output_index: int, content_index: int, text: str) -> str:
    return format_named_sse_event(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": text,
        },
    )


def create_function_call_arguments_delta_event(item_id: str, output_index: int, delta: str) -> str:
    return format_named_sse_event(
        "response.function_call_arguments.delta",
        {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "output_index": output_index,
            "delta": delta,
        },
    )


def create_function_call_arguments_done_event(item_id: str, output_index: int, arguments: str) -> str:
    return format_named_sse_event(
        "response.function_call_arguments.done",
        {
            "type": "response.function_call_arguments.done",
            "item_id": item_id,
            "output_index": output_index,
            "arguments": arguments,
        },
    )


def create_content_part_done_event(
    item_id: str,
    output_index: int,
    content_index: int,
    part: dict[str, Any],
) -> str:
    return format_named_sse_event(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        },
    )


def create_output_item_done_event(output_index: int, item: dict[str, Any]) -> str:
    return format_named_sse_event(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        },
    )


def create_response_completed_event(response: Response) -> str:
    return format_named_sse_event(
        "response.completed",
        {"type": "response.completed", "response": response.model_dump()},
    )


def _is_custom_event(chunk: Any) -> bool:
    return callable(getattr(chunk, "to_event_dict", None))


def _is_function_call_chunk(chunk: Any) -> bool:
    return (
        hasattr(chunk, "function_call_id")
        and chunk.function_call_id is not None
        and hasattr(chunk, "function_call_arguments")
    )


def _text_from_chunk(chunk: Any, chunk_mapper: ChunkMapper) -> str:
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "content"):
        content = chunk.content
        if isinstance(content, str):
            return content
    mapped = chunk_mapper(chunk)
    return mapped if isinstance(mapped, str) else str(mapped)


def _in_progress_response(resp_id: str, model_name: str) -> Response:
    return Response(
        id=resp_id,
        object="response",
        created_at=int(time.time()),
        status="in_progress",
        model=model_name,
        output=[],
    )


def create_responses_streaming_response(  # noqa: C901, PLR0915
    result: Generator[Any, None, None],
    resp_id: str,
    model_name: str,
    chunk_mapper: ChunkMapper = default_chunk_mapper,
) -> StreamingResponse:
    """Wrap a sync generator and emit OpenAI Responses API named events."""

    def stream_events() -> Generator[str, None, None]:  # noqa: C901, PLR0912, PLR0915
        response = _in_progress_response(resp_id, model_name)
        yield create_response_created_event(response)
        yield create_response_in_progress_event(response)

        output_index = 0
        content_index = 0
        text_item_id: str | None = None
        text_parts: list[str] = []
        function_call_item_id: str | None = None
        function_call_call_id: str | None = None
        function_call_name: str | None = None
        function_call_argument_chunks: list[str] = []
        final_output: list[dict[str, Any]] = []

        def _start_text_item() -> list[str]:
            nonlocal text_item_id
            if text_item_id is not None:
                return []
            text_item_id = f"msg_{uuid.uuid4().hex}"
            in_progress_item = {
                "id": text_item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            part_template = {"type": "output_text", "text": "", "annotations": []}
            return [
                create_output_item_added_event(output_index, in_progress_item),
                create_content_part_added_event(text_item_id, output_index, content_index, part_template),
            ]

        def _finalize_text_item() -> list[str]:
            nonlocal text_item_id, text_parts, output_index
            if text_item_id is None:
                return []
            full_text = "".join(text_parts)
            final_part = {"type": "output_text", "text": full_text, "annotations": []}
            final_item = {
                "id": text_item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [final_part],
            }
            final_output.append(final_item)
            events = [
                create_output_text_done_event(text_item_id, output_index, content_index, full_text),
                create_content_part_done_event(text_item_id, output_index, content_index, final_part),
                create_output_item_done_event(output_index, final_item),
            ]
            text_item_id = None
            text_parts = []
            output_index += 1
            return events

        def _start_function_call_item(call_id: str, name: str | None) -> list[str]:
            nonlocal function_call_item_id, function_call_call_id, function_call_name
            nonlocal function_call_argument_chunks
            if function_call_item_id is not None:
                return []
            function_call_item_id = f"fc_{uuid.uuid4().hex}"
            function_call_call_id = call_id
            function_call_name = name or "function"
            function_call_argument_chunks = []
            in_progress_item = {
                "id": function_call_item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": function_call_call_id,
                "name": function_call_name,
                "arguments": "",
            }
            return [create_output_item_added_event(output_index, in_progress_item)]

        def _finalize_function_call_item() -> list[str]:
            nonlocal function_call_item_id, function_call_call_id, function_call_name
            nonlocal function_call_argument_chunks, output_index
            if function_call_item_id is None:
                return []
            full_arguments = "".join(function_call_argument_chunks)
            final_item = {
                "id": function_call_item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": function_call_call_id,
                "name": function_call_name or "function",
                "arguments": full_arguments,
            }
            final_output.append(final_item)
            events = [
                create_function_call_arguments_done_event(function_call_item_id, output_index, full_arguments),
                create_output_item_done_event(output_index, final_item),
            ]
            function_call_item_id = None
            function_call_call_id = None
            function_call_name = None
            function_call_argument_chunks = []
            output_index += 1
            return events

        for chunk in result:
            if isinstance(chunk, Response):
                yield create_response_completed_event(chunk)
                return

            if _is_custom_event(chunk):
                event = chunk.to_event_dict()
                event_name = event.get("type", "response.event")
                yield format_named_sse_event(event_name, event)
                continue

            if _is_function_call_chunk(chunk):
                for event in _finalize_text_item():
                    yield event
                call_id = str(chunk.function_call_id)
                name = chunk.function_call_name if hasattr(chunk, "function_call_name") else None
                raw_arguments = chunk.function_call_arguments
                arguments_delta = "" if raw_arguments is None else str(raw_arguments)

                for event in _start_function_call_item(call_id, name):
                    yield event
                if name is not None:
                    function_call_name = name

                if arguments_delta:
                    function_call_argument_chunks.append(arguments_delta)
                    if function_call_item_id is None:
                        msg = "Function call item ID must be initialized before delta events."
                        raise RuntimeError(msg)
                    yield create_function_call_arguments_delta_event(
                        function_call_item_id,
                        output_index,
                        arguments_delta,
                    )
                continue

            text = _text_from_chunk(chunk, chunk_mapper)
            if not text:
                continue

            for event in _finalize_function_call_item():
                yield event
            for event in _start_text_item():
                yield event

            text_parts.append(text)
            if text_item_id is None:
                msg = "Text item ID must be initialized before delta events."
                raise RuntimeError(msg)
            yield create_output_text_delta_event(text_item_id, output_index, content_index, text)

        for event in _finalize_text_item():
            yield event
        for event in _finalize_function_call_item():
            yield event

        completed = Response(
            id=resp_id,
            object="response",
            created_at=response.created_at,
            status="completed",
            model=model_name,
            output=final_output,
        )
        yield create_response_completed_event(completed)

    return StreamingResponse(stream_events(), media_type="text/event-stream")


def create_async_responses_streaming_response(  # noqa: C901, PLR0915
    result: AsyncGenerator[Any, None],
    resp_id: str,
    model_name: str,
    chunk_mapper: ChunkMapper = default_chunk_mapper,
) -> StreamingResponse:
    """Wrap an async generator and emit OpenAI Responses API named events."""

    async def stream_events_async() -> AsyncGenerator[str, None]:  # noqa: C901, PLR0912, PLR0915
        response = _in_progress_response(resp_id, model_name)
        yield create_response_created_event(response)
        yield create_response_in_progress_event(response)

        output_index = 0
        content_index = 0
        text_item_id: str | None = None
        text_parts: list[str] = []
        function_call_item_id: str | None = None
        function_call_call_id: str | None = None
        function_call_name: str | None = None
        function_call_argument_chunks: list[str] = []
        final_output: list[dict[str, Any]] = []

        def _start_text_item() -> list[str]:
            nonlocal text_item_id
            if text_item_id is not None:
                return []
            text_item_id = f"msg_{uuid.uuid4().hex}"
            in_progress_item = {
                "id": text_item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            part_template = {"type": "output_text", "text": "", "annotations": []}
            return [
                create_output_item_added_event(output_index, in_progress_item),
                create_content_part_added_event(text_item_id, output_index, content_index, part_template),
            ]

        def _finalize_text_item() -> list[str]:
            nonlocal text_item_id, text_parts, output_index
            if text_item_id is None:
                return []
            full_text = "".join(text_parts)
            final_part = {"type": "output_text", "text": full_text, "annotations": []}
            final_item = {
                "id": text_item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [final_part],
            }
            final_output.append(final_item)
            events = [
                create_output_text_done_event(text_item_id, output_index, content_index, full_text),
                create_content_part_done_event(text_item_id, output_index, content_index, final_part),
                create_output_item_done_event(output_index, final_item),
            ]
            text_item_id = None
            text_parts = []
            output_index += 1
            return events

        def _start_function_call_item(call_id: str, name: str | None) -> list[str]:
            nonlocal function_call_item_id, function_call_call_id, function_call_name
            nonlocal function_call_argument_chunks
            if function_call_item_id is not None:
                return []
            function_call_item_id = f"fc_{uuid.uuid4().hex}"
            function_call_call_id = call_id
            function_call_name = name or "function"
            function_call_argument_chunks = []
            in_progress_item = {
                "id": function_call_item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": function_call_call_id,
                "name": function_call_name,
                "arguments": "",
            }
            return [create_output_item_added_event(output_index, in_progress_item)]

        def _finalize_function_call_item() -> list[str]:
            nonlocal function_call_item_id, function_call_call_id, function_call_name
            nonlocal function_call_argument_chunks, output_index
            if function_call_item_id is None:
                return []
            full_arguments = "".join(function_call_argument_chunks)
            final_item = {
                "id": function_call_item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": function_call_call_id,
                "name": function_call_name or "function",
                "arguments": full_arguments,
            }
            final_output.append(final_item)
            events = [
                create_function_call_arguments_done_event(function_call_item_id, output_index, full_arguments),
                create_output_item_done_event(output_index, final_item),
            ]
            function_call_item_id = None
            function_call_call_id = None
            function_call_name = None
            function_call_argument_chunks = []
            output_index += 1
            return events

        async for chunk in result:
            if isinstance(chunk, Response):
                yield create_response_completed_event(chunk)
                return

            if _is_custom_event(chunk):
                event = chunk.to_event_dict()
                event_name = event.get("type", "response.event")
                yield format_named_sse_event(event_name, event)
                continue

            if _is_function_call_chunk(chunk):
                for event in _finalize_text_item():
                    yield event
                call_id = str(chunk.function_call_id)
                name = chunk.function_call_name if hasattr(chunk, "function_call_name") else None
                raw_arguments = chunk.function_call_arguments
                arguments_delta = "" if raw_arguments is None else str(raw_arguments)

                for event in _start_function_call_item(call_id, name):
                    yield event
                if name is not None:
                    function_call_name = name

                if arguments_delta:
                    function_call_argument_chunks.append(arguments_delta)
                    if function_call_item_id is None:
                        msg = "Function call item ID must be initialized before delta events."
                        raise RuntimeError(msg)
                    yield create_function_call_arguments_delta_event(
                        function_call_item_id,
                        output_index,
                        arguments_delta,
                    )
                continue

            text = _text_from_chunk(chunk, chunk_mapper)
            if not text:
                continue

            for event in _finalize_function_call_item():
                yield event
            for event in _start_text_item():
                yield event

            text_parts.append(text)
            if text_item_id is None:
                msg = "Text item ID must be initialized before delta events."
                raise RuntimeError(msg)
            yield create_output_text_delta_event(text_item_id, output_index, content_index, text)

        for event in _finalize_text_item():
            yield event
        for event in _finalize_function_call_item():
            yield event

        completed = Response(
            id=resp_id,
            object="response",
            created_at=response.created_at,
            status="completed",
            model=model_name,
            output=final_output,
        )
        yield create_response_completed_event(completed)

    return StreamingResponse(stream_events_async(), media_type="text/event-stream")


__all__ = [
    "create_async_responses_streaming_response",
    "create_function_call_arguments_delta_event",
    "create_function_call_arguments_done_event",
    "create_output_item_added_event",
    "create_output_item_done_event",
    "create_output_text_delta_event",
    "create_output_text_done_event",
    "create_response_completed_event",
    "create_response_created_event",
    "create_response_in_progress_event",
    "create_responses_streaming_response",
    "format_named_sse_event",
    "response_from_text",
]
