"""Responses API streaming helpers."""

import json
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Any

from fastapi.responses import StreamingResponse

from fastapi_openai_compat._shared import ChunkMapper, _extract_reasoning_text, _is_custom_event, default_chunk_mapper
from fastapi_openai_compat.responses.models import Response


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


def create_reasoning_summary_part_added_event(
    item_id: str,
    output_index: int,
    summary_index: int,
    part: dict[str, Any],
) -> str:
    return format_named_sse_event(
        "response.reasoning_summary_part.added",
        {
            "type": "response.reasoning_summary_part.added",
            "item_id": item_id,
            "output_index": output_index,
            "summary_index": summary_index,
            "part": part,
        },
    )


def create_reasoning_summary_text_delta_event(item_id: str, output_index: int, summary_index: int, delta: str) -> str:
    return format_named_sse_event(
        "response.reasoning_summary_text.delta",
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": item_id,
            "output_index": output_index,
            "summary_index": summary_index,
            "delta": delta,
        },
    )


def create_reasoning_summary_text_done_event(item_id: str, output_index: int, summary_index: int, text: str) -> str:
    return format_named_sse_event(
        "response.reasoning_summary_text.done",
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": item_id,
            "output_index": output_index,
            "summary_index": summary_index,
            "text": text,
        },
    )


def create_reasoning_summary_part_done_event(
    item_id: str,
    output_index: int,
    summary_index: int,
    part: dict[str, Any],
) -> str:
    return format_named_sse_event(
        "response.reasoning_summary_part.done",
        {
            "type": "response.reasoning_summary_part.done",
            "item_id": item_id,
            "output_index": output_index,
            "summary_index": summary_index,
            "part": part,
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


# ---------------------------------------------------------------------------
# Streaming state machine
# ---------------------------------------------------------------------------


class _ResponseStreamState:
    """
    Encapsulates the event-generation state for a Responses API stream.

    Both the sync and async streaming wrappers delegate all logic here so
    chunk processing, text/function-call item tracking, and finalization
    are defined exactly once.
    """

    def __init__(self, resp_id: str, model_name: str, chunk_mapper: ChunkMapper) -> None:
        self._resp_id = resp_id
        self._model_name = model_name
        self._chunk_mapper = chunk_mapper

        self._response = Response(
            id=resp_id,
            object="response",
            created_at=int(time.time()),
            status="in_progress",
            model=model_name,
            output=[],
        )

        self._output_index = 0
        self._content_index = 0
        self._text_item_id: str | None = None
        self._text_parts: list[str] = []
        self._reasoning_item_id: str | None = None
        self._reasoning_parts: list[str] = []
        self._fc_item_id: str | None = None
        self._fc_call_id: str | None = None
        self._fc_name: str | None = None
        self._fc_arg_chunks: list[str] = []
        self._final_output: list[dict[str, Any]] = []
        self.done = False

    def start(self) -> list[str]:
        """Emit ``response.created`` and ``response.in_progress``."""
        return [
            create_response_created_event(self._response),
            create_response_in_progress_event(self._response),
        ]

    def process_chunk(self, chunk: Any) -> list[str]:
        """Process one yielded chunk and return the SSE events it produces."""
        if isinstance(chunk, Response):
            self.done = True
            return [create_response_completed_event(chunk)]

        if _is_custom_event(chunk):
            event = chunk.to_event_dict()
            event_name = event.get("type", "response.event")
            return [format_named_sse_event(event_name, event)]

        reasoning_text = _extract_reasoning_text(chunk)
        if reasoning_text is not None:
            events: list[str] = []
            events.extend(self._finalize_text())
            events.extend(self._finalize_function_call())
            item_id, init_events = self._ensure_reasoning_item()
            events.extend(init_events)
            self._reasoning_parts.append(reasoning_text)
            events.append(create_reasoning_summary_text_delta_event(item_id, self._output_index, 0, reasoning_text))
            return events

        fc_deltas = self._extract_function_call_deltas(chunk)
        if fc_deltas is not None:
            events: list[str] = []
            for call_id, name, arguments in fc_deltas:
                events.extend(self._handle_function_call_delta(call_id, name, arguments))
            return events

        text = self._text_from_chunk(chunk)
        if not text:
            return []

        events: list[str] = []
        events.extend(self._finalize_reasoning())
        events.extend(self._finalize_function_call())
        item_id, init_events = self._ensure_text_item()
        events.extend(init_events)
        self._text_parts.append(text)
        events.append(create_output_text_delta_event(item_id, self._output_index, self._content_index, text))
        return events

    def finalize(self) -> list[str]:
        """Close any open items and emit ``response.completed``."""
        events: list[str] = []
        events.extend(self._finalize_reasoning())
        events.extend(self._finalize_text())
        events.extend(self._finalize_function_call())
        completed = Response(
            id=self._resp_id,
            object="response",
            created_at=self._response.created_at,
            status="completed",
            model=self._model_name,
            output=self._final_output,
        )
        events.append(create_response_completed_event(completed))
        return events

    # -- private helpers ----------------------------------------------------

    def _text_from_chunk(self, chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        if hasattr(chunk, "content"):
            content = chunk.content
            if isinstance(content, str):
                return content
        mapped = self._chunk_mapper(chunk)
        return mapped if isinstance(mapped, str) else str(mapped)

    def _extract_function_call_deltas(self, chunk: Any) -> list[tuple[str, str | None, Any]] | None:
        """
        Normalize both chunk formats into ``[(call_id, name, arguments), ...]``.

        Recognises two shapes:
        * **Explicit attributes** -- ``function_call_id``, ``function_call_name``,
          ``function_call_arguments`` (one delta per chunk).
        * **tool_calls list** -- ``chunk.tool_calls``, each entry carrying
          ``.id``, ``.tool_name`` / ``.name``, ``.arguments`` (Haystack
          ``StreamingChunk`` / ``ToolCallDelta`` style).

        Returns ``None`` when the chunk is neither format.
        """
        if (
            hasattr(chunk, "function_call_id")
            and chunk.function_call_id is not None
            and hasattr(chunk, "function_call_arguments")
        ):
            name = chunk.function_call_name if hasattr(chunk, "function_call_name") else None
            return [(str(chunk.function_call_id), name, chunk.function_call_arguments)]

        tool_calls = getattr(chunk, "tool_calls", None)
        if tool_calls:
            deltas: list[tuple[str, str | None, Any]] = []
            for tc in tool_calls:
                call_id = getattr(tc, "id", None)
                if call_id is None:
                    call_id = self._fc_call_id or f"call_{uuid.uuid4().hex}"
                tc_name = getattr(tc, "tool_name", None) or getattr(tc, "name", None)
                deltas.append((str(call_id), tc_name, getattr(tc, "arguments", None)))
            return deltas

        return None

    def _handle_function_call_delta(self, call_id: str, name: str | None, arguments: Any) -> list[str]:
        events: list[str] = []
        events.extend(self._finalize_reasoning())
        events.extend(self._finalize_text())

        arguments_delta = "" if arguments is None else str(arguments)

        if self._fc_item_id is not None and self._fc_call_id != call_id:
            events.extend(self._finalize_function_call())

        fc_item_id, init_events = self._ensure_function_call_item(call_id, name)
        events.extend(init_events)
        if name is not None:
            self._fc_name = name

        if arguments_delta:
            self._fc_arg_chunks.append(arguments_delta)
            events.append(create_function_call_arguments_delta_event(fc_item_id, self._output_index, arguments_delta))
        return events

    def _ensure_reasoning_item(self) -> tuple[str, list[str]]:
        if self._reasoning_item_id is not None:
            return self._reasoning_item_id, []
        self._reasoning_item_id = f"rs_{uuid.uuid4().hex}"
        in_progress_item = {
            "id": self._reasoning_item_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
        }
        part_template = {"type": "summary_text", "text": ""}
        return self._reasoning_item_id, [
            create_output_item_added_event(self._output_index, in_progress_item),
            create_reasoning_summary_part_added_event(self._reasoning_item_id, self._output_index, 0, part_template),
        ]

    def _finalize_reasoning(self) -> list[str]:
        if self._reasoning_item_id is None:
            return []
        full_text = "".join(self._reasoning_parts)
        final_part = {"type": "summary_text", "text": full_text}
        final_item = {
            "id": self._reasoning_item_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [final_part],
        }
        self._final_output.append(final_item)
        events = [
            create_reasoning_summary_text_done_event(self._reasoning_item_id, self._output_index, 0, full_text),
            create_reasoning_summary_part_done_event(self._reasoning_item_id, self._output_index, 0, final_part),
            create_output_item_done_event(self._output_index, final_item),
        ]
        self._reasoning_item_id = None
        self._reasoning_parts = []
        self._output_index += 1
        return events

    def _ensure_text_item(self) -> tuple[str, list[str]]:
        if self._text_item_id is not None:
            return self._text_item_id, []
        self._text_item_id = f"msg_{uuid.uuid4().hex}"
        in_progress_item = {
            "id": self._text_item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        part_template = {"type": "output_text", "text": "", "annotations": []}
        return self._text_item_id, [
            create_output_item_added_event(self._output_index, in_progress_item),
            create_content_part_added_event(self._text_item_id, self._output_index, self._content_index, part_template),
        ]

    def _finalize_text(self) -> list[str]:
        if self._text_item_id is None:
            return []
        full_text = "".join(self._text_parts)
        final_part = {"type": "output_text", "text": full_text, "annotations": []}
        final_item = {
            "id": self._text_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [final_part],
        }
        self._final_output.append(final_item)
        events = [
            create_output_text_done_event(self._text_item_id, self._output_index, self._content_index, full_text),
            create_content_part_done_event(self._text_item_id, self._output_index, self._content_index, final_part),
            create_output_item_done_event(self._output_index, final_item),
        ]
        self._text_item_id = None
        self._text_parts = []
        self._output_index += 1
        return events

    def _ensure_function_call_item(self, call_id: str, name: str | None) -> tuple[str, list[str]]:
        if self._fc_item_id is not None:
            return self._fc_item_id, []
        self._fc_item_id = f"fc_{uuid.uuid4().hex}"
        self._fc_call_id = call_id
        self._fc_name = name or "function"
        self._fc_arg_chunks = []
        in_progress_item = {
            "id": self._fc_item_id,
            "type": "function_call",
            "status": "in_progress",
            "call_id": self._fc_call_id,
            "name": self._fc_name,
            "arguments": "",
        }
        return self._fc_item_id, [create_output_item_added_event(self._output_index, in_progress_item)]

    def _finalize_function_call(self) -> list[str]:
        if self._fc_item_id is None:
            return []
        full_arguments = "".join(self._fc_arg_chunks)
        final_item = {
            "id": self._fc_item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": self._fc_call_id,
            "name": self._fc_name or "function",
            "arguments": full_arguments,
        }
        self._final_output.append(final_item)
        events = [
            create_function_call_arguments_done_event(self._fc_item_id, self._output_index, full_arguments),
            create_output_item_done_event(self._output_index, final_item),
        ]
        self._fc_item_id = None
        self._fc_call_id = None
        self._fc_name = None
        self._fc_arg_chunks = []
        self._output_index += 1
        return events


# ---------------------------------------------------------------------------
# Public streaming wrappers
# ---------------------------------------------------------------------------


def create_responses_streaming_response(
    result: Generator[Any, None, None],
    resp_id: str,
    model_name: str,
    chunk_mapper: ChunkMapper = default_chunk_mapper,
) -> StreamingResponse:
    """Wrap a sync generator and emit OpenAI Responses API named events."""

    def stream_events() -> Generator[str, None, None]:
        state = _ResponseStreamState(resp_id, model_name, chunk_mapper)
        yield from state.start()
        for chunk in result:
            yield from state.process_chunk(chunk)
            if state.done:
                return
        yield from state.finalize()

    return StreamingResponse(stream_events(), media_type="text/event-stream")


def create_async_responses_streaming_response(
    result: AsyncGenerator[Any, None],
    resp_id: str,
    model_name: str,
    chunk_mapper: ChunkMapper = default_chunk_mapper,
) -> StreamingResponse:
    """Wrap an async generator and emit OpenAI Responses API named events."""

    async def stream_events_async() -> AsyncGenerator[str, None]:
        state = _ResponseStreamState(resp_id, model_name, chunk_mapper)
        for event in state.start():
            yield event
        async for chunk in result:
            for event in state.process_chunk(chunk):
                yield event
            if state.done:
                return
        for event in state.finalize():
            yield event

    return StreamingResponse(stream_events_async(), media_type="text/event-stream")


__all__ = [
    "create_async_responses_streaming_response",
    "create_function_call_arguments_delta_event",
    "create_function_call_arguments_done_event",
    "create_output_item_added_event",
    "create_output_item_done_event",
    "create_output_text_delta_event",
    "create_output_text_done_event",
    "create_reasoning_summary_part_added_event",
    "create_reasoning_summary_part_done_event",
    "create_reasoning_summary_text_delta_event",
    "create_reasoning_summary_text_done_event",
    "create_response_completed_event",
    "create_response_created_event",
    "create_response_in_progress_event",
    "create_responses_streaming_response",
    "format_named_sse_event",
    "response_from_text",
]
