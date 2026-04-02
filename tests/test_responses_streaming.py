import asyncio
import json
from collections.abc import AsyncGenerator

import pytest

from fastapi_openai_compat._shared import default_chunk_mapper
from fastapi_openai_compat.responses.models import Response
from fastapi_openai_compat.responses.streaming import (
    create_async_responses_streaming_response,
    create_output_text_delta_event,
    create_response_completed_event,
    create_response_created_event,
    create_responses_streaming_response,
    format_named_sse_event,
    response_from_text,
)


async def _collect_events(streaming_response):
    return [chunk async for chunk in streaming_response.body_iterator if chunk.strip()]


def _parse_sse_event(raw: str) -> tuple[str, dict]:
    lines = raw.strip().split("\n")
    event_name = lines[0].removeprefix("event: ")
    data = json.loads(lines[1].removeprefix("data: "))
    return event_name, data


@pytest.mark.unit
class TestResponseEventHelpers:
    def test_response_from_text(self):
        resp = response_from_text("Hello world", "resp_123", "my-model")
        assert resp.id == "resp_123"
        assert resp.model == "my-model"
        assert resp.object == "response"
        assert resp.status == "completed"
        assert len(resp.output) == 1
        msg = resp.output[0]
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "output_text"
        assert msg["content"][0]["text"] == "Hello world"

    def test_format_named_sse_event(self):
        result = format_named_sse_event("response.created", {"type": "response.created", "foo": "bar"})
        assert result.startswith("event: response.created\n")
        assert "\ndata: " in result
        assert result.endswith("\n\n")
        payload = json.loads(result.split("\n")[1].removeprefix("data: "))
        assert payload == {"type": "response.created", "foo": "bar"}

    def test_create_response_created_event(self):
        resp = Response(id="resp_1", created_at=1000, model="m", output=[])
        event = create_response_created_event(resp)
        assert event.startswith("event: response.created\n")
        data_line = event.split("\n")[1]
        payload = json.loads(data_line.removeprefix("data: "))
        assert payload["type"] == "response.created"
        assert payload["response"]["id"] == "resp_1"

    def test_create_output_text_delta_event(self):
        event = create_output_text_delta_event("msg_1", 0, 0, "Hello")
        assert "event: response.output_text.delta" in event
        data_line = event.split("\n")[1]
        payload = json.loads(data_line.removeprefix("data: "))
        assert payload["delta"] == "Hello"
        assert payload["item_id"] == "msg_1"
        assert payload["output_index"] == 0
        assert payload["content_index"] == 0

    def test_create_response_completed_event(self):
        resp = Response(id="resp_1", created_at=1000, model="m", output=[])
        event = create_response_completed_event(resp)
        assert "event: response.completed" in event
        data_line = event.split("\n")[1]
        payload = json.loads(data_line.removeprefix("data: "))
        assert payload["type"] == "response.completed"
        assert payload["response"]["id"] == "resp_1"


@pytest.mark.unit
class TestResponseStreaming:
    def test_sync_streaming_text_chunks(self):
        def gen():
            yield "Hello "
            yield "world!"

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))

        parsed_events = [_parse_sse_event(e) for e in events]
        event_names = [name for name, _ in parsed_events]
        assert "response.created" in event_names
        assert "response.output_text.delta" in event_names
        assert "response.output_text.done" in event_names
        assert "response.completed" in event_names

        deltas = [data for name, data in parsed_events if name == "response.output_text.delta"]
        delta_texts = [d["delta"] for d in deltas]
        assert "Hello " in delta_texts
        assert "world!" in delta_texts

        done_events = [data for name, data in parsed_events if name == "response.output_text.done"]
        assert done_events[0]["text"] == "Hello world!"

    def test_sync_streaming_response_object(self):
        full_resp = Response(id="resp_1", created_at=1000, model="m", output=[])

        def gen():
            yield full_resp

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        last_name, _ = _parse_sse_event(events[-1])
        assert last_name == "response.completed"

    def test_async_streaming_text(self):
        async def _run() -> list[str]:
            async def gen() -> AsyncGenerator[str, None]:
                yield "Hi "
                yield "there"

            resp = create_async_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
            return await _collect_events(resp)

        events = asyncio.run(_run())

        event_names = [_parse_sse_event(e)[0] for e in events]
        assert "response.created" in event_names
        assert "response.output_text.delta" in event_names
        assert "response.completed" in event_names

    def test_sync_streaming_function_call_chunk(self):
        class FunctionCallChunk:
            def __init__(self, *, call_id: str, name: str | None, arguments: str):
                self.function_call_id = call_id
                self.function_call_name = name
                self.function_call_arguments = arguments

        def gen():
            yield FunctionCallChunk(call_id="call_123", name="get_weather", arguments='{"location":')
            yield FunctionCallChunk(call_id="call_123", name=None, arguments='"Boston"}')

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        event_names = [name for name, _ in parsed_events]
        assert "response.output_item.added" in event_names
        assert "response.function_call_arguments.delta" in event_names
        assert "response.function_call_arguments.done" in event_names
        assert "response.output_item.done" in event_names

        deltas = [data for name, data in parsed_events if name == "response.function_call_arguments.delta"]
        assert deltas[0]["delta"] == '{"location":'
        assert deltas[1]["delta"] == '"Boston"}'

        done_args = [data for name, data in parsed_events if name == "response.function_call_arguments.done"]
        assert done_args[0]["arguments"] == '{"location":"Boston"}'

        output_items_done = [data for name, data in parsed_events if name == "response.output_item.done"]
        item = output_items_done[0]["item"]
        assert item["type"] == "function_call"
        assert item["call_id"] == "call_123"
        assert item["name"] == "get_weather"
        assert item["arguments"] == '{"location":"Boston"}'

    def test_sync_streaming_function_call_none_arguments_not_stringified(self):
        class FunctionCallChunk:
            def __init__(self, *, call_id: str, name: str | None, arguments: str | None):
                self.function_call_id = call_id
                self.function_call_name = name
                self.function_call_arguments = arguments

        def gen():
            yield FunctionCallChunk(call_id="call_123", name="get_weather", arguments=None)
            yield FunctionCallChunk(call_id="call_123", name=None, arguments='{"city":"Rome"}')

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        deltas = [data["delta"] for name, data in parsed_events if name == "response.function_call_arguments.delta"]
        assert "None" not in deltas
        done_args = [
            data["arguments"] for name, data in parsed_events if name == "response.function_call_arguments.done"
        ]
        assert done_args[0] == '{"city":"Rome"}'

    def test_sync_streaming_distinct_function_calls_are_emitted_as_separate_items(self):
        class FunctionCallChunk:
            def __init__(self, *, call_id: str, name: str | None, arguments: str):
                self.function_call_id = call_id
                self.function_call_name = name
                self.function_call_arguments = arguments

        def gen():
            yield FunctionCallChunk(call_id="call_1", name="lookup", arguments='{"q":')
            yield FunctionCallChunk(call_id="call_1", name=None, arguments='"books"}')
            yield FunctionCallChunk(call_id="call_2", name="weather", arguments='{"city":"Rome"}')

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]
        completed_event = next(data for name, data in parsed_events if name == "response.completed")
        output = completed_event["response"]["output"]

        assert len(output) == 2
        assert output[0]["type"] == "function_call"
        assert output[0]["call_id"] == "call_1"
        assert output[0]["name"] == "lookup"
        assert output[0]["arguments"] == '{"q":"books"}'
        assert output[1]["type"] == "function_call"
        assert output[1]["call_id"] == "call_2"
        assert output[1]["name"] == "weather"
        assert output[1]["arguments"] == '{"city":"Rome"}'

    def test_sync_streaming_mixed_text_and_function_call_keeps_both_items(self):
        class FunctionCallChunk:
            def __init__(self, *, call_id: str, name: str | None, arguments: str):
                self.function_call_id = call_id
                self.function_call_name = name
                self.function_call_arguments = arguments

        def gen():
            yield "Thinking..."
            yield FunctionCallChunk(call_id="call_123", name="get_weather", arguments='{"city":"Rome"}')

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        completed_event = next(data for name, data in parsed_events if name == "response.completed")
        output = completed_event["response"]["output"]
        assert len(output) == 2
        assert output[0]["type"] == "message"
        assert output[1]["type"] == "function_call"

    def test_sync_streaming_empty_generator_completes_with_empty_output(self):
        def gen():
            if False:
                yield "never"

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        event_names = [name for name, _ in parsed_events]
        assert "response.created" in event_names
        assert "response.completed" in event_names
        completed_event = next(data for name, data in parsed_events if name == "response.completed")
        assert completed_event["response"]["output"] == []

    def test_sync_streaming_custom_event_is_emitted(self):
        class CustomEvent:
            def to_event_dict(self) -> dict:
                return {"type": "response.status", "data": {"phase": "thinking"}}

        def gen():
            yield CustomEvent()
            yield "Done"

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        event_names = [name for name, _ in parsed_events]
        assert "response.status" in event_names
        assert "response.output_text.delta" in event_names

    def test_sync_streaming_mapper_fallback_and_empty_text_skipped(self):
        class ChunkWithNonStringContent:
            def __init__(self, payload):
                self.content = payload

        chunks = [
            ChunkWithNonStringContent({"ignored": True}),
            ChunkWithNonStringContent({"ignored": True}),
            "tail",
        ]

        def gen():
            yield from chunks

        mapped_values = iter([123, ""])

        def mapper(_chunk) -> str | int:
            return next(mapped_values)

        resp = create_responses_streaming_response(gen(), "resp_1", "m", mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        deltas = [data["delta"] for name, data in parsed_events if name == "response.output_text.delta"]
        assert deltas == ["123", "tail"]

    def test_async_streaming_response_object_passthrough(self):
        async def _run() -> list[str]:
            full_resp = Response(id="resp_passthrough", created_at=1234, model="m", output=[])

            async def gen() -> AsyncGenerator[Response, None]:
                yield full_resp

            resp = create_async_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
            return await _collect_events(resp)

        events = asyncio.run(_run())
        parsed_events = [_parse_sse_event(e) for e in events]
        completed_data = next(data for name, data in parsed_events if name == "response.completed")
        assert completed_data["response"]["id"] == "resp_passthrough"

    def test_async_streaming_custom_event_and_empty_text_skipped(self):
        class CustomEvent:
            def to_event_dict(self) -> dict:
                return {"type": "response.note", "data": {"x": 1}}

        async def _run() -> list[str]:
            async def gen() -> AsyncGenerator[CustomEvent | str, None]:
                yield CustomEvent()
                yield ""
                yield "hello"

            resp = create_async_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
            return await _collect_events(resp)

        events = asyncio.run(_run())
        parsed_events = [_parse_sse_event(e) for e in events]
        event_names = [name for name, _ in parsed_events]
        assert "response.note" in event_names
        deltas = [data["delta"] for name, data in parsed_events if name == "response.output_text.delta"]
        assert deltas == ["hello"]

    def test_async_streaming_function_call_chunk(self):
        class FunctionCallChunk:
            def __init__(self, *, call_id: str, name: str | None, arguments: str):
                self.function_call_id = call_id
                self.function_call_name = name
                self.function_call_arguments = arguments

        async def _run() -> list[str]:
            async def gen() -> AsyncGenerator[FunctionCallChunk, None]:
                yield FunctionCallChunk(call_id="call_1", name="lookup", arguments='{"q":')
                yield FunctionCallChunk(call_id="call_1", name=None, arguments='"books"}')

            resp = create_async_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
            return await _collect_events(resp)

        events = asyncio.run(_run())
        parsed_events = [_parse_sse_event(e) for e in events]
        event_names = [name for name, _ in parsed_events]
        assert "response.function_call_arguments.delta" in event_names
        assert "response.function_call_arguments.done" in event_names

        final_arguments = [
            data["arguments"] for name, data in parsed_events if name == "response.function_call_arguments.done"
        ]
        assert final_arguments == ['{"q":"books"}']

    def test_async_streaming_function_call_then_text_keeps_both_items(self):
        class FunctionCallChunk:
            def __init__(self, *, call_id: str, name: str | None, arguments: str):
                self.function_call_id = call_id
                self.function_call_name = name
                self.function_call_arguments = arguments

        async def _run() -> list[str]:
            async def gen() -> AsyncGenerator[FunctionCallChunk | str, None]:
                yield FunctionCallChunk(call_id="call_123", name="tool", arguments='{"city":"Rome"}')
                yield "after-tool"

            resp = create_async_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
            return await _collect_events(resp)

        events = asyncio.run(_run())
        parsed_events = [_parse_sse_event(e) for e in events]
        completed_event = next(data for name, data in parsed_events if name == "response.completed")
        output = completed_event["response"]["output"]
        assert len(output) == 2
        assert output[0]["type"] == "function_call"
        assert output[1]["type"] == "message"

    def test_async_streaming_distinct_function_calls_are_emitted_as_separate_items(self):
        class FunctionCallChunk:
            def __init__(self, *, call_id: str, name: str | None, arguments: str):
                self.function_call_id = call_id
                self.function_call_name = name
                self.function_call_arguments = arguments

        async def _run() -> list[str]:
            async def gen() -> AsyncGenerator[FunctionCallChunk, None]:
                yield FunctionCallChunk(call_id="call_1", name="lookup", arguments='{"q":')
                yield FunctionCallChunk(call_id="call_1", name=None, arguments='"books"}')
                yield FunctionCallChunk(call_id="call_2", name="weather", arguments='{"city":"Rome"}')

            resp = create_async_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
            return await _collect_events(resp)

        events = asyncio.run(_run())
        parsed_events = [_parse_sse_event(e) for e in events]
        completed_event = next(data for name, data in parsed_events if name == "response.completed")
        output = completed_event["response"]["output"]

        assert len(output) == 2
        assert output[0]["type"] == "function_call"
        assert output[0]["call_id"] == "call_1"
        assert output[0]["name"] == "lookup"
        assert output[0]["arguments"] == '{"q":"books"}'
        assert output[1]["type"] == "function_call"
        assert output[1]["call_id"] == "call_2"
        assert output[1]["name"] == "weather"
        assert output[1]["arguments"] == '{"city":"Rome"}'


@pytest.mark.unit
class TestResponseStreamingToolCalls:
    """Tests for tool_calls-style chunks (Haystack StreamingChunk format) in Responses API."""

    def test_sync_streaming_tool_calls_chunk(self):
        from dataclasses import dataclass, field

        @dataclass
        class FakeToolCallDelta:
            index: int
            tool_name: str = ""
            arguments: str = ""
            id: str | None = None

        @dataclass
        class FakeChunk:
            content: str = ""
            tool_calls: list | None = field(default=None)

        def gen():
            yield FakeChunk(
                tool_calls=[FakeToolCallDelta(index=0, id="call_1", tool_name="get_weather", arguments='{"city":')],
            )
            yield FakeChunk(
                tool_calls=[FakeToolCallDelta(index=0, arguments='"Paris"}')],
            )

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        event_names = [name for name, _ in parsed_events]
        assert "response.output_item.added" in event_names
        assert "response.function_call_arguments.delta" in event_names
        assert "response.function_call_arguments.done" in event_names
        assert "response.output_item.done" in event_names

        deltas = [data for name, data in parsed_events if name == "response.function_call_arguments.delta"]
        assert deltas[0]["delta"] == '{"city":'
        assert deltas[1]["delta"] == '"Paris"}'

        done_args = [data for name, data in parsed_events if name == "response.function_call_arguments.done"]
        assert done_args[0]["arguments"] == '{"city":"Paris"}'

        output_items_done = [data for name, data in parsed_events if name == "response.output_item.done"]
        item = output_items_done[0]["item"]
        assert item["type"] == "function_call"
        assert item["call_id"] == "call_1"
        assert item["name"] == "get_weather"

    def test_sync_streaming_tool_calls_name_via_name_attr(self):
        from dataclasses import dataclass, field

        @dataclass
        class AltToolCall:
            index: int
            name: str
            arguments: str
            id: str | None = None

        @dataclass
        class AltChunk:
            content: str = ""
            tool_calls: list | None = field(default=None)

        def gen():
            yield AltChunk(
                tool_calls=[AltToolCall(index=0, id="call_1", name="search", arguments='{"q":"test"}')],
            )

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        output_items_done = [data for name, data in parsed_events if name == "response.output_item.done"]
        item = output_items_done[0]["item"]
        assert item["name"] == "search"
        assert item["arguments"] == '{"q":"test"}'

    def test_sync_streaming_mixed_text_then_tool_calls(self):
        from dataclasses import dataclass, field

        @dataclass
        class FakeToolCallDelta:
            index: int
            tool_name: str
            arguments: str
            id: str | None = None

        @dataclass
        class FakeChunk:
            content: str = ""
            tool_calls: list | None = field(default=None)

        def gen():
            yield "Thinking..."
            yield FakeChunk(
                tool_calls=[
                    FakeToolCallDelta(index=0, id="call_1", tool_name="get_weather", arguments='{"city":"Rome"}')
                ],
            )

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        completed_event = next(data for name, data in parsed_events if name == "response.completed")
        output = completed_event["response"]["output"]
        assert len(output) == 2
        assert output[0]["type"] == "message"
        assert output[1]["type"] == "function_call"

    def test_sync_streaming_distinct_tool_calls_are_separate_items(self):
        from dataclasses import dataclass, field

        @dataclass
        class FakeToolCallDelta:
            index: int
            tool_name: str = ""
            arguments: str = ""
            id: str | None = None

        @dataclass
        class FakeChunk:
            content: str = ""
            tool_calls: list | None = field(default=None)

        def gen():
            yield FakeChunk(
                tool_calls=[FakeToolCallDelta(index=0, id="call_1", tool_name="lookup", arguments='{"q":')],
            )
            yield FakeChunk(
                tool_calls=[FakeToolCallDelta(index=0, arguments='"books"}')],
            )
            yield FakeChunk(
                tool_calls=[FakeToolCallDelta(index=0, id="call_2", tool_name="weather", arguments='{"city":"Rome"}')],
            )

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]
        completed_event = next(data for name, data in parsed_events if name == "response.completed")
        output = completed_event["response"]["output"]

        assert len(output) == 2
        assert output[0]["type"] == "function_call"
        assert output[0]["call_id"] == "call_1"
        assert output[0]["name"] == "lookup"
        assert output[0]["arguments"] == '{"q":"books"}'
        assert output[1]["type"] == "function_call"
        assert output[1]["call_id"] == "call_2"
        assert output[1]["name"] == "weather"
        assert output[1]["arguments"] == '{"city":"Rome"}'

    def test_async_streaming_tool_calls_chunk(self):
        from dataclasses import dataclass, field

        @dataclass
        class FakeToolCallDelta:
            index: int
            tool_name: str
            arguments: str
            id: str | None = None

        @dataclass
        class FakeChunk:
            content: str = ""
            tool_calls: list | None = field(default=None)

        async def _run() -> list[str]:
            async def gen() -> AsyncGenerator[FakeChunk, None]:
                yield FakeChunk(
                    tool_calls=[FakeToolCallDelta(index=0, id="call_1", tool_name="search", arguments='{"q":"test"}')],
                )

            resp = create_async_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
            return await _collect_events(resp)

        events = asyncio.run(_run())
        parsed_events = [_parse_sse_event(e) for e in events]
        event_names = [name for name, _ in parsed_events]
        assert "response.function_call_arguments.delta" in event_names
        assert "response.function_call_arguments.done" in event_names

        final_arguments = [
            data["arguments"] for name, data in parsed_events if name == "response.function_call_arguments.done"
        ]
        assert final_arguments == ['{"q":"test"}']

    def test_haystack_streaming_chunk_tool_calls(self):
        """Real Haystack StreamingChunk + ToolCallDelta produces function_call events."""
        from haystack.dataclasses import StreamingChunk
        from haystack.dataclasses.streaming_chunk import ToolCallDelta

        def gen():
            yield StreamingChunk(
                content="",
                tool_calls=[
                    ToolCallDelta(index=0, id="call_1", tool_name="get_weather", arguments=""),
                ],
                index=0,
            )
            yield StreamingChunk(
                content="",
                tool_calls=[
                    ToolCallDelta(index=0, arguments='{"city": "Paris"}'),
                ],
                index=0,
            )

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        event_names = [name for name, _ in parsed_events]
        assert "response.function_call_arguments.delta" in event_names
        assert "response.function_call_arguments.done" in event_names
        assert "response.output_item.done" in event_names

        done_item = next(data for name, data in parsed_events if name == "response.output_item.done")
        assert done_item["item"]["type"] == "function_call"
        assert done_item["item"]["call_id"] == "call_1"
        assert done_item["item"]["name"] == "get_weather"
        assert done_item["item"]["arguments"] == '{"city": "Paris"}'


@pytest.mark.unit
class TestResponseStreamingReasoning:
    def test_sync_streaming_reasoning_then_text(self):
        from dataclasses import dataclass

        @dataclass
        class FakeReasoning:
            reasoning_text: str

        @dataclass
        class ReasoningChunk:
            content: str = ""
            reasoning: FakeReasoning | None = None

        def gen():
            yield ReasoningChunk(reasoning=FakeReasoning("step 1"))
            yield ReasoningChunk(reasoning=FakeReasoning("step 2"))
            yield "answer"

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]
        event_names = [name for name, _ in parsed_events]

        assert "response.output_item.added" in event_names
        assert "response.reasoning_summary_part.added" in event_names
        assert "response.reasoning_summary_text.delta" in event_names
        assert "response.reasoning_summary_text.done" in event_names
        assert "response.reasoning_summary_part.done" in event_names
        assert "response.output_text.delta" in event_names
        assert "response.completed" in event_names

        deltas = [data["delta"] for name, data in parsed_events if name == "response.reasoning_summary_text.delta"]
        assert deltas == ["step 1", "step 2"]

        done_text = next(data for name, data in parsed_events if name == "response.reasoning_summary_text.done")
        assert done_text["text"] == "step 1step 2"

        completed = next(data for name, data in parsed_events if name == "response.completed")
        output = completed["response"]["output"]
        assert len(output) == 2
        assert output[0]["type"] == "reasoning"
        assert output[0]["summary"] == [{"type": "summary_text", "text": "step 1step 2"}]
        assert output[1]["type"] == "message"

    def test_async_streaming_reasoning_then_text(self):
        from dataclasses import dataclass

        @dataclass
        class FakeReasoning:
            reasoning_text: str

        @dataclass
        class ReasoningChunk:
            content: str = ""
            reasoning: FakeReasoning | None = None

        async def _run():
            async def gen() -> AsyncGenerator:
                yield ReasoningChunk(reasoning=FakeReasoning("thinking"))
                yield "done"

            resp = create_async_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
            return await _collect_events(resp)

        events = asyncio.run(_run())
        parsed_events = [_parse_sse_event(e) for e in events]

        completed = next(data for name, data in parsed_events if name == "response.completed")
        output = completed["response"]["output"]
        assert len(output) == 2
        assert output[0]["type"] == "reasoning"
        assert output[0]["summary"][0]["text"] == "thinking"
        assert output[1]["type"] == "message"

    def test_sync_streaming_only_reasoning(self):
        from dataclasses import dataclass

        @dataclass
        class FakeReasoning:
            reasoning_text: str

        @dataclass
        class ReasoningChunk:
            content: str = ""
            reasoning: FakeReasoning | None = None

        def gen():
            yield ReasoningChunk(reasoning=FakeReasoning("just thinking"))

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        completed = next(data for name, data in parsed_events if name == "response.completed")
        output = completed["response"]["output"]
        assert len(output) == 1
        assert output[0]["type"] == "reasoning"
        assert output[0]["summary"][0]["text"] == "just thinking"

    def test_haystack_streaming_chunk_reasoning(self):
        from haystack.dataclasses import ReasoningContent, StreamingChunk

        def gen():
            yield StreamingChunk(
                content="",
                reasoning=ReasoningContent(reasoning_text="analyzing the problem"),
                index=0,
            )
            yield StreamingChunk(content="here is the answer", index=0)

        resp = create_responses_streaming_response(gen(), "resp_1", "m", default_chunk_mapper)
        events = asyncio.run(_collect_events(resp))
        parsed_events = [_parse_sse_event(e) for e in events]

        completed = next(data for name, data in parsed_events if name == "response.completed")
        output = completed["response"]["output"]
        assert len(output) == 2
        assert output[0]["type"] == "reasoning"
        assert output[0]["summary"][0]["text"] == "analyzing the problem"
        assert output[1]["type"] == "message"
        assert output[1]["content"][0]["text"] == "here is the answer"
