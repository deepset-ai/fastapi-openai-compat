import json
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field

import pytest
from haystack.dataclasses import StreamingChunk
from haystack.dataclasses.streaming_chunk import ToolCallDelta

from fastapi_openai_compat._shared import _is_custom_event
from fastapi_openai_compat.chat_completions.streaming import (
    chat_completion_response,
    create_async_streaming_response,
    create_sse_data_msg,
    create_sync_streaming_response,
    event_to_sse_msg,
)


@pytest.mark.unit
class TestEventToSseMsg:
    def test_formats_event_as_sse(self):
        result = event_to_sse_msg({"type": "status", "data": {"done": True}})
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        payload = json.loads(result[len("data: ") :])
        assert "event" in payload
        assert payload["event"]["type"] == "status"


@pytest.mark.unit
class TestCreateSseDataMsg:
    def test_creates_chunk_msg(self):
        msg = create_sse_data_msg(resp_id="r-1", model_name="m", chunk_content="hello")
        assert msg.startswith("data: ")
        payload = json.loads(msg[len("data: ") :])
        assert payload["object"] == "chat.completion.chunk"
        assert payload["model"] == "m"
        assert payload["choices"][0]["delta"]["content"] == "hello"
        assert payload["choices"][0]["finish_reason"] is None

    def test_creates_stop_msg(self):
        msg = create_sse_data_msg(resp_id="r-2", model_name="m", finish_reason="stop")
        payload = json.loads(msg[len("data: ") :])
        assert payload["choices"][0]["finish_reason"] == "stop"
        assert payload["choices"][0]["delta"]["content"] == ""

    def test_creates_custom_finish_reason(self):
        msg = create_sse_data_msg(resp_id="r-3", model_name="m", finish_reason="length")
        payload = json.loads(msg[len("data: ") :])
        assert payload["choices"][0]["finish_reason"] == "length"


@pytest.mark.unit
class TestChatCompletionResponse:
    def test_non_streaming_response(self):
        resp = chat_completion_response(result="Hello!", resp_id="id-1", model_name="test-model")
        assert resp.object == "chat.completion"
        assert resp.id == "id-1"
        assert resp.model == "test-model"
        assert resp.choices[0].message.content == "Hello!"
        assert resp.choices[0].finish_reason == "stop"


@pytest.mark.unit
class TestCreateSyncStreamingResponse:
    @pytest.mark.asyncio
    async def test_streams_streaming_chunks(self):
        def gen() -> Generator[StreamingChunk | str, None, None]:
            yield StreamingChunk(content="Hello ")
            yield StreamingChunk(content="world")

        response = create_sync_streaming_response(gen(), resp_id="r-1", model_name="m")
        assert response.media_type == "text/event-stream"

        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 3

        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["content"] == "Hello "

        last = json.loads(chunks[-1][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streams_plain_strings(self):
        def gen() -> Generator[str, None, None]:
            yield "one"
            yield "two"

        response = create_sync_streaming_response(gen(), resp_id="r-2", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 3
        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["content"] == "one"


@pytest.mark.unit
class TestCreateAsyncStreamingResponse:
    @pytest.mark.asyncio
    async def test_streams_streaming_chunks(self):
        async def gen() -> AsyncGenerator[StreamingChunk | str, None]:
            yield StreamingChunk(content="async ")
            yield StreamingChunk(content="world")

        response = create_async_streaming_response(gen(), resp_id="r-3", model_name="m")
        assert response.media_type == "text/event-stream"

        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 3

        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["content"] == "async "

        last = json.loads(chunks[-1][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streams_plain_strings(self):
        async def gen() -> AsyncGenerator[str, None]:
            yield "alpha"
            yield "beta"

        response = create_async_streaming_response(gen(), resp_id="r-4", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 3
        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["content"] == "alpha"


@pytest.mark.unit
class TestStreamingChunkToolCalls:
    @pytest.mark.asyncio
    async def test_sync_stream_with_tool_calls(self):
        def gen():
            yield StreamingChunk(
                content="",
                tool_calls=[
                    ToolCallDelta(index=0, id="call_1", tool_name="get_weather", arguments='{"city": "Paris"}')
                ],
                index=0,
            )
            yield StreamingChunk(content="", finish_reason="tool_calls")

        response = create_sync_streaming_response(gen(), resp_id="r-tc", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 2

        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
        assert first["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"
        assert first["choices"][0]["delta"]["content"] is None

        last = json.loads(chunks[1][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_async_stream_with_tool_calls(self):
        async def gen():
            yield StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=0, id="call_2", tool_name="search", arguments='{"q": "test"}')],
                index=0,
            )
            yield StreamingChunk(content="", finish_reason="tool_calls")

        response = create_async_streaming_response(gen(), resp_id="r-tc-async", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 2

        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_tool_call_with_finish_reason_on_same_chunk(self):
        def gen():
            yield StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=0, id="c1", tool_name="fn", arguments="{}")],
                index=0,
                finish_reason="tool_calls",
            )

        response = create_sync_streaming_response(gen(), resp_id="r-combo", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 1

        parsed = json.loads(chunks[0][len("data: ") :])
        assert parsed["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "fn"
        assert parsed["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.unit
class TestStreamingChunkFinishReason:
    @pytest.mark.asyncio
    async def test_finish_reason_propagated(self):
        def gen():
            yield StreamingChunk(content="hello ")
            yield StreamingChunk(content="", finish_reason="length")

        response = create_sync_streaming_response(gen(), resp_id="r-fr", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 2

        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["content"] == "hello "

        last = json.loads(chunks[1][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "length"

    @pytest.mark.asyncio
    async def test_explicit_stop_prevents_auto_stop(self):
        def gen():
            yield StreamingChunk(content="done")
            yield StreamingChunk(content="", finish_reason="stop")

        response = create_sync_streaming_response(gen(), resp_id="r-ns", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 2

        last = json.loads(chunks[1][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_auto_stop_when_no_finish_reason(self):
        def gen():
            yield StreamingChunk(content="hello")

        response = create_sync_streaming_response(gen(), resp_id="r-as", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 2

        last = json.loads(chunks[1][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "stop"


@pytest.mark.unit
class TestDuckTypingToolCalls:
    @pytest.mark.asyncio
    async def test_duck_typed_tool_calls(self):
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
            finish_reason: str | None = None

        def gen():
            yield FakeChunk(
                tool_calls=[FakeToolCallDelta(index=0, tool_name="my_tool", arguments="{}", id="c1")],
            )
            yield FakeChunk(finish_reason="tool_calls")

        response = create_sync_streaming_response(gen(), resp_id="r-duck", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 2

        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "my_tool"
        assert first["choices"][0]["delta"]["tool_calls"][0]["id"] == "c1"

    @pytest.mark.asyncio
    async def test_duck_typed_name_fallback(self):
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
            finish_reason: str | None = None

        def gen():
            yield AltChunk(
                tool_calls=[AltToolCall(index=0, name="alt_tool", arguments='{"x": 1}', id="c2")],
            )
            yield AltChunk(finish_reason="stop")

        response = create_sync_streaming_response(gen(), resp_id="r-alt", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]
        assert len(chunks) == 2

        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "alt_tool"


class FakeEvent:
    def __init__(self, description: str, done: bool = False):
        self.description = description
        self.done = done

    def to_event_dict(self) -> dict:
        return {"type": "status", "data": {"description": self.description, "done": self.done}}


@pytest.mark.unit
class TestIsCustomEvent:
    def test_recognizes_object_with_to_event_dict(self):
        assert _is_custom_event(FakeEvent("hi")) is True

    def test_rejects_plain_string(self):
        assert _is_custom_event("hello") is False

    def test_rejects_none(self):
        assert _is_custom_event(None) is False

    def test_rejects_non_callable_attribute(self):
        @dataclass
        class BadEvent:
            to_event_dict: str = "not callable"

        assert _is_custom_event(BadEvent()) is False


@pytest.mark.unit
class TestCustomEventSyncStreaming:
    @pytest.mark.asyncio
    async def test_custom_events_interspersed_with_text(self):
        def gen() -> Generator[str | FakeEvent, None, None]:
            yield FakeEvent("Processing...")
            yield "Hello "
            yield "world"
            yield FakeEvent("Done", done=True)

        response = create_sync_streaming_response(gen(), resp_id="r-evt", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]

        assert len(chunks) == 5

        evt1 = json.loads(chunks[0][len("data: ") :])
        assert evt1["event"]["type"] == "status"
        assert evt1["event"]["data"]["description"] == "Processing..."
        assert evt1["event"]["data"]["done"] is False

        text1 = json.loads(chunks[1][len("data: ") :])
        assert text1["choices"][0]["delta"]["content"] == "Hello "

        text2 = json.loads(chunks[2][len("data: ") :])
        assert text2["choices"][0]["delta"]["content"] == "world"

        evt2 = json.loads(chunks[3][len("data: ") :])
        assert evt2["event"]["type"] == "status"
        assert evt2["event"]["data"]["done"] is True

        stop = json.loads(chunks[4][len("data: ") :])
        assert stop["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_only_custom_events_no_auto_stop(self):
        """When only custom events are yielded, no finish_reason="stop" is appended."""

        def gen() -> Generator[FakeEvent, None, None]:
            yield FakeEvent("start")
            yield FakeEvent("end", done=True)

        response = create_sync_streaming_response(gen(), resp_id="r-only-evt", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]

        assert len(chunks) == 2
        for chunk in chunks:
            payload = json.loads(chunk[len("data: ") :])
            assert "event" in payload

    @pytest.mark.asyncio
    async def test_custom_events_do_not_affect_explicit_finish(self):
        def gen() -> Generator[str | FakeEvent, None, None]:
            yield FakeEvent("Working...")
            yield StreamingChunk(content="done")
            yield StreamingChunk(content="", finish_reason="length")
            yield FakeEvent("Finished", done=True)

        response = create_sync_streaming_response(gen(), resp_id="r-evt-fr", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]

        assert len(chunks) == 4

        last_completion = json.loads(chunks[2][len("data: ") :])
        assert last_completion["choices"][0]["finish_reason"] == "length"


@pytest.mark.unit
class TestCustomEventAsyncStreaming:
    @pytest.mark.asyncio
    async def test_custom_events_interspersed_with_text(self):
        async def gen() -> AsyncGenerator[str | FakeEvent, None]:
            yield FakeEvent("Processing...")
            yield "Hello "
            yield "world"
            yield FakeEvent("Done", done=True)

        response = create_async_streaming_response(gen(), resp_id="r-evt-async", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]

        assert len(chunks) == 5

        evt1 = json.loads(chunks[0][len("data: ") :])
        assert evt1["event"]["type"] == "status"
        assert evt1["event"]["data"]["description"] == "Processing..."

        text1 = json.loads(chunks[1][len("data: ") :])
        assert text1["choices"][0]["delta"]["content"] == "Hello "

        stop = json.loads(chunks[4][len("data: ") :])
        assert stop["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_only_custom_events_async(self):
        async def gen() -> AsyncGenerator[FakeEvent, None]:
            yield FakeEvent("start")

        response = create_async_streaming_response(gen(), resp_id="r-only-evt-a", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]

        assert len(chunks) == 1
        payload = json.loads(chunks[0][len("data: ") :])
        assert "event" in payload


@pytest.mark.unit
class TestReasoningStreaming:
    @pytest.mark.asyncio
    async def test_sync_stream_with_reasoning_then_text(self):
        @dataclass
        class ReasoningChunk:
            content: str = ""
            reasoning: object = None

        @dataclass
        class FakeReasoning:
            reasoning_text: str = ""

        def gen():
            yield ReasoningChunk(reasoning=FakeReasoning(reasoning_text="thinking..."))
            yield StreamingChunk(content="answer")

        response = create_sync_streaming_response(gen(), resp_id="r-reason", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]

        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["reasoning_content"] == "thinking..."
        assert first["choices"][0]["delta"]["content"] is None

        second = json.loads(chunks[1][len("data: ") :])
        assert second["choices"][0]["delta"]["content"] == "answer"

    @pytest.mark.asyncio
    async def test_async_stream_with_reasoning(self):
        @dataclass
        class ReasoningChunk:
            content: str = ""
            reasoning: object = None

        @dataclass
        class FakeReasoning:
            reasoning_text: str = ""

        async def gen():
            yield ReasoningChunk(reasoning=FakeReasoning(reasoning_text="step 1"))
            yield ReasoningChunk(reasoning=FakeReasoning(reasoning_text="step 2"))
            yield StreamingChunk(content="result")

        response = create_async_streaming_response(gen(), resp_id="r-reason-a", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]

        r1 = json.loads(chunks[0][len("data: ") :])
        assert r1["choices"][0]["delta"]["reasoning_content"] == "step 1"

        r2 = json.loads(chunks[1][len("data: ") :])
        assert r2["choices"][0]["delta"]["reasoning_content"] == "step 2"

        text = json.loads(chunks[2][len("data: ") :])
        assert text["choices"][0]["delta"]["content"] == "result"

        stop = json.loads(chunks[3][len("data: ") :])
        assert stop["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_haystack_streaming_chunk_with_reasoning(self):
        from haystack.dataclasses import ReasoningContent

        def gen():
            yield StreamingChunk(
                content="",
                reasoning=ReasoningContent(reasoning_text="let me think"),
                index=0,
            )
            yield StreamingChunk(content="done", index=0)

        response = create_sync_streaming_response(gen(), resp_id="r-hs-reason", model_name="m")
        chunks = [chunk async for chunk in response.body_iterator]

        first = json.loads(chunks[0][len("data: ") :])
        assert first["choices"][0]["delta"]["reasoning_content"] == "let me think"
        assert first["choices"][0]["delta"]["content"] is None

        second = json.loads(chunks[1][len("data: ") :])
        assert second["choices"][0]["delta"]["content"] == "done"
