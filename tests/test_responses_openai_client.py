import asyncio
import io
import time
from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport
from openai import AsyncOpenAI

from fastapi_openai_compat.files import create_files_router
from fastapi_openai_compat.responses.models import Response
from fastapi_openai_compat.responses.router import create_responses_router


@asynccontextmanager
async def _openai_client_for(
    run_response,
    *,
    include_files_api: bool = False,
):
    app = FastAPI()
    uploaded_files: dict[str, dict[str, Any]] = {}

    if include_files_api:

        def run_file_upload(filename: str | None, _content_type: str | None, content: bytes, purpose: str):
            file_id = f"file_{len(uploaded_files) + 1}"
            uploaded_files[file_id] = {
                "filename": filename,
                "purpose": purpose,
                "bytes": len(content),
            }
            return {
                "id": file_id,
                "object": "file",
                "bytes": len(content),
                "created_at": int(time.time()),
                "filename": filename,
                "purpose": purpose,
                "status": "processed",
            }

        app.include_router(create_files_router(run_file_upload=run_file_upload))

    router = create_responses_router(
        list_models=lambda: ["test-model"],
        run_response=run_response,
        include_models_endpoints=True,
    )
    app.include_router(router)

    transport = ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    client = AsyncOpenAI(api_key="test-key", base_url="http://testserver/v1", http_client=http_client)
    try:
        yield client, uploaded_files
    finally:
        await http_client.aclose()


@pytest.mark.integration
def test_openai_responses_create():
    def run_response(_model, _input_items, _body):
        return "Hello from SDK test!"

    async def _run() -> None:
        async with _openai_client_for(run_response) as (client, _):
            response = await client.responses.create(model="test-model", input="Hello")
            assert response.id is not None
            assert response.object == "response"
            assert len(response.output) == 1
            assert response.output[0].type == "message"
            assert response.output[0].content[0].type == "output_text"
            assert response.output[0].content[0].text == "Hello from SDK test!"

    asyncio.run(_run())


@pytest.mark.integration
def test_openai_responses_streaming():
    def run_response(_model, _input_items, body):
        if body.get("stream"):

            def _gen() -> Generator[str, None, None]:
                yield "Hello "
                yield "from "
                yield "SDK"

            return _gen()
        return "not-streamed"

    async def _run() -> None:
        async with _openai_client_for(run_response) as (client, _):
            stream = await client.responses.create(model="test-model", input="Hello", stream=True)

            event_types: list[str] = []
            deltas: list[str] = []
            async for event in stream:
                event_types.append(event.type)
                if event.type == "response.output_text.delta":
                    deltas.append(event.delta)

            assert "response.created" in event_types
            assert "response.completed" in event_types
            assert "".join(deltas) == "Hello from SDK"

    asyncio.run(_run())


@pytest.mark.integration
def test_openai_responses_streaming_function_call_events():
    class FunctionCallChunk:
        def __init__(self, *, call_id: str, name: str | None, arguments: str):
            self.function_call_id = call_id
            self.function_call_name = name
            self.function_call_arguments = arguments

    def run_response(_model, _input_items, body):
        if body.get("stream"):

            def _gen() -> Generator[FunctionCallChunk, None, None]:
                yield FunctionCallChunk(call_id="call_1", name="lookup", arguments='{"q":')
                yield FunctionCallChunk(call_id="call_1", name=None, arguments='"books"}')

            return _gen()
        return "not-streamed"

    async def _run() -> None:
        async with _openai_client_for(run_response) as (client, _):
            stream = await client.responses.create(model="test-model", input="Hello", stream=True)

            event_types: list[str] = []
            argument_deltas: list[str] = []
            final_arguments: list[str] = []
            async for event in stream:
                event_types.append(event.type)
                if event.type == "response.function_call_arguments.delta":
                    argument_deltas.append(event.delta)
                if event.type == "response.function_call_arguments.done":
                    final_arguments.append(event.arguments)

            assert "response.function_call_arguments.delta" in event_types
            assert "response.function_call_arguments.done" in event_types
            assert "".join(argument_deltas) == '{"q":"books"}'
            assert final_arguments[0] == '{"q":"books"}'

    asyncio.run(_run())


@pytest.mark.integration
def test_openai_responses_create_function_call_output_item():
    def run_response(model, _input_items, _body):
        return Response(
            id="resp_fc",
            created_at=int(time.time()),
            model=model,
            output=[
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                }
            ],
        )

    async def _run() -> None:
        async with _openai_client_for(run_response) as (client, _):
            response = await client.responses.create(model="test-model", input="Weather?")
            assert response.id == "resp_fc"
            assert len(response.output) == 1
            assert response.output[0].type == "function_call"
            assert response.output[0].call_id == "call_1"
            assert response.output[0].name == "get_weather"
            assert response.output[0].arguments == '{"city":"Paris"}'

    asyncio.run(_run())


@pytest.mark.integration
def test_openai_responses_streaming_tool_calls_via_streaming_chunk():
    """OpenAI SDK parses function_call events from Haystack StreamingChunk + ToolCallDelta."""
    from haystack.dataclasses import StreamingChunk
    from haystack.dataclasses.streaming_chunk import ToolCallDelta

    def run_response(_model, _input_items, body):
        if body.get("stream"):

            def _gen() -> Generator[StreamingChunk, None, None]:
                yield StreamingChunk(
                    content="",
                    tool_calls=[ToolCallDelta(index=0, id="call_1", tool_name="get_weather", arguments='{"city":')],
                    index=0,
                )
                yield StreamingChunk(
                    content="",
                    tool_calls=[ToolCallDelta(index=0, arguments='"Paris"}')],
                    index=0,
                )

            return _gen()
        return "not-streamed"

    async def _run() -> None:
        async with _openai_client_for(run_response) as (client, _):
            stream = await client.responses.create(model="test-model", input="Weather?", stream=True)

            event_types: list[str] = []
            argument_deltas: list[str] = []
            final_arguments: list[str] = []
            async for event in stream:
                event_types.append(event.type)
                if event.type == "response.function_call_arguments.delta":
                    argument_deltas.append(event.delta)
                if event.type == "response.function_call_arguments.done":
                    final_arguments.append(event.arguments)

            assert "response.function_call_arguments.delta" in event_types
            assert "response.function_call_arguments.done" in event_types
            assert "response.output_item.done" in event_types
            assert "".join(argument_deltas) == '{"city":"Paris"}'
            assert final_arguments[0] == '{"city":"Paris"}'

    asyncio.run(_run())


@pytest.mark.integration
def test_openai_responses_multimodal_input_image_and_file_url():
    def run_response(_model, input_items, _body):
        content_items = input_items[0]["content"]
        seen_types = [item["type"] for item in content_items]
        return ",".join(seen_types)

    async def _run() -> None:
        async with _openai_client_for(run_response) as (client, _):
            response = await client.responses.create(
                model="test-model",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this"},
                            {"type": "input_image", "image_url": "https://example.com/test.png"},
                            {"type": "input_file", "file_url": "https://example.com/doc.pdf"},
                        ],
                    }
                ],
            )
            assert response.output[0].content[0].text == "input_text,input_image,input_file"

    asyncio.run(_run())


@pytest.mark.integration
def test_openai_file_upload_then_responses_with_file_id():
    def run_response(_model, input_items, _body):
        file_part = next(
            part for part in input_items[0]["content"] if isinstance(part, dict) and part.get("type") == "input_file"
        )
        return f"got-file:{file_part['file_id']}"

    async def _run() -> None:
        async with _openai_client_for(run_response, include_files_api=True) as (client, uploaded_files):
            uploaded = await client.files.create(
                file=("notes.txt", io.BytesIO(b"hello world"), "text/plain"),
                purpose="user_data",
            )
            assert uploaded.id in uploaded_files

            response = await client.responses.create(
                model="test-model",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": uploaded.id},
                            {"type": "input_text", "text": "Summarize"},
                        ],
                    }
                ],
            )
            assert response.output[0].content[0].text == f"got-file:{uploaded.id}"

    asyncio.run(_run())


@pytest.mark.integration
def test_openai_responses_create_without_input_uses_previous_response_id():
    def run_response(_model, input_items, body):
        return f"items={len(input_items)};previous={body.get('previous_response_id')}"

    async def _run() -> None:
        async with _openai_client_for(run_response) as (client, _):
            response = await client.responses.create(
                model="test-model",
                previous_response_id="resp_prev_1",
                instructions="Continue.",
            )
            assert response.output[0].content[0].text == "items=0;previous=resp_prev_1"

    asyncio.run(_run())


@pytest.mark.integration
def test_openai_responses_streaming_reasoning_events():
    from haystack.dataclasses import ReasoningContent, StreamingChunk

    def run_response(_model, _input_items, body):
        if body.get("stream"):

            def _gen() -> Generator[StreamingChunk, None, None]:
                yield StreamingChunk(content="", reasoning=ReasoningContent(reasoning_text="analyzing"), index=0)
                yield StreamingChunk(content="result", index=0)

            return _gen()
        return "not-streamed"

    async def _run() -> None:
        async with _openai_client_for(run_response) as (client, _):
            stream = await client.responses.create(model="test-model", input="think", stream=True)

            event_types: list[str] = []
            reasoning_deltas: list[str] = []
            text_deltas: list[str] = []
            async for event in stream:
                event_types.append(event.type)
                if event.type == "response.reasoning_summary_text.delta":
                    reasoning_deltas.append(event.delta)
                if event.type == "response.output_text.delta":
                    text_deltas.append(event.delta)

            assert "response.reasoning_summary_text.delta" in event_types
            assert "response.reasoning_summary_text.done" in event_types
            assert "response.reasoning_summary_part.added" in event_types
            assert "response.reasoning_summary_part.done" in event_types
            assert "response.output_text.delta" in event_types
            assert "response.completed" in event_types
            assert reasoning_deltas == ["analyzing"]
            assert text_deltas == ["result"]

    asyncio.run(_run())


@pytest.mark.integration
def test_openai_responses_non_streaming_reasoning_item():
    def run_response(model, _input_items, _body):
        return Response(
            id="resp_reason",
            created_at=int(time.time()),
            model=model,
            output=[
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "thought process"}],
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                },
            ],
        )

    async def _run() -> None:
        async with _openai_client_for(run_response) as (client, _):
            response = await client.responses.create(model="test-model", input="think")
            assert len(response.output) == 2
            assert response.output[0].type == "reasoning"
            assert response.output[0].summary[0].text == "thought process"
            assert response.output[1].type == "message"
            assert response.output[1].content[0].text == "answer"

    asyncio.run(_run())
