import json
from collections.abc import Generator

import httpx
import pytest
from fastapi import FastAPI
from haystack.dataclasses import ReasoningContent, StreamingChunk
from httpx import ASGITransport
from openai import AsyncOpenAI

from fastapi_openai_compat import CompletionResult, create_openai_router


class StatusEvent:
    def __init__(self, description: str, done: bool = False):
        self.description = description
        self.done = done

    def to_event_dict(self) -> dict:
        return {"type": "status", "data": {"description": self.description, "done": self.done}}


def _build_app() -> FastAPI:  # noqa: C901
    def list_models() -> list[str]:
        return ["echo-pipeline", "streaming-pipeline", "event-streaming-pipeline", "reasoning-pipeline"]

    def run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
        last = messages[-1]["content"] if messages else ""
        if model == "streaming-pipeline":

            def _gen() -> Generator[str, None, None]:
                for word in last.split():
                    yield word + " "

            return _gen()
        if model == "event-streaming-pipeline":

            def _gen_events() -> Generator[str | StatusEvent, None, None]:
                yield StatusEvent("Working...")
                for word in last.split():
                    yield word + " "
                yield StatusEvent("Done", done=True)

            return _gen_events()
        if model == "reasoning-pipeline":

            def _gen_reasoning() -> Generator[StreamingChunk, None, None]:
                yield StreamingChunk(content="", reasoning=ReasoningContent(reasoning_text="let me think"), index=0)
                yield StreamingChunk(content="the answer", index=0)

            return _gen_reasoning()
        return f"Echo: {last}"

    app = FastAPI()
    router = create_openai_router(list_models=list_models, run_completion=run_completion)
    app.include_router(router)
    return app


@pytest.fixture()
async def openai_client():
    app = _build_app()
    transport = ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    client = AsyncOpenAI(api_key="test-key", base_url="http://testserver/v1", http_client=http_client)
    yield client
    await http_client.aclose()


@pytest.fixture()
def raw_http_client():
    app = _build_app()
    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_list_models(openai_client):
    models = await openai_client.models.list()
    ids = [m.id for m in models.data]
    assert "echo-pipeline" in ids
    assert "streaming-pipeline" in ids


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_chat_completion(openai_client):
    response = await openai_client.chat.completions.create(
        model="echo-pipeline",
        messages=[{"role": "user", "content": "hello world"}],
    )
    assert response.choices[0].message.content == "Echo: hello world"
    assert response.choices[0].finish_reason == "stop"
    assert response.model == "echo-pipeline"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_chat_streaming(openai_client):
    stream = await openai_client.chat.completions.create(
        model="streaming-pipeline",
        messages=[{"role": "user", "content": "foo bar baz"}],
        stream=True,
    )

    contents = []
    finish_reasons = []
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            contents.append(delta.content)
        if chunk.choices[0].finish_reason:
            finish_reasons.append(chunk.choices[0].finish_reason)

    assert contents == ["foo ", "bar ", "baz "]
    assert "stop" in finish_reasons


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_extra_params_accepted(openai_client):
    response = await openai_client.chat.completions.create(
        model="echo-pipeline",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        max_tokens=100,
    )
    assert response.choices[0].message.content == "Echo: hi"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_response_has_expected_fields(openai_client):
    response = await openai_client.chat.completions.create(
        model="echo-pipeline",
        messages=[{"role": "user", "content": "test"}],
    )
    assert response.id is not None
    assert response.created is not None
    assert response.object == "chat.completion"
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.index == 0
    assert choice.message.role == "assistant"
    assert choice.finish_reason == "stop"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_streaming_chunk_fields(openai_client):
    stream = await openai_client.chat.completions.create(
        model="streaming-pipeline",
        messages=[{"role": "user", "content": "a"}],
        stream=True,
    )

    chunks = [chunk async for chunk in stream]
    assert len(chunks) >= 2

    content_chunk = chunks[0]
    assert content_chunk.object == "chat.completion.chunk"
    assert content_chunk.id is not None
    assert content_chunk.model == "streaming-pipeline"
    assert content_chunk.choices[0].delta.role == "assistant"

    stop_chunk = chunks[-1]
    assert stop_chunk.choices[0].finish_reason == "stop"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_custom_events_raw_sse(raw_http_client):
    """Verify custom SSE events are present in the raw HTTP response."""
    resp = await raw_http_client.post(
        "/v1/chat/completions",
        json={
            "model": "event-streaming-pipeline",
            "messages": [{"role": "user", "content": "hello world"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    events = [json.loads(dl[len("data: ") :]) for dl in data_lines]

    custom_events = [e for e in events if "event" in e]
    assert len(custom_events) == 2
    assert custom_events[0]["event"]["type"] == "status"
    assert custom_events[0]["event"]["data"]["description"] == "Working..."
    assert custom_events[1]["event"]["data"]["done"] is True

    completion_chunks = [e for e in events if "choices" in e]
    contents = [
        c["choices"][0]["delta"]["content"]
        for c in completion_chunks
        if c["choices"][0].get("delta", {}).get("content")
    ]
    assert contents == ["hello ", "world "]

    last = completion_chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    await raw_http_client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_streaming_with_custom_events_still_works(openai_client):
    """The OpenAI SDK gracefully handles streams containing custom events."""
    stream = await openai_client.chat.completions.create(
        model="event-streaming-pipeline",
        messages=[{"role": "user", "content": "foo bar"}],
        stream=True,
    )

    contents = []
    finish_reasons = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            contents.append(delta.content)
        if chunk.choices[0].finish_reason:
            finish_reasons.append(chunk.choices[0].finish_reason)

    assert contents == ["foo ", "bar "]
    assert "stop" in finish_reasons


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reasoning_chunks_in_raw_sse(raw_http_client):
    resp = await raw_http_client.post(
        "/v1/chat/completions",
        json={
            "model": "reasoning-pipeline",
            "messages": [{"role": "user", "content": "think"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    events = [json.loads(dl[len("data: ") :]) for dl in data_lines]

    completion_chunks = [e for e in events if "choices" in e]
    reasoning = completion_chunks[0]["choices"][0]["delta"]
    assert reasoning["reasoning_content"] == "let me think"
    assert reasoning["content"] is None

    text = completion_chunks[1]["choices"][0]["delta"]
    assert text["content"] == "the answer"
    assert text["reasoning_content"] is None
    await raw_http_client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_sdk_streaming_with_reasoning_does_not_break(openai_client):
    stream = await openai_client.chat.completions.create(
        model="reasoning-pipeline",
        messages=[{"role": "user", "content": "think"}],
        stream=True,
    )

    contents = []
    finish_reasons = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            contents.append(delta.content)
        if chunk.choices[0].finish_reason:
            finish_reasons.append(chunk.choices[0].finish_reason)

    assert contents == ["the answer"]
    assert "stop" in finish_reasons
