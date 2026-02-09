from collections.abc import Generator

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport
from openai import AsyncOpenAI

from fastapi_openai_compat import CompletionResult, create_openai_router


def _build_app() -> FastAPI:
    def list_models() -> list[str]:
        return ["echo-pipeline", "streaming-pipeline"]

    def run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
        last = messages[-1]["content"] if messages else ""
        if model == "streaming-pipeline":

            def _gen() -> Generator[str, None, None]:
                for word in last.split():
                    yield word + " "

            return _gen()
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
