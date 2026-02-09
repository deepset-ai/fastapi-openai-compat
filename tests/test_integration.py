import asyncio
import json
import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from fastapi_openai_compat import ChatCompletion, ChatRequest, Choice, CompletionResult, Message, create_openai_router


@pytest.mark.integration
def test_get_models_v1(client: TestClient):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    names = [m["id"] for m in data["data"]]
    assert "echo-pipeline" in names
    assert "streaming-pipeline" in names
    assert all(m["object"] == "model" for m in data["data"])


@pytest.mark.integration
def test_get_models_alias(client: TestClient):
    resp = client.get("/models")
    assert resp.status_code == 200
    assert resp.json()["object"] == "list"


@pytest.mark.integration
def test_models_owned_by_default(client: TestClient):
    resp = client.get("/v1/models")
    for model in resp.json()["data"]:
        assert model["owned_by"] == "custom"


@pytest.mark.integration
def test_models_custom_owned_by():
    async def _list() -> list[str]:
        return ["a"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        return "nope"

    router = create_openai_router(
        list_models=_list,
        run_completion=_run,
        owned_by="my-org",
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.get("/v1/models")
        assert resp.json()["data"][0]["owned_by"] == "my-org"


@pytest.mark.integration
def test_chat_completion_non_streaming(client: TestClient):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "echo-pipeline",
            "messages": [{"role": "user", "content": "world"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Echo: world"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["choices"][0]["message"]["role"] == "assistant"


@pytest.mark.integration
def test_chat_completion_alias_path(client: TestClient):
    resp = client.post(
        "/chat/completions",
        json={
            "model": "echo-pipeline",
            "messages": [{"role": "user", "content": "test"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Echo: test"


@pytest.mark.integration
def test_chat_completion_extra_fields_forwarded(client: TestClient):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "echo-pipeline",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "max_tokens": 50,
        },
    )
    assert resp.status_code == 200


@pytest.mark.integration
def test_chat_completion_sync_streaming(client: TestClient):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "streaming-pipeline",
            "messages": [{"role": "user", "content": "hello world"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]

    assert len(data_lines) == 3

    first = json.loads(data_lines[0][len("data: ") :])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["content"] == "hello "

    last = json.loads(data_lines[-1][len("data: ") :])
    assert last["choices"][0]["finish_reason"] == "stop"


@pytest.mark.integration
def test_chat_completion_async_streaming(client: TestClient):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "async-streaming-pipeline",
            "messages": [{"role": "user", "content": "foo bar baz"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]

    assert len(data_lines) == 4

    contents = []
    for dl in data_lines[:-1]:
        chunk = json.loads(dl[len("data: ") :])
        contents.append(chunk["choices"][0]["delta"]["content"])

    assert contents == ["foo ", "bar ", "baz "]


@pytest.mark.integration
def test_unknown_model_returns_500(client: TestClient):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert resp.status_code == 500
    assert "Pipeline execution failed" in resp.json()["detail"]


@pytest.mark.integration
def test_run_completion_raising_http_exception():
    async def failing_run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        raise HTTPException(status_code=404, detail="Model not found")

    async def _list() -> list[str]:
        return ["m"]

    app = FastAPI()
    router = create_openai_router(
        list_models=_list,
        run_completion=failing_run,
    )
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Model not found"


@pytest.mark.integration
def test_pre_hook_modifies_request(client_with_hooks: TestClient):
    resp = client_with_hooks.post(
        "/v1/chat/completions",
        json={
            "model": "echo-pipeline",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert content == "ECHO: HI"


@pytest.mark.integration
def test_post_hook_transforms_result(client_with_hooks: TestClient):
    resp = client_with_hooks.post(
        "/v1/chat/completions",
        json={
            "model": "echo-pipeline",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    content = resp.json()["choices"][0]["message"]["content"]
    assert content == "ECHO: HELLO"


@pytest.mark.integration
def test_pre_hook_can_abort_with_http_exception():
    async def rejecting_hook(req: ChatRequest) -> ChatRequest:
        raise HTTPException(status_code=403, detail="Forbidden by hook")

    async def _list() -> list[str]:
        return ["m"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        return "nope"

    app = FastAPI()
    router = create_openai_router(
        list_models=_list,
        run_completion=_run,
        pre_hook=rejecting_hook,
    )
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Forbidden by hook"


@pytest.mark.integration
def test_streaming_with_hooks_still_works(client_with_hooks: TestClient):
    resp = client_with_hooks.post(
        "/v1/chat/completions",
        json={
            "model": "streaming-pipeline",
            "messages": [{"role": "user", "content": "a b"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert len(data_lines) == 3


@pytest.mark.integration
def test_observer_pre_hook_does_not_modify_request():
    observed_models: list[str] = []

    async def observer_pre(req: ChatRequest) -> None:
        observed_models.append(req.model)

    async def _list() -> list[str]:
        return ["echo"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        return f"Echo: {messages[-1]['content']}"

    app = FastAPI()
    router = create_openai_router(
        list_models=_list,
        run_completion=_run,
        pre_hook=observer_pre,
    )
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "echo", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Echo: hello"
        assert observed_models == ["echo"]


@pytest.mark.integration
def test_observer_post_hook_does_not_modify_result():
    observed_results: list[str] = []

    async def observer_post(result: CompletionResult) -> None:
        if isinstance(result, str):
            observed_results.append(result)

    async def _list() -> list[str]:
        return ["echo"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        return f"Echo: {messages[-1]['content']}"

    app = FastAPI()
    router = create_openai_router(
        list_models=_list,
        run_completion=_run,
        post_hook=observer_post,
    )
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "echo", "messages": [{"role": "user", "content": "world"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Echo: world"
        assert observed_results == ["Echo: world"]


@pytest.mark.integration
def test_observer_hooks_with_streaming():
    pre_called = False
    post_called = False

    async def observer_pre(req: ChatRequest) -> None:
        nonlocal pre_called
        pre_called = True

    async def observer_post(result: CompletionResult) -> None:
        nonlocal post_called
        post_called = True

    async def _list() -> list[str]:
        return ["stream"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        async def _gen():
            yield "chunk1"
            yield "chunk2"

        return _gen()

    app = FastAPI()
    router = create_openai_router(
        list_models=_list,
        run_completion=_run,
        pre_hook=observer_pre,
        post_hook=observer_post,
    )
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "stream", "messages": [{"role": "user", "content": "x"}], "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) == 3
        assert pre_called
        assert post_called


@pytest.mark.integration
def test_sync_list_models(client_sync: TestClient):
    resp = client_sync.get("/v1/models")
    assert resp.status_code == 200
    names = [m["id"] for m in resp.json()["data"]]
    assert "echo-pipeline" in names


@pytest.mark.integration
def test_sync_run_completion(client_sync: TestClient):
    resp = client_sync.post(
        "/v1/chat/completions",
        json={
            "model": "echo-pipeline",
            "messages": [{"role": "user", "content": "sync test"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Echo: sync test"


@pytest.mark.integration
def test_sync_run_completion_streaming(client_sync: TestClient):
    resp = client_sync.post(
        "/v1/chat/completions",
        json={
            "model": "streaming-pipeline",
            "messages": [{"role": "user", "content": "a b c"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    lines = resp.text.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("data: ")]
    assert len(data_lines) == 4


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sync_callbacks_do_not_block_event_loop():
    def slow_list_models() -> list[str]:
        time.sleep(0.1)
        return ["slow-model"]

    def slow_run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        time.sleep(0.1)
        return f"done: {messages[-1]['content']}"

    app = FastAPI()
    router = create_openai_router(
        list_models=slow_list_models,
        run_completion=slow_run,
    )
    app.include_router(router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        t0 = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            ac.post(
                "/v1/chat/completions",
                json={"model": "slow-model", "messages": [{"role": "user", "content": "a"}]},
            ),
            ac.post(
                "/v1/chat/completions",
                json={"model": "slow-model", "messages": [{"role": "user", "content": "b"}]},
            ),
        )
        elapsed = asyncio.get_event_loop().time() - t0

        for resp in results:
            assert resp.status_code == 200

        assert elapsed < 0.35, f"Expected concurrent execution but took {elapsed:.2f}s"


@pytest.mark.integration
def test_sync_pre_hook_transforms_request(client_with_sync_hooks: TestClient):
    resp = client_with_sync_hooks.post(
        "/v1/chat/completions",
        json={
            "model": "echo-pipeline",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert content == "ECHO: HI"


@pytest.mark.integration
def test_sync_post_hook_transforms_result(client_with_sync_hooks: TestClient):
    resp = client_with_sync_hooks.post(
        "/v1/chat/completions",
        json={
            "model": "echo-pipeline",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    content = resp.json()["choices"][0]["message"]["content"]
    assert content == "ECHO: HELLO"


@pytest.mark.integration
def test_sync_observer_hooks():
    observed: list[str] = []

    def sync_observer_pre(req: ChatRequest) -> None:
        observed.append(f"pre:{req.model}")

    def sync_observer_post(result: CompletionResult) -> None:
        if isinstance(result, str):
            observed.append(f"post:{result}")

    async def _list() -> list[str]:
        return ["echo"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        return f"Echo: {messages[-1]['content']}"

    app = FastAPI()
    router = create_openai_router(
        list_models=_list,
        run_completion=_run,
        pre_hook=sync_observer_pre,
        post_hook=sync_observer_post,
    )
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "echo", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Echo: hi"
        assert observed == ["pre:echo", "post:Echo: hi"]


@pytest.mark.integration
def test_custom_chunk_mapper_sync_stream():
    from dataclasses import dataclass

    @dataclass
    class MyChunk:
        text: str
        metadata: dict

    async def _list() -> list[str]:
        return ["custom"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        def _gen():
            yield MyChunk(text="hello ", metadata={"score": 0.9})
            yield MyChunk(text="world", metadata={"score": 0.8})

        return _gen()

    def my_mapper(chunk):
        return chunk.text

    app = FastAPI()
    router = create_openai_router(
        list_models=_list,
        run_completion=_run,
        chunk_mapper=my_mapper,
    )
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "custom", "messages": [{"role": "user", "content": "x"}], "stream": True},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) == 3
        first = json.loads(data_lines[0][len("data: ") :])
        assert first["choices"][0]["delta"]["content"] == "hello "
        second = json.loads(data_lines[1][len("data: ") :])
        assert second["choices"][0]["delta"]["content"] == "world"


@pytest.mark.integration
def test_custom_chunk_mapper_async_stream():
    async def _list() -> list[str]:
        return ["custom"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        async def _gen():
            yield {"payload": "async "}
            yield {"payload": "chunks"}

        return _gen()

    def dict_mapper(chunk):
        return chunk["payload"]

    app = FastAPI()
    router = create_openai_router(
        list_models=_list,
        run_completion=_run,
        chunk_mapper=dict_mapper,
    )
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "custom", "messages": [{"role": "user", "content": "x"}], "stream": True},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) == 3
        first = json.loads(data_lines[0][len("data: ") :])
        assert first["choices"][0]["delta"]["content"] == "async "
        second = json.loads(data_lines[1][len("data: ") :])
        assert second["choices"][0]["delta"]["content"] == "chunks"


@pytest.mark.integration
def test_tool_call_non_streaming():
    async def _list() -> list[str]:
        return ["tool-model"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        return ChatCompletion(
            id="resp-tool-1",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    app = FastAPI()
    router = create_openai_router(list_models=_list, run_completion=_run)
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "tool-model", "messages": [{"role": "user", "content": "weather?"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "resp-tool-1"
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert data["choices"][0]["message"]["content"] is None
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert data["usage"]["total_tokens"] == 15


@pytest.mark.integration
def test_tool_call_streaming():
    async def _list() -> list[str]:
        return ["tool-model"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        async def _gen():
            yield ChatCompletion(
                id="resp-tc-stream",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[
                    Choice(
                        index=0,
                        delta=Message(
                            role="assistant",
                            tool_calls=[
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ],
                        ),
                    )
                ],
            )
            yield ChatCompletion(
                id="resp-tc-stream",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[
                    Choice(
                        index=0,
                        delta=Message(
                            role="assistant",
                            tool_calls=[{"index": 0, "function": {"arguments": '{"city": "Paris"}'}}],
                        ),
                    )
                ],
            )
            yield ChatCompletion(
                id="resp-tc-stream",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[Choice(index=0, delta=Message(role="assistant"), finish_reason="tool_calls")],
            )

        return _gen()

    app = FastAPI()
    router = create_openai_router(list_models=_list, run_completion=_run)
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={
                "model": "tool-model",
                "messages": [{"role": "user", "content": "weather?"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) == 3

        first = json.loads(data_lines[0][len("data: ") :])
        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"

        second = json.loads(data_lines[1][len("data: ") :])
        assert second["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'

        last = json.loads(data_lines[2][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.integration
def test_chat_completion_passthrough_with_usage():
    async def _list() -> list[str]:
        return ["m"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        return ChatCompletion(
            id="custom-id",
            object="chat.completion",
            created=1234567890,
            model=model,
            choices=[Choice(index=0, message=Message(role="assistant", content="done"), finish_reason="stop")],
            usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            system_fingerprint="fp_test123",
        )

    app = FastAPI()
    router = create_openai_router(list_models=_list, run_completion=_run)
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        data = resp.json()
        assert data["id"] == "custom-id"
        assert data["created"] == 1234567890
        assert data["usage"]["total_tokens"] == 6
        assert data["system_fingerprint"] == "fp_test123"


@pytest.mark.integration
def test_streaming_chunk_tool_calls():
    from haystack.dataclasses import StreamingChunk
    from haystack.dataclasses.streaming_chunk import ToolCallDelta

    def _list() -> list[str]:
        return ["tool-model"]

    def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        def _gen():
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
            yield StreamingChunk(content="", finish_reason="tool_calls")

        return _gen()

    app = FastAPI()
    router = create_openai_router(list_models=_list, run_completion=_run)
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "tool-model", "messages": [{"role": "user", "content": "weather?"}], "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) == 3

        first = json.loads(data_lines[0][len("data: ") :])
        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert first["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"
        assert first["choices"][0]["delta"]["content"] is None

        second = json.loads(data_lines[1][len("data: ") :])
        assert second["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'

        last = json.loads(data_lines[2][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.integration
def test_streaming_chunk_tool_calls_async():
    from haystack.dataclasses import StreamingChunk
    from haystack.dataclasses.streaming_chunk import ToolCallDelta

    def _list() -> list[str]:
        return ["tool-model"]

    async def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        async def _gen():
            yield StreamingChunk(
                content="",
                tool_calls=[ToolCallDelta(index=0, id="call_1", tool_name="search", arguments='{"q": "test"}')],
                index=0,
            )
            yield StreamingChunk(content="", finish_reason="tool_calls")

        return _gen()

    app = FastAPI()
    router = create_openai_router(list_models=_list, run_completion=_run)
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "tool-model", "messages": [{"role": "user", "content": "x"}], "stream": True},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) == 2

        first = json.loads(data_lines[0][len("data: ") :])
        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "search"
        assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"q": "test"}'


@pytest.mark.integration
def test_streaming_chunk_finish_reason_propagated():
    from haystack.dataclasses import StreamingChunk

    def _list() -> list[str]:
        return ["m"]

    def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        def _gen():
            yield StreamingChunk(content="truncated")
            yield StreamingChunk(content="", finish_reason="length")

        return _gen()

    app = FastAPI()
    router = create_openai_router(list_models=_list, run_completion=_run)
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) == 2

        first = json.loads(data_lines[0][len("data: ") :])
        assert first["choices"][0]["delta"]["content"] == "truncated"

        last = json.loads(data_lines[1][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "length"


@pytest.mark.integration
def test_streaming_chunk_auto_stop_when_no_finish_reason():
    from haystack.dataclasses import StreamingChunk

    def _list() -> list[str]:
        return ["m"]

    def _run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        def _gen():
            yield StreamingChunk(content="hello ")
            yield StreamingChunk(content="world")

        return _gen()

    app = FastAPI()
    router = create_openai_router(list_models=_list, run_completion=_run)
    app.include_router(router)
    with TestClient(app) as tc:
        resp = tc.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True},
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) == 3

        last = json.loads(data_lines[-1][len("data: ") :])
        assert last["choices"][0]["finish_reason"] == "stop"
