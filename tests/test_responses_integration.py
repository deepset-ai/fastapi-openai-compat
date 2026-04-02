import asyncio
import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from fastapi_openai_compat import create_models_router, create_openai_router
from fastapi_openai_compat.responses.models import Response
from fastapi_openai_compat.responses.router import create_responses_router


def _ok_response(_model, _input_items, _body):
    return "ok"


def _chat_ok(_model, _messages, _body):
    return "chat-ok"


def _resp_ok(_model, _input_items, _body):
    return "resp-ok"


def _make_app(run_response, pre_hook=None, post_hook=None, *, include_models_endpoints=False):
    app = FastAPI()
    router = create_responses_router(
        list_models=lambda: ["test-model"],
        run_response=run_response,
        pre_hook=pre_hook,
        post_hook=post_hook,
        include_models_endpoints=include_models_endpoints,
    )
    app.include_router(router)
    return app


@pytest.fixture()
def echo_client():
    def run_response(model, input_items, body):
        text = ""
        for item in input_items:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content", "")
                text = content if isinstance(content, str) else str(content)
        return f"Echo: {text}"

    app = _make_app(run_response)
    with TestClient(app) as tc:
        yield tc


@pytest.mark.integration
def test_non_streaming_text(echo_client: TestClient):
    resp = echo_client.post("/v1/responses", json={"model": "test-model", "input": "Hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert len(data["output"]) == 1
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["content"][0]["text"] == "Echo: Hello"


@pytest.mark.integration
def test_non_streaming_input_items(echo_client: TestClient):
    input_items = [{"type": "message", "role": "user", "content": "World"}]
    resp = echo_client.post("/v1/responses", json={"model": "test-model", "input": input_items})
    assert resp.status_code == 200
    assert resp.json()["output"][0]["content"][0]["text"] == "Echo: World"


@pytest.mark.integration
def test_streaming():
    def run_response(model, input_items, body):
        def gen():
            yield "Hello "
            yield "world!"

        return gen()

    app = _make_app(run_response)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "Hi", "stream": True})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    blocks = [block for block in resp.text.split("\n\n") if block.strip()]
    event_names = [
        line.removeprefix("event: ") for block in blocks for line in block.split("\n") if line.startswith("event: ")
    ]
    assert "response.created" in event_names
    assert "response.output_text.delta" in event_names
    assert "response.completed" in event_names


@pytest.mark.integration
def test_response_object_passthrough():
    def run_response(model, input_items, body):
        return Response(
            id="resp_custom",
            created_at=int(time.time()),
            model=model,
            output=[
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Custom!", "annotations": []}],
                }
            ],
        )

    app = _make_app(run_response)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "Hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "resp_custom"
    assert data["output"][0]["content"][0]["text"] == "Custom!"


@pytest.mark.integration
def test_pre_hook_transformer():
    def pre_hook(req):
        req.input = "Modified"
        return req

    def run_response(model, input_items, body):
        return f"Got: {input_items}"

    app = _make_app(run_response, pre_hook=pre_hook)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "Original"})
    assert resp.status_code == 200
    assert "Modified" in resp.json()["output"][0]["content"][0]["text"]


@pytest.mark.integration
def test_pre_hook_observer_does_not_modify_request():
    observed = []

    def observer(req):
        observed.append(req.model)

    app = _make_app(_ok_response, pre_hook=observer)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "Hi"})
    assert resp.status_code == 200
    assert observed == ["test-model"]


@pytest.mark.integration
def test_post_hook_observer_does_not_modify_result():
    observed = []

    def observer(result):
        observed.append(type(result).__name__)

    app = _make_app(_ok_response, post_hook=observer)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "Hi"})
    assert resp.status_code == 200
    assert observed == ["str"]


@pytest.mark.integration
def test_alias_endpoint():
    app = _make_app(_ok_response)
    with TestClient(app) as client:
        resp = client.post("/responses", json={"model": "test-model", "input": "Hi"})
    assert resp.status_code == 200


@pytest.mark.integration
def test_error_handling():
    def run_response(model, input_items, body):
        msg = "Something broke"
        raise ValueError(msg)

    app = _make_app(run_response)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "Hi"})
    assert resp.status_code == 500
    assert "Pipeline execution failed" in resp.json()["detail"]


@pytest.mark.integration
def test_http_exception_passthrough():
    def run_response(model, input_items, body):
        raise HTTPException(status_code=404, detail="Model not found")

    app = _make_app(run_response)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "Hi"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Model not found"


@pytest.mark.integration
def test_models_endpoints_off_by_default():
    app = _make_app(_ok_response)
    with TestClient(app) as client:
        resp_v1 = client.get("/v1/models")
        resp_alias = client.get("/models")
    assert resp_v1.status_code == 404
    assert resp_alias.status_code == 404


@pytest.mark.integration
def test_models_endpoints_can_be_enabled():
    app = _make_app(_ok_response, include_models_endpoints=True)
    with TestClient(app) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert any(model["id"] == "test-model" for model in data["data"])


@pytest.mark.integration
def test_sync_callbacks_do_not_block_event_loop():
    def slow_run_response(model, input_items, body):
        time.sleep(0.1)
        return "ok"

    app = _make_app(slow_run_response)

    async def _run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            t0 = asyncio.get_event_loop().time()
            results = await asyncio.gather(
                client.post("/v1/responses", json={"model": "test-model", "input": "a"}),
                client.post("/v1/responses", json={"model": "test-model", "input": "b"}),
            )
            elapsed = asyncio.get_event_loop().time() - t0
        return results, elapsed

    results, elapsed = asyncio.run(_run())
    assert all(r.status_code == 200 for r in results)
    assert elapsed < 0.35, f"Expected concurrent execution but took {elapsed:.2f}s"


@pytest.mark.integration
def test_can_coexist_with_chat_router_in_same_app():
    app = FastAPI()

    chat_router = create_openai_router(
        list_models=lambda: ["test-model"],
        run_completion=_chat_ok,
    )
    responses_router = create_responses_router(
        list_models=lambda: ["test-model"],
        run_response=_resp_ok,
        include_models_endpoints=False,
    )
    app.include_router(chat_router)
    app.include_router(responses_router)

    with TestClient(app) as client:
        models = client.get("/v1/models")
        chat = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
        )
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "hello"})

    assert models.status_code == 200
    assert models.json()["object"] == "list"
    assert chat.status_code == 200
    assert resp.status_code == 200


@pytest.mark.integration
def test_streaming_tool_calls_via_streaming_chunk():
    """Full HTTP round-trip: StreamingChunk + ToolCallDelta → SSE function_call events."""
    from haystack.dataclasses import StreamingChunk
    from haystack.dataclasses.streaming_chunk import ToolCallDelta

    def run_response(_model, _input_items, body):
        if body.get("stream"):

            def _gen():
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

    app = _make_app(run_response)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "weather?", "stream": True})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    blocks = [block for block in resp.text.split("\n\n") if block.strip()]
    event_names = [
        line.removeprefix("event: ") for block in blocks for line in block.split("\n") if line.startswith("event: ")
    ]
    assert "response.created" in event_names
    assert "response.function_call_arguments.delta" in event_names
    assert "response.function_call_arguments.done" in event_names
    assert "response.output_item.done" in event_names
    assert "response.completed" in event_names


@pytest.mark.integration
def test_streaming_tool_calls_async():
    """Async version: StreamingChunk + ToolCallDelta through the Responses API."""
    from haystack.dataclasses import StreamingChunk
    from haystack.dataclasses.streaming_chunk import ToolCallDelta

    async def run_response(_model, _input_items, body):
        if body.get("stream"):

            async def _gen():
                yield StreamingChunk(
                    content="",
                    tool_calls=[ToolCallDelta(index=0, id="call_1", tool_name="search", arguments='{"q":"test"}')],
                    index=0,
                )

            return _gen()
        return "not-streamed"

    app = _make_app(run_response)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "search", "stream": True})

    assert resp.status_code == 200
    blocks = [block for block in resp.text.split("\n\n") if block.strip()]
    event_names = [
        line.removeprefix("event: ") for block in blocks for line in block.split("\n") if line.startswith("event: ")
    ]
    assert "response.function_call_arguments.delta" in event_names
    assert "response.function_call_arguments.done" in event_names


@pytest.mark.integration
def test_can_use_dedicated_models_router_without_shadowing():
    app = FastAPI()

    models_router = create_models_router(list_models=lambda: ["test-model"])
    chat_router = create_openai_router(
        list_models=lambda: ["test-model"],
        run_completion=_chat_ok,
        include_models_endpoints=False,
    )
    responses_router = create_responses_router(
        list_models=lambda: ["test-model"],
        run_response=_resp_ok,
        include_models_endpoints=False,
    )

    app.include_router(models_router)
    app.include_router(chat_router)
    app.include_router(responses_router)

    model_routes = [route for route in app.routes if getattr(route, "path", None) == "/v1/models"]
    assert len(model_routes) == 1

    with TestClient(app) as client:
        models = client.get("/v1/models")
        alias_models = client.get("/models")
        chat = client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
        )
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "hello"})

    assert models.status_code == 200
    assert alias_models.status_code == 200
    assert chat.status_code == 200
    assert resp.status_code == 200


@pytest.mark.integration
def test_streaming_reasoning_with_haystack_chunks():
    from haystack.dataclasses import ReasoningContent, StreamingChunk

    def run_response(_model, _input_items, body):
        if body.get("stream"):

            def _gen():
                yield StreamingChunk(
                    content="",
                    reasoning=ReasoningContent(reasoning_text="analyzing..."),
                    index=0,
                )
                yield StreamingChunk(content="the answer", index=0)

            return _gen()
        return "not-streamed"

    app = _make_app(run_response)
    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={"model": "test-model", "input": "think", "stream": True})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    blocks = [block for block in resp.text.split("\n\n") if block.strip()]
    event_names = [
        line.removeprefix("event: ") for block in blocks for line in block.split("\n") if line.startswith("event: ")
    ]
    assert "response.reasoning_summary_text.delta" in event_names
    assert "response.reasoning_summary_text.done" in event_names
    assert "response.output_text.delta" in event_names
    assert "response.completed" in event_names

    import json

    completed_block = next(b for b in blocks if "response.completed" in b)
    data_line = next(line for line in completed_block.split("\n") if line.startswith("data: "))
    completed = json.loads(data_line.removeprefix("data: "))
    output = completed["response"]["output"]
    assert len(output) == 2
    assert output[0]["type"] == "reasoning"
    assert output[0]["summary"][0]["text"] == "analyzing..."
    assert output[1]["type"] == "message"
    assert output[1]["content"][0]["text"] == "the answer"
