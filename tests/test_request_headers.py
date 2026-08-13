"""Request headers are forwarded to callbacks that opt in, and only to those."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi_openai_compat import CompletionResult, create_openai_router
from fastapi_openai_compat._shared import callable_accepts_kwarg
from fastapi_openai_compat.responses import create_responses_router


def _client(run_completion) -> TestClient:
    app = FastAPI()
    app.include_router(create_openai_router(list_models=lambda: ["p"], run_completion=run_completion))
    return TestClient(app)


def _chat(client: TestClient, headers: dict[str, str] | None = None):
    return client.post(
        "/v1/chat/completions",
        json={"model": "p", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )


# --- opting in -------------------------------------------------------------------------------


@pytest.mark.integration
def test_headers_forwarded_when_callback_declares_them():
    seen = {}

    async def run(model: str, messages: list[dict], body: dict, headers: dict[str, str]) -> CompletionResult:
        seen.update(headers)
        return "ok"

    resp = _chat(_client(run), {"Authorization": "Bearer alice-token"})

    assert resp.status_code == 200
    # Starlette lower-cases header names.
    assert seen["authorization"] == "Bearer alice-token"


@pytest.mark.integration
def test_headers_forwarded_via_kwargs():
    seen = {}

    async def run(model: str, messages: list[dict], body: dict, **kwargs) -> CompletionResult:
        seen.update(kwargs.get("headers", {}))
        return "ok"

    resp = _chat(_client(run), {"X-Tenant": "acme"})

    assert resp.status_code == 200
    assert seen["x-tenant"] == "acme"


@pytest.mark.integration
def test_headers_forwarded_to_sync_callback():
    """Sync callbacks run in a threadpool, so the kwarg has to survive that wrapper."""
    seen = {}

    def run(model: str, messages: list[dict], body: dict, headers: dict[str, str]) -> CompletionResult:
        seen.update(headers)
        return "ok"

    resp = _chat(_client(run), {"Authorization": "Bearer bob-token"})

    assert resp.status_code == 200
    assert seen["authorization"] == "Bearer bob-token"


# --- not opting in (backwards compatibility) -------------------------------------------------


@pytest.mark.integration
def test_callback_without_headers_parameter_is_unaffected():
    """The pre-existing three-argument signature must keep working untouched."""
    calls = []

    async def run(model: str, messages: list[dict], body: dict) -> CompletionResult:
        calls.append((model, len(messages), sorted(body)[:1]))
        return "ok"

    resp = _chat(_client(run), {"Authorization": "Bearer nobody-should-see-this"})

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert calls == [("p", 1, ["messages"])]


@pytest.mark.integration
def test_no_headers_sent_still_works():
    async def run(model: str, messages: list[dict], body: dict, headers: dict[str, str]) -> CompletionResult:
        # The client always sends some headers (host, accept, ...), but never the one below.
        assert "authorization" not in headers
        return "ok"

    assert _chat(_client(run)).status_code == 200


# --- responses endpoint ----------------------------------------------------------------------


@pytest.mark.integration
def test_responses_endpoint_forwards_headers():
    seen = {}

    async def run_response(model: str, input_items: list[dict], body: dict, headers: dict[str, str]):
        seen.update(headers)
        return "ok"

    app = FastAPI()
    app.include_router(create_responses_router(list_models=lambda: ["p"], run_response=run_response))
    resp = TestClient(app).post(
        "/v1/responses",
        json={"model": "p", "input": "hi"},
        headers={"Authorization": "Bearer carol-token"},
    )

    assert resp.status_code == 200
    assert seen["authorization"] == "Bearer carol-token"


@pytest.mark.integration
def test_responses_callback_without_headers_parameter_is_unaffected():
    async def run_response(model: str, input_items: list[dict], body: dict):
        return "ok"

    app = FastAPI()
    app.include_router(create_responses_router(list_models=lambda: ["p"], run_response=run_response))
    resp = TestClient(app).post("/v1/responses", json={"model": "p", "input": "hi"}, headers={"X-Tenant": "acme"})

    assert resp.status_code == 200


# --- the opt-in detector ---------------------------------------------------------------------


def test_callable_accepts_kwarg_detects_explicit_parameter():
    def fn(model, messages, body, headers):
        pass

    assert callable_accepts_kwarg(fn, "headers") is True


def test_callable_accepts_kwarg_detects_var_keyword():
    def fn(model, messages, body, **kwargs):
        pass

    assert callable_accepts_kwarg(fn, "headers") is True


def test_callable_accepts_kwarg_rejects_missing_parameter():
    def fn(model, messages, body):
        pass

    assert callable_accepts_kwarg(fn, "headers") is False


def test_callable_accepts_kwarg_ignores_var_positional():
    """*args is not a way to receive a keyword argument."""

    def fn(*args):
        pass

    assert callable_accepts_kwarg(fn, "headers") is False


def test_callable_accepts_kwarg_on_unintrospectable_callable():
    """Builtins without a signature must not raise; they simply do not opt in."""
    assert callable_accepts_kwarg(print, "headers") is False
