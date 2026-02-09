import time
from collections.abc import AsyncGenerator, Callable, Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from haystack.dataclasses import StreamingChunk

from fastapi_openai_compat import ChatRequest, CompletionResult, create_openai_router

FAKE_MODELS = ["echo-pipeline", "streaming-pipeline", "async-streaming-pipeline", "str-streaming-pipeline"]


def _echo(last_message: str) -> CompletionResult:
    return f"Echo: {last_message}"


def _sync_stream(last_message: str) -> CompletionResult:
    def _gen() -> Generator[StreamingChunk | str, None, None]:
        for word in last_message.split():
            yield StreamingChunk(content=word + " ")

    return _gen()


def _async_stream(last_message: str) -> CompletionResult:
    async def _gen() -> AsyncGenerator[StreamingChunk | str, None]:
        for word in last_message.split():
            yield StreamingChunk(content=word + " ")

    return _gen()


def _str_stream(last_message: str) -> CompletionResult:
    def _gen() -> Generator[str, None, None]:
        for word in last_message.split():
            yield word + " "

    return _gen()


_HANDLERS: dict[str, Callable[[str], CompletionResult]] = {
    "echo-pipeline": _echo,
    "streaming-pipeline": _sync_stream,
    "async-streaming-pipeline": _async_stream,
    "str-streaming-pipeline": _str_stream,
}


async def fake_list_models() -> list[str]:
    return list(FAKE_MODELS)


async def fake_run_completion(model: str, messages: list[dict], body: dict) -> CompletionResult:
    last_message = messages[-1]["content"] if messages else ""
    handler = _HANDLERS.get(model)
    if handler is None:
        msg = f"Unknown model: {model}"
        raise ValueError(msg)
    return handler(last_message)


def fake_list_models_sync() -> list[str]:
    time.sleep(0)
    return list(FAKE_MODELS)


def fake_run_completion_sync(model: str, messages: list[dict], body: dict) -> CompletionResult:
    time.sleep(0)
    last_message = messages[-1]["content"] if messages else ""
    handler = _HANDLERS.get(model)
    if handler is None:
        msg = f"Unknown model: {model}"
        raise ValueError(msg)
    return handler(last_message)


@pytest.fixture()
def app() -> FastAPI:
    application = FastAPI()
    router = create_openai_router(
        list_models=fake_list_models,
        run_completion=fake_run_completion,
    )
    application.include_router(router)
    return application


@pytest.fixture()
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(app) as tc:
        yield tc


@pytest.fixture()
def app_sync() -> FastAPI:
    application = FastAPI()
    router = create_openai_router(
        list_models=fake_list_models_sync,
        run_completion=fake_run_completion_sync,
    )
    application.include_router(router)
    return application


@pytest.fixture()
def client_sync(app_sync: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(app_sync) as tc:
        yield tc


@pytest.fixture()
def app_with_hooks() -> FastAPI:
    async def pre_hook(req: ChatRequest) -> ChatRequest:
        req.messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        return req

    async def post_hook(result: CompletionResult) -> CompletionResult:
        if isinstance(result, str):
            return result.upper()
        return result

    application = FastAPI()
    router = create_openai_router(
        list_models=fake_list_models,
        run_completion=fake_run_completion,
        pre_hook=pre_hook,
        post_hook=post_hook,
    )
    application.include_router(router)
    return application


@pytest.fixture()
def client_with_hooks(app_with_hooks: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(app_with_hooks) as tc:
        yield tc


@pytest.fixture()
def app_with_sync_hooks() -> FastAPI:
    def sync_pre_hook(req: ChatRequest) -> ChatRequest:
        req.messages.insert(0, {"role": "system", "content": "injected"})
        return req

    def sync_post_hook(result: CompletionResult) -> CompletionResult:
        if isinstance(result, str):
            return result.upper()
        return result

    application = FastAPI()
    router = create_openai_router(
        list_models=fake_list_models,
        run_completion=fake_run_completion,
        pre_hook=sync_pre_hook,
        post_hook=sync_post_hook,
    )
    application.include_router(router)
    return application


@pytest.fixture()
def client_with_sync_hooks(app_with_sync_hooks: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(app_with_sync_hooks) as tc:
        yield tc
