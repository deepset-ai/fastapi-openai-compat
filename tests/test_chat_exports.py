import warnings

import pytest


@pytest.mark.unit
def test_chat_router_exports_and_alias_behavior():
    from fastapi_openai_compat import create_chat_completion_router, create_openai_router

    assert callable(create_chat_completion_router)
    assert callable(create_openai_router)

    def _list_models() -> list[str]:
        return ["m"]

    def _run_completion(model: str, messages: list[dict], body: dict) -> str:
        _ = (model, messages, body)
        return "ok"

    router_new = create_chat_completion_router(
        list_models=_list_models,
        run_completion=_run_completion,
    )
    router_old = create_openai_router(
        list_models=_list_models,
        run_completion=_run_completion,
    )

    new_paths = sorted((route.path, tuple(sorted(route.methods or []))) for route in router_new.routes)
    old_paths = sorted((route.path, tuple(sorted(route.methods or []))) for route in router_old.routes)
    assert new_paths == old_paths


@pytest.mark.unit
def test_chat_and_responses_module_exports():
    from fastapi_openai_compat import MessageParam, create_models_router
    from fastapi_openai_compat.chat_completions import create_chat_completion_router
    from fastapi_openai_compat.responses import InputItem, OutputItem, create_responses_router

    assert callable(create_chat_completion_router)
    assert callable(create_responses_router)
    assert callable(create_models_router)
    assert MessageParam.__origin__ is dict
    assert InputItem.__origin__ is dict
    assert OutputItem.__origin__ is dict


@pytest.mark.unit
def test_chat_package_is_canonical_and_root_modules_reexport():
    from fastapi_openai_compat.chat_completions.models import ChatCompletion as ChatCompletionFromPackage
    from fastapi_openai_compat.chat_completions.router import (
        create_chat_completion_router as create_router_from_package,
    )
    from fastapi_openai_compat.chat_completions.streaming import (
        create_sync_streaming_response as create_streaming_from_package,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from fastapi_openai_compat.models import ChatCompletion as ChatCompletionFromRoot
        from fastapi_openai_compat.router import create_chat_completion_router as create_router_from_root
        from fastapi_openai_compat.streaming import create_sync_streaming_response as create_streaming_from_root

    assert ChatCompletionFromPackage.__module__ == "fastapi_openai_compat.chat_completions.models"
    assert create_router_from_package.__module__ == "fastapi_openai_compat.chat_completions.router"
    assert create_streaming_from_package.__module__ == "fastapi_openai_compat.chat_completions.streaming"

    assert ChatCompletionFromRoot is ChatCompletionFromPackage
    assert create_router_from_root is create_router_from_package
    assert create_streaming_from_root is create_streaming_from_package


@pytest.mark.unit
def test_root_shim_modules_emit_deprecation_warnings():
    with pytest.warns(DeprecationWarning, match="fastapi_openai_compat.models"):
        import importlib

        import fastapi_openai_compat.models

        importlib.reload(fastapi_openai_compat.models)

    with pytest.warns(DeprecationWarning, match="fastapi_openai_compat.router"):
        import fastapi_openai_compat.router

        importlib.reload(fastapi_openai_compat.router)

    with pytest.warns(DeprecationWarning, match="fastapi_openai_compat.streaming"):
        import fastapi_openai_compat.streaming

        importlib.reload(fastapi_openai_compat.streaming)
