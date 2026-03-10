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
    from fastapi_openai_compat.chat_completions import create_chat_completion_router
    from fastapi_openai_compat.responses import create_responses_router

    assert callable(create_chat_completion_router)
    assert callable(create_responses_router)
