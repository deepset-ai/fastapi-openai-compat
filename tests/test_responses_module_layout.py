import pytest

from fastapi_openai_compat.responses.models import Response, ResponseRequest
from fastapi_openai_compat.responses.router import create_responses_router
from fastapi_openai_compat.responses.streaming import create_responses_streaming_response, response_from_text


@pytest.mark.unit
def test_responses_package_is_canonical_and_legacy_modules_removed():
    import importlib

    assert Response.__module__ == "fastapi_openai_compat.responses.models"
    assert ResponseRequest.__module__ == "fastapi_openai_compat.responses.models"
    assert create_responses_router.__module__ == "fastapi_openai_compat.responses.router"
    assert create_responses_streaming_response.__module__ == "fastapi_openai_compat.responses.streaming"
    assert response_from_text.__module__ == "fastapi_openai_compat.responses.streaming"

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fastapi_openai_compat.responses_models")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fastapi_openai_compat.responses_router")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fastapi_openai_compat.responses_streaming")
