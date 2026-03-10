import pytest


@pytest.mark.unit
def test_responses_api_exports():
    from fastapi_openai_compat import (
        Response,
        ResponseFunctionCall,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseRequest,
        create_responses_router,
    )

    assert callable(create_responses_router)
    assert Response is not None
    assert ResponseRequest is not None
    assert ResponseOutputText is not None
    assert ResponseOutputMessage is not None
    assert ResponseFunctionCall is not None
