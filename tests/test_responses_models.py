import time

import pytest

from fastapi_openai_compat.responses.models import (
    Response,
    ResponseFunctionCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseRequest,
)


@pytest.mark.unit
class TestResponseRequest:
    def test_minimal_request(self):
        req = ResponseRequest(model="gpt-4", input="Hello")
        assert req.model == "gpt-4"
        assert req.input == "Hello"
        assert req.stream is False
        assert req.instructions is None
        assert req.tools is None

    def test_request_with_input_items(self):
        items = [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "system", "content": "Be helpful"},
        ]
        req = ResponseRequest(model="gpt-4", input=items)
        assert req.input == items

    def test_extra_fields_allowed(self):
        req = ResponseRequest(model="gpt-4", input="Hi", temperature=0.5, top_p=0.9)
        body = req.model_dump()
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.9

    def test_tools_and_tool_choice_forwarded(self):
        tools = [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}, "strict": True}]
        req = ResponseRequest(model="gpt-4", input="Weather?", tools=tools, tool_choice="auto")
        assert req.tools == tools
        assert req.tool_choice == "auto"


@pytest.mark.unit
class TestResponseOutput:
    def test_output_text_defaults(self):
        out = ResponseOutputText(text="Hello world")
        assert out.type == "output_text"
        assert out.text == "Hello world"
        assert out.annotations == []

    def test_output_text_with_annotations(self):
        annotations = [{"type": "url_citation", "url": "https://example.com", "title": "Example"}]
        out = ResponseOutputText(text="See [1]", annotations=annotations)
        assert out.annotations == annotations

    def test_output_message(self):
        msg = ResponseOutputMessage(
            id="msg_123",
            content=[{"type": "output_text", "text": "Hi", "annotations": []}],
        )
        assert msg.type == "message"
        assert msg.role == "assistant"
        assert msg.status == "completed"
        assert msg.id == "msg_123"

    def test_function_call_item(self):
        fc = ResponseFunctionCall(
            id="fc_123",
            call_id="call_abc",
            name="get_weather",
            arguments='{"location": "Boston"}',
        )
        assert fc.type == "function_call"
        assert fc.status == "completed"
        assert fc.name == "get_weather"


@pytest.mark.unit
class TestResponseObject:
    def test_response_minimal(self):
        resp = Response(
            id="resp_123",
            created_at=int(time.time()),
            model="gpt-4",
            output=[{"type": "message", "id": "msg_1", "role": "assistant", "content": []}],
        )
        assert resp.object == "response"
        assert resp.status == "completed"
        assert resp.error is None

    def test_response_allows_extra_fields(self):
        resp = Response(
            id="resp_123",
            created_at=1000,
            model="gpt-4",
            output=[],
            reasoning={"effort": "high", "summary": None},
        )
        body = resp.model_dump()
        assert body["reasoning"] == {"effort": "high", "summary": None}
