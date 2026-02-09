import pytest

from fastapi_openai_compat.models import ChatCompletion, ChatRequest, Choice, Message, ModelObject, ModelsResponse


@pytest.mark.unit
class TestChatRequest:
    def test_minimal_request(self):
        req = ChatRequest(model="test", messages=[{"role": "user", "content": "hi"}])
        assert req.model == "test"
        assert req.stream is False
        assert len(req.messages) == 1

    def test_stream_flag(self):
        req = ChatRequest(model="test", messages=[], stream=True)
        assert req.stream is True

    def test_extra_fields_allowed(self):
        req = ChatRequest(
            model="test",
            messages=[],
            temperature=0.7,
            max_tokens=100,
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 100

    def test_model_dump_includes_extras(self):
        req = ChatRequest(model="m", messages=[], top_p=0.9)
        dumped = req.model_dump()
        assert dumped["top_p"] == 0.9
        assert dumped["model"] == "m"

    def test_multimodal_messages_accepted(self):
        req = ChatRequest(
            model="vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ],
        )
        assert len(req.messages) == 1
        assert isinstance(req.messages[0]["content"], list)

    def test_tool_messages_accepted(self):
        req = ChatRequest(
            model="tool-model",
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 72F"},
            ],
        )
        assert len(req.messages) == 3

    def test_tools_and_tool_choice_forwarded(self):
        req = ChatRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
        )
        dumped = req.model_dump()
        assert "tools" in dumped
        assert dumped["tool_choice"] == "auto"

    def test_response_format_forwarded(self):
        req = ChatRequest(
            model="m",
            messages=[],
            response_format={"type": "json_object"},
        )
        assert req.response_format == {"type": "json_object"}


@pytest.mark.unit
class TestMessage:
    def test_user_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"

    def test_assistant_message(self):
        msg = Message(role="assistant", content="hi there")
        assert msg.content == "hi there"

    def test_system_role(self):
        msg = Message(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"

    def test_tool_role(self):
        msg = Message(role="tool", content="result data")
        assert msg.role == "tool"

    def test_content_can_be_none(self):
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        )
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_refusal(self):
        msg = Message(role="assistant", content=None, refusal="I cannot help with that.")
        assert msg.refusal == "I cannot help with that."

    def test_default_content_is_none(self):
        msg = Message(role="assistant")
        assert msg.content is None
        assert msg.tool_calls is None
        assert msg.refusal is None


@pytest.mark.unit
class TestChoice:
    def test_non_streaming_choice(self):
        choice = Choice(index=0, message=Message(role="assistant", content="ok"), finish_reason="stop")
        assert choice.delta is None
        assert choice.message is not None
        assert choice.finish_reason == "stop"

    def test_streaming_choice(self):
        choice = Choice(index=0, delta=Message(role="assistant", content="chunk"))
        assert choice.message is None
        assert choice.delta is not None
        assert choice.finish_reason is None

    def test_finish_reason_length(self):
        choice = Choice(index=0, message=Message(role="assistant", content="truncated"), finish_reason="length")
        assert choice.finish_reason == "length"

    def test_finish_reason_tool_calls(self):
        choice = Choice(
            index=0,
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            ),
            finish_reason="tool_calls",
        )
        assert choice.finish_reason == "tool_calls"
        assert choice.message.content is None

    def test_finish_reason_content_filter(self):
        choice = Choice(index=0, message=Message(role="assistant", content=""), finish_reason="content_filter")
        assert choice.finish_reason == "content_filter"


@pytest.mark.unit
class TestChatCompletion:
    def test_full_response(self):
        resp = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=1000,
            model="test-model",
            choices=[Choice(index=0, message=Message(role="assistant", content="done"), finish_reason="stop")],
        )
        assert resp.id == "test-123"
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1

    def test_chunk_response(self):
        resp = ChatCompletion(
            id="test-456",
            object="chat.completion.chunk",
            created=1000,
            model="test-model",
            choices=[Choice(index=0, delta=Message(role="assistant", content="hi"))],
        )
        assert resp.object == "chat.completion.chunk"

    def test_json_serialization_roundtrip(self):
        resp = ChatCompletion(
            id="rt-1",
            object="chat.completion",
            created=999,
            model="m",
            choices=[Choice(index=0, message=Message(role="assistant", content="x"), finish_reason="stop")],
        )
        json_str = resp.model_dump_json()
        restored = ChatCompletion.model_validate_json(json_str)
        assert restored.id == resp.id
        assert restored.choices[0].message.content == "x"

    def test_usage_field(self):
        resp = ChatCompletion(
            id="u-1",
            object="chat.completion",
            created=1000,
            model="m",
            choices=[Choice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert resp.usage["total_tokens"] == 15

    def test_usage_defaults_to_none(self):
        resp = ChatCompletion(
            id="u-2",
            object="chat.completion",
            created=1000,
            model="m",
            choices=[],
        )
        assert resp.usage is None
        assert resp.system_fingerprint is None

    def test_system_fingerprint(self):
        resp = ChatCompletion(
            id="sf-1",
            object="chat.completion",
            created=1000,
            model="m",
            choices=[],
            system_fingerprint="fp_abc123",
        )
        assert resp.system_fingerprint == "fp_abc123"

    def test_tool_call_response(self):
        resp = ChatCompletion(
            id="tc-1",
            object="chat.completion",
            created=1000,
            model="m",
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
        )
        json_str = resp.model_dump_json()
        restored = ChatCompletion.model_validate_json(json_str)
        assert restored.choices[0].message.content is None
        assert restored.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"
        assert restored.choices[0].finish_reason == "tool_calls"


@pytest.mark.unit
class TestModelsResponse:
    def test_models_list(self):
        resp = ModelsResponse(
            data=[
                ModelObject(id="a", name="a", object="model", created=1, owned_by="test"),
                ModelObject(id="b", name="b", object="model", created=2, owned_by="test"),
            ],
            object="list",
        )
        assert len(resp.data) == 2
        assert resp.data[0].id == "a"
        assert resp.object == "list"
