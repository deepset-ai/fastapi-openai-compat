"""Chat Completions models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fastapi_openai_compat._shared import OpenAIBaseModel

MessageParam = dict[str, Any]
"""A single chat message in OpenAI Chat Completions format.

This is a ``dict`` alias -- not a Pydantic model -- so that the library
stays forward-compatible when OpenAI adds new message shapes.  The
library validates the outer list structure; individual message contents
are passed through to your ``run_completion`` callback as-is.

Common message shapes::

    # Plain text
    {"role": "user", "content": "Hello!"}
    {"role": "system", "content": "You are a helpful assistant."}

    # Multimodal content parts
    {"role": "user", "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://..."}},
    ]}

    # Tool result
    {"role": "tool", "tool_call_id": "call_abc", "content": "72°F"}

See the `OpenAI Chat Completions API reference
<https://platform.openai.com/docs/api-reference/chat/create>`_
for the full message specification.
"""


class ModelObject(BaseModel):
    """Represents a single model in the OpenAI /models response."""

    id: str = Field(description="Unique identifier for the model.")
    name: str = Field(description="Human-readable name of the model.")
    object: Literal["model"] = Field(description="Object type, always `model`.")
    created: int = Field(description="Unix timestamp (seconds) when the model was created.")
    owned_by: str = Field(description="Organization or entity that owns the model.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "my-rag-pipeline",
                    "name": "my-rag-pipeline",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "custom",
                }
            ]
        }
    )


class ModelsResponse(BaseModel):
    """Response schema for the OpenAI-compatible `/v1/models` endpoint."""

    data: list[ModelObject] = Field(description="List of available models.")
    object: Literal["list"] = Field(description="Object type, always `list`.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "data": [
                        {
                            "id": "my-rag-pipeline",
                            "name": "my-rag-pipeline",
                            "object": "model",
                            "created": 1700000000,
                            "owned_by": "custom",
                        },
                        {
                            "id": "my-chat-pipeline",
                            "name": "my-chat-pipeline",
                            "object": "model",
                            "created": 1700000000,
                            "owned_by": "custom",
                        },
                    ],
                    "object": "list",
                }
            ]
        }
    )


class ChatRequest(OpenAIBaseModel):
    """
    Incoming chat completion request in OpenAI format.

    Any additional fields (e.g. ``temperature``, ``max_tokens``, ``top_p``,
    ``tools``, ``tool_choice``, ``response_format``) are accepted and
    forwarded in the ``body`` dict passed to ``run_completion``.

    Messages support all OpenAI content types including plain text,
    multimodal content parts (images, audio), and tool call results.
    """

    model: str = Field(description="The model (pipeline) name to use for completion.")
    messages: list[MessageParam] = Field(
        description=(
            "A list of messages comprising the conversation so far. "
            "Each message is a dict with at least `role` and `content` keys. "
            "Content can be a string or an array of content parts for multimodal inputs "
            "(text, image_url, input_audio)."
        ),
    )
    stream: bool = Field(
        default=False,
        description="If true, the response is streamed back as server-sent events (SSE).",
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "model": "my-rag-pipeline",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What is Haystack?"},
                    ],
                    "stream": False,
                },
                {
                    "model": "my-chat-pipeline",
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": 512,
                },
                {
                    "model": "my-vision-pipeline",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this image?"},
                                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                            ],
                        }
                    ],
                },
            ]
        },
    )


class Message(OpenAIBaseModel):
    """
    A single message in a chat completion response.

    Supports text content, tool calls, and safety refusals per the OpenAI spec.
    """

    role: str = Field(description="The role of the message author (e.g. assistant, user, system, tool).")
    content: str | None = Field(
        default=None,
        description="The text content of the message. Null when the assistant response is a tool call only.",
    )
    reasoning_content: str | None = Field(
        default=None,
        description="Reasoning content from the model (e.g. chain-of-thought).",
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Tool calls generated by the model, if any.",
    )
    refusal: str | None = Field(
        default=None,
        description="The refusal message if the model refused to respond.",
    )


class Choice(OpenAIBaseModel):
    """A single choice in a chat completion response."""

    index: int = Field(description="Zero-based index of this choice in the choices array.")
    delta: Message | None = Field(
        default=None,
        description="A partial message delta (present only in streaming chunks).",
    )
    finish_reason: str | None = Field(
        default=None,
        description=(
            "The reason the model stopped generating: "
            "`stop` (natural end), `length` (max tokens), "
            "`tool_calls` (tool invocation), or `content_filter`."
        ),
    )
    logprobs: None | dict = Field(
        default=None,
        description="Log probability information (not currently populated).",
    )
    message: Message | None = Field(
        default=None,
        description="The complete assistant message (present only in non-streaming responses).",
    )


class ChatCompletion(OpenAIBaseModel):
    """Chat completion response, used for both full and streamed (chunk) responses."""

    id: str = Field(description="Unique identifier for this completion.")
    object: Literal["chat.completion"] | Literal["chat.completion.chunk"] = Field(
        description="Object type: `chat.completion` for full responses, `chat.completion.chunk` for streamed chunks.",
    )
    created: int = Field(description="Unix timestamp (seconds) when the completion was created.")
    model: str = Field(description="The model (pipeline) that generated the completion.")
    choices: list[Choice] = Field(description="List of completion choices.")
    usage: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Token usage statistics for the request. Contains `prompt_tokens`, `completion_tokens`, and `total_tokens`."
        ),
    )
    system_fingerprint: str | None = Field(
        default=None,
        description="Backend configuration fingerprint for determinism tracking.",
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "id": "my-rag-pipeline-a1b2c3d4",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "my-rag-pipeline",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Haystack is an open-source AI framework."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "total_tokens": 20,
                    },
                }
            ]
        },
    )


__all__ = [
    "ChatCompletion",
    "ChatRequest",
    "Choice",
    "Message",
    "MessageParam",
    "ModelObject",
    "ModelsResponse",
    "OpenAIBaseModel",
]
