"""Responses API models."""

from typing import Any, Literal

from pydantic import Field

from fastapi_openai_compat._shared import OpenAIBaseModel

InputItem = dict[str, Any]
"""A single normalized input item in OpenAI Responses API format.

This is a ``dict`` alias -- not a Pydantic model -- so that the library
stays forward-compatible when OpenAI adds new item types.  The library
validates the outer list structure; individual item contents are passed
through to your ``run_response`` callback as-is.

Common item shapes (discriminated by ``type``)::

    # User message
    {"type": "message", "role": "user", "content": "Hello"}
    {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "Describe this image"},
        {"type": "input_image", "image_url": "https://..."},
    ]}

    # Function call output (tool result returned by the client)
    {"type": "function_call_output", "call_id": "call_abc", "output": "72°F"}

See the `OpenAI Responses API reference
<https://platform.openai.com/docs/api-reference/responses/create>`_
for the full set of input item types.
"""

OutputItem = dict[str, Any]
"""A single output item in a Responses API ``Response``.

Same design rationale as :data:`InputItem` -- a ``dict`` alias for
forward-compatibility.

Common output shapes::

    # Assistant text message
    {"type": "message", "role": "assistant", "content": [...]}

    # Function call
    {"type": "function_call", "call_id": "call_abc",
     "name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}

See the `OpenAI Responses API reference
<https://platform.openai.com/docs/api-reference/responses/object>`_
for the full set of output item types.
"""


class ResponseRequest(OpenAIBaseModel):
    """Incoming OpenAI Responses API request."""

    model: str = Field(description="Model ID used to generate the response.")
    input: str | list[InputItem] | None = Field(
        default=None,
        description="Either a text shorthand, explicit input items, or omitted follow-up input.",
    )
    instructions: str | None = Field(
        default=None,
        description="Optional system/developer instructions for this response.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response as named server-sent events.",
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="Tool definitions available to the model (e.g. function tools).",
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None,
        description="Tool choice mode or explicit tool choice object.",
    )
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter.")
    top_logprobs: int | None = Field(default=None, description="Top logprobs count for token positions.")
    temperature: float | None = Field(default=None, description="Sampling temperature.")
    max_output_tokens: int | None = Field(default=None, description="Maximum output tokens.")
    previous_response_id: str | None = Field(default=None, description="Previous response ID for conversation state.")
    parallel_tool_calls: bool | None = Field(default=None, description="Allow or disallow parallel tool calls.")
    stream_options: dict[str, Any] | None = Field(default=None, description="Streaming response options.")
    text: dict[str, Any] | None = Field(default=None, description="Text output configuration.")
    reasoning: dict[str, Any] | None = Field(default=None, description="Reasoning model options.")
    store: bool | None = Field(default=None, description="Whether to store generated response.")
    service_tier: str | None = Field(default=None, description="Requested service tier.")
    metadata: dict[str, Any] | None = Field(default=None, description="Arbitrary metadata map.")


class ResponseOutputText(OpenAIBaseModel):
    """Text content part in an output message."""

    type: Literal["output_text"] = "output_text"
    text: str = Field(description="Generated output text.")
    annotations: list[dict[str, Any]] = Field(default_factory=list, description="Text annotations.")


class ResponseOutputMessage(OpenAIBaseModel):
    """Message output item in a response."""

    id: str = Field(description="Unique message item identifier.")
    type: Literal["message"] = "message"
    status: str = Field(default="completed", description="Item status.")
    role: Literal["assistant"] = "assistant"
    content: list[dict[str, Any]] = Field(description="Message content parts.")


class ResponseFunctionCall(OpenAIBaseModel):
    """Function call output item in a response."""

    type: Literal["function_call"] = "function_call"
    id: str = Field(description="Unique function call item identifier.")
    call_id: str = Field(description="Call identifier used by function call outputs.")
    name: str = Field(description="Function name.")
    arguments: str = Field(description="JSON-encoded function arguments.")
    status: str = Field(default="completed", description="Item status.")


class Response(OpenAIBaseModel):
    """Top-level OpenAI Responses API object."""

    id: str = Field(description="Unique response identifier.")
    object: Literal["response"] = Field(default="response", description="Object type.")
    created_at: int = Field(description="Unix timestamp (seconds) when created.")
    status: str = Field(default="completed", description="Response status.")
    model: str = Field(description="Model that generated the response.")
    output: list[OutputItem] = Field(description="Output items (messages, tool calls, etc.).")
    usage: dict[str, Any] | None = Field(default=None, description="Token usage details.")
    error: dict[str, Any] | None = Field(default=None, description="Error payload if generation failed.")
    incomplete_details: dict[str, Any] | None = Field(
        default=None,
        description="Details explaining incomplete responses.",
    )


__all__ = [
    "InputItem",
    "OutputItem",
    "Response",
    "ResponseFunctionCall",
    "ResponseOutputMessage",
    "ResponseOutputText",
    "ResponseRequest",
]
