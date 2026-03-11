"""Files API models."""

from typing import Literal

from pydantic import Field

from fastapi_openai_compat._shared import OpenAIBaseModel


class FileObject(OpenAIBaseModel):
    """OpenAI-compatible file object returned by the Files API."""

    id: str = Field(description="Unique file identifier.")
    object: Literal["file"] = Field(default="file", description="Object type.")
    bytes: int = Field(description="Size of the uploaded file in bytes.")
    created_at: int = Field(description="Unix timestamp (seconds) when uploaded.")
    filename: str | None = Field(default=None, description="Original uploaded filename.")
    purpose: str = Field(description="Upload purpose (for example: user_data, assistants).")
    status: str = Field(default="processed", description="File processing status.")
