"""Shared types and helpers used across router modules."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

ChunkMapper = Callable[[Any], str]
PreHook = Callable[..., Any]
PostHook = Callable[..., Any]


class OpenAIBaseModel(BaseModel):
    """Base model that allows extra fields, matching OpenAI's permissive request schema."""

    model_config = ConfigDict(extra="allow")


def default_chunk_mapper(chunk: Any) -> str:
    """
    Default chunk-to-string mapper.

    Handles plain ``str`` chunks, objects with a ``.content`` attribute
    (e.g. Haystack ``StreamingChunk``), and falls back to ``str(chunk)``.
    """
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "content"):
        content = chunk.content
        return content if isinstance(content, str) else str(content) if content is not None else ""
    return str(chunk)


def _is_custom_event(chunk: Any) -> bool:
    """Check if a chunk is a custom SSE event via duck typing (.to_event_dict())."""
    return callable(getattr(chunk, "to_event_dict", None))


__all__ = ["ChunkMapper", "OpenAIBaseModel", "PostHook", "PreHook", "default_chunk_mapper"]
