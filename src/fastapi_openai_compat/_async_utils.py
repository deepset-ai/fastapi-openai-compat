"""Async utility helpers shared across router factories."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

from fastapi.concurrency import run_in_threadpool


def ensure_async(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a sync callable so it runs in a threadpool; return async callables as-is."""
    if inspect.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return await run_in_threadpool(fn, *args, **kwargs)

    return _wrapper


__all__ = ["ensure_async"]
