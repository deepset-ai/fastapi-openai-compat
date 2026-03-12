"""Files API router factory."""

import logging
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from fastapi_openai_compat._async_utils import ensure_async
from fastapi_openai_compat.files.models import FileObject

logger = logging.getLogger("fastapi_openai_compat.files")

FileUploadResult = FileObject | dict[str, Any]
RunFileUploadFn = Callable[..., Any]


def create_files_router(
    *,
    run_file_upload: RunFileUploadFn,
    tags: list[str] | None = None,
) -> APIRouter:
    """
    Create a FastAPI APIRouter with OpenAI-compatible file upload endpoints.

    This router exposes:
    - `POST /v1/files`
    - `POST /files` (alias)

    The `run_file_upload` callback is invoked as:
    `(filename, content_type, content, purpose)`

    - `filename`: uploaded file name, if provided by the client.
    - `content_type`: uploaded content type, if provided by the client.
    - `content`: full file bytes.
    - `purpose`: form `purpose` value from the request.

    The callback may return a `FileObject` or a dict matching the same schema.
    """
    _run_file_upload = ensure_async(run_file_upload)
    _tags = tags or ["openai"]

    router = APIRouter()

    files_params: dict[str, Any] = {
        "response_model": FileObject,
        "tags": _tags,
        "summary": "Upload file",
        "description": (
            "Uploads a file in an OpenAI-compatible format. Intended for use with clients calling `files.create(...)`."
        ),
        "responses": {
            200: {"description": "File uploaded successfully."},
            500: {"description": "File upload callback failed or returned unsupported data."},
        },
    }

    @router.post("/v1/files", **files_params, operation_id="openai_files_create")
    @router.post("/files", **files_params, operation_id="openai_files_create_alias")
    async def files_endpoint(
        file: Annotated[UploadFile, File(...)],
        purpose: Annotated[str, Form(...)],
    ) -> FileObject:
        try:
            content = await file.read()
            result: FileUploadResult = await _run_file_upload(
                file.filename,
                file.content_type,
                content,
                purpose,
            )
        except HTTPException:
            raise
        except Exception as exc:
            error_msg = f"File upload failed: {exc!s}"
            logger.exception("Files API upload error")
            raise HTTPException(status_code=500, detail=error_msg) from exc

        if isinstance(result, FileObject):
            return result
        if isinstance(result, dict):
            return FileObject.model_validate(result)
        raise HTTPException(status_code=500, detail="Unsupported response type from run_file_upload")

    return router


__all__ = [
    "FileUploadResult",
    "RunFileUploadFn",
    "create_files_router",
]
