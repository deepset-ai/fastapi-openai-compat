import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fastapi_openai_compat.files import FileObject, create_files_router


@pytest.mark.integration
def test_files_upload_sync_callback_v1_and_alias():
    callback_calls: list[dict] = []

    def run_file_upload(filename: str | None, content_type: str | None, content: bytes, purpose: str):
        callback_calls.append(
            {
                "filename": filename,
                "content_type": content_type,
                "content": content,
                "purpose": purpose,
            }
        )
        return {
            "id": f"file_{len(callback_calls)}",
            "object": "file",
            "bytes": len(content),
            "created_at": int(time.time()),
            "filename": filename or "unknown",
            "purpose": purpose,
            "status": "processed",
        }

    app = FastAPI()
    app.include_router(create_files_router(run_file_upload=run_file_upload))

    with TestClient(app) as client:
        resp = client.post(
            "/v1/files",
            data={"purpose": "user_data"},
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "file_1"
        assert data["object"] == "file"
        assert data["bytes"] == 11
        assert data["filename"] == "notes.txt"
        assert data["purpose"] == "user_data"
        assert data["status"] == "processed"

        alias_resp = client.post(
            "/files",
            data={"purpose": "assistants"},
            files={"file": ("a.txt", b"abc", "text/plain")},
        )
        assert alias_resp.status_code == 200
        alias_data = alias_resp.json()
        assert alias_data["id"] == "file_2"
        assert alias_data["bytes"] == 3
        assert alias_data["purpose"] == "assistants"

    assert len(callback_calls) == 2
    assert callback_calls[0]["filename"] == "notes.txt"
    assert callback_calls[0]["content_type"] == "text/plain"
    assert callback_calls[0]["content"] == b"hello world"


@pytest.mark.integration
def test_files_upload_async_callback_can_return_model():
    async def run_file_upload(filename: str | None, _content_type: str | None, content: bytes, purpose: str):
        return FileObject(
            id="file_async_1",
            object="file",
            bytes=len(content),
            created_at=1700000000,
            filename=filename or "unknown",
            purpose=purpose,
            status="processed",
        )

    app = FastAPI()
    app.include_router(create_files_router(run_file_upload=run_file_upload))

    with TestClient(app) as client:
        resp = client.post(
            "/v1/files",
            data={"purpose": "user_data"},
            files={"file": ("async.txt", b"xyz", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "file_async_1"
        assert data["filename"] == "async.txt"
        assert data["bytes"] == 3
