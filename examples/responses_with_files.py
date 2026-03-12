"""
Responses API example with file upload and file_id input flow.

This example stores uploaded files on local disk and exposes:
- POST /v1/files (via create_files_router callback)
- POST /v1/responses
- GET /v1/models (via create_models_router)

It demonstrates OpenAI client compatibility for:
1) file upload via files.create(...)
2) passing input_file with file_id to responses.create(...)

Run the server:
    pip install fastapi-openai-compat "fastapi[standard]"  # includes uvicorn
    python examples/responses_with_files.py

Run the e2e client (in a second terminal):
    pip install openai
    python examples/responses_with_files_client.py

Or test with curl (requires jq):
    FILE_ID=$(curl -s http://localhost:8000/v1/files \
      -F "purpose=user_data" \
      -F "file=@README.md" | jq -r '.id')

    curl http://localhost:8000/v1/responses \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"responses-files\",
        \"input\": [{
          \"role\": \"user\",
          \"content\": [
            {\"type\": \"input_file\", \"file_id\": \"${FILE_ID}\"},
            {\"type\": \"input_text\", \"text\": \"Summarize this file briefly.\"}
          ]
        }]
      }" | jq
"""

import time
from collections.abc import Generator
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from fastapi_openai_compat import (
    InputItem,
    ResponseResult,
    create_files_router,
    create_models_router,
    create_responses_router,
)

MODEL = "responses-files"
FILES: dict[str, dict] = {}
UPLOAD_DIR = Path(__file__).resolve().parent / ".uploaded_files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def list_models() -> list[str]:
    return [MODEL]


def run_response(model: str, input_items: list[InputItem], body: dict) -> ResponseResult:
    _ = body
    references: list[str] = []
    instructions: list[str] = []

    for part in _iter_content_parts(input_items):
        if part.get("type") == "input_text":
            text = str(part.get("text", "")).strip()
            if text:
                instructions.append(text)
            continue

        file_ref = _file_reference(part)
        if file_ref is not None:
            references.append(file_ref)

    if not references:
        return "No input_file items found. Send input_file with file_id, file_url, or file_data."

    instruction = instructions[0] if instructions else "No instruction text provided."
    refs = "; ".join(references)
    return f"Model {model} received files: {refs}. Instruction: {instruction}"


def _iter_content_parts(input_items: list[InputItem]) -> Generator[dict, None, None]:
    for item in input_items:
        content = item.get("content")
        if not isinstance(content, list):
            continue
        yield from content


def _file_reference(part: dict) -> str | None:
    if part.get("type") != "input_file":
        return None

    if "file_id" in part:
        file_id = str(part["file_id"])
        meta = FILES.get(file_id)
        if meta is None:
            return f"{file_id} (unknown)"
        return f"{file_id} ({meta['filename']}, {meta['bytes']} bytes, {meta['path']})"

    if "file_url" in part:
        return f"url:{part['file_url']}"

    if "filename" in part and "file_data" in part:
        return f"inline:{part['filename']}"

    return "input_file:unresolved"


app = FastAPI(title="Responses API Files Example")


def run_file_upload(filename: str | None, _content_type: str | None, content: bytes, purpose: str) -> dict:
    file_id = f"file_{len(FILES) + 1}"
    safe_filename = Path(filename or "unknown").name
    suffix = Path(safe_filename).suffix
    stored_path = UPLOAD_DIR / f"{file_id}{suffix}"
    stored_path.write_bytes(content)

    FILES[file_id] = {
        "filename": safe_filename,
        "purpose": purpose,
        "bytes": len(content),
        "path": str(stored_path),
        "stored_at": int(time.time()),
    }
    return {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": int(time.time()),
        "filename": safe_filename,
        "purpose": purpose,
        "status": "processed",
    }


app.include_router(create_files_router(run_file_upload=run_file_upload))
app.include_router(create_models_router(list_models=list_models))
app.include_router(
    create_responses_router(
        list_models=list_models,
        run_response=run_response,
        include_models_endpoints=False,
    )
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104
