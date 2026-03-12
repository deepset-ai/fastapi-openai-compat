"""
Client for the responses_with_files.py server.

Demonstrates file upload and file_id usage in the Responses API using the
official OpenAI Python client.

Prerequisites:
    pip install openai

Usage:
    1. Start the server in one terminal:
       python examples/responses_with_files.py

    2. Run this client in another terminal:
       python examples/responses_with_files_client.py
"""

from pathlib import Path

from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
client = OpenAI(base_url=BASE_URL, api_key="unused")

README_PATH = Path(__file__).resolve().parent.parent / "README.md"


def demo_list_models() -> None:
    print("=== List models ===")
    models = client.models.list()
    for m in models.data:
        print(f"  - {m.id}")
    print()


def demo_upload_and_respond() -> None:
    print("=== Upload file + Responses API with file_id ===")

    print(f"  Uploading {README_PATH.name}...")
    with README_PATH.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")
    print(f"  File ID: {uploaded.id}")
    print(f"  Filename: {uploaded.filename}")
    print(f"  Bytes: {uploaded.bytes}")
    print()

    print("  Sending response request with file_id + instruction...")
    response = client.responses.create(
        model="responses-files",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": uploaded.id},
                    {"type": "input_text", "text": "Summarize this file briefly."},
                ],
            }
        ],
    )
    text = response.output[0].content[0].text
    print(f"  Response: {text}")
    print()


def demo_no_file() -> None:
    print("=== Responses API without file (plain text) ===")
    response = client.responses.create(
        model="responses-files",
        input="Just a text message, no files.",
    )
    text = response.output[0].content[0].text
    print(f"  Response: {text}")
    print()


if __name__ == "__main__":
    demo_list_models()
    demo_upload_and_respond()
    demo_no_file()
    print("All demos completed.")
