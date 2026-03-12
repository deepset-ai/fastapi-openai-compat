"""
Client for the responses_basic.py server.

Demonstrates every model exposed by the basic Responses API example using the
official OpenAI Python client.

Prerequisites:
    pip install openai

Usage:
    1. Start the server in one terminal:
       python examples/responses_basic.py

    2. Run this client in another terminal:
       python examples/responses_basic_client.py
"""

from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
client = OpenAI(base_url=BASE_URL, api_key="unused")


def demo_list_models() -> None:
    print("=== List models ===")
    models = client.models.list()
    for m in models.data:
        print(f"  - {m.id}")
    print()


def demo_non_streaming_text() -> None:
    print("=== Non-streaming text (responses-echo) ===")
    response = client.responses.create(
        model="responses-echo",
        input="Hello from the OpenAI client!",
    )
    text = response.output[0].content[0].text
    print(f"  Response: {text}")
    print()


def demo_streaming_text() -> None:
    print("=== Streaming text (responses-stream) ===")
    stream = client.responses.create(
        model="responses-stream",
        input="Hello from streaming",
        stream=True,
    )
    print("  Response: ", end="")
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    print("\n")


def demo_non_streaming_function_call() -> None:
    print("=== Non-streaming function call (responses-function) ===")
    response = client.responses.create(
        model="responses-function",
        input="Weather in Paris?",
    )
    item = response.output[0]
    print(f"  Type: {item.type}")
    print(f"  Function: {item.name}")
    print(f"  Arguments: {item.arguments}")
    print(f"  Call ID: {item.call_id}")
    print()


def demo_streaming_function_call() -> None:
    print("=== Streaming function call (responses-function) ===")
    stream = client.responses.create(
        model="responses-function",
        input="Weather in Rome?",
        stream=True,
    )
    print("  Events:")
    for event in stream:
        if event.type == "response.function_call_arguments.delta":
            print(f"    delta: {event.delta!r}")
        elif event.type == "response.function_call_arguments.done":
            print(f"    done:  {event.arguments}")
        elif event.type == "response.output_item.done":
            print(f"    output_item.done: type={event.item.type}, name={event.item.name}")
    print()


if __name__ == "__main__":
    demo_list_models()
    demo_non_streaming_text()
    demo_streaming_text()
    demo_non_streaming_function_call()
    demo_streaming_function_call()
    print("All demos completed.")
