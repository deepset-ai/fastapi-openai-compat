# Examples

## Basic example

A minimal server with two models (`echo` and `echo-stream`) that requires no external API keys.

```bash
pip install fastapi-openai-compat "fastapi[standard]"

fastapi dev examples/basic.py
# or
python examples/basic.py
```

Test it:

```bash
# List models
curl http://localhost:8000/v1/models | python -m json.tool

# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "echo", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "echo-stream", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Haystack chat example

Wraps a Haystack `OpenAIChatGenerator` into an OpenAI-compatible API with streaming support.

```bash
pip install fastapi-openai-compat[haystack] "fastapi[standard]"
export OPENAI_API_KEY="sk-..."

fastapi dev examples/haystack_chat.py
# or
python examples/haystack_chat.py
```

Test it:

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is Haystack?"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What is Haystack?"}], "stream": true}'
```

## API documentation

Once the server is running, FastAPI automatically serves interactive API docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Using the OpenAI Python client

Both examples are fully compatible with the official OpenAI Python client:

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Non-streaming
response = client.chat.completions.create(
    model="echo",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="echo-stream",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```
