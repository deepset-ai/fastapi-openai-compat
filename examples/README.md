# Examples

## Table of contents

- **Chat Completions API**
  - [Basic echo server](#basic-echo-server) -- no API keys needed
  - [Haystack chat](#haystack-chat) -- real LLM via Haystack
- **Responses API**
  - [Basic responses server](#basic-responses-server) -- text, streaming, function calls
  - [Responses with files](#responses-with-files) -- file upload + `input_file.file_id`
- [API documentation](#api-documentation)
- [Using the OpenAI Python client](#using-the-openai-python-client)

---

## Chat Completions API

### Basic echo server

[`basic.py`](basic.py) -- A minimal server with four models (`echo`, `echo-stream`, `echo-metadata`, and `echo-events`) that requires no external API keys.

```bash
pip install fastapi-openai-compat "fastapi[standard]"

python examples/basic.py
```

Test with curl:

```bash
# List models
curl http://localhost:8000/v1/models | jq

# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "echo", "messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "echo-stream", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'

# Metadata in request and response
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "echo-metadata", "messages": [{"role": "user", "content": "Hello!"}], "metadata": {"request_id": "abc-123"}}'

# Streaming with custom SSE events (e.g. Open WebUI status updates)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "echo-events", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

### Haystack chat

[`haystack_chat.py`](haystack_chat.py) -- Wraps a Haystack `OpenAIChatGenerator` into an OpenAI-compatible API with streaming support.

```bash
pip install fastapi-openai-compat[haystack] "fastapi[standard]"
export OPENAI_API_KEY="sk-..."

python examples/haystack_chat.py
```

Test with curl:

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

---

## Responses API

### Basic responses server

[`responses_basic.py`](responses_basic.py) -- Minimal Responses API server with:

- non-streaming text outputs
- streaming text outputs
- streaming and non-streaming function call outputs

```bash
pip install fastapi-openai-compat "fastapi[standard]"

python examples/responses_basic.py
```

**E2e client** ([`responses_basic_client.py`](responses_basic_client.py)) -- exercises every model using the official OpenAI Python client:

```bash
pip install openai
python examples/responses_basic_client.py
```

Or test with curl:

```bash
# List models
curl http://localhost:8000/v1/models | jq

# Non-streaming response
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "responses-echo", "input": "Hello!"}'

# Streaming response (SSE)
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "responses-stream", "input": "Hello from streaming", "stream": true}'

# Streaming function call events
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "responses-function", "input": "Weather in Paris?", "stream": true}'
```

### Responses with files

[`responses_with_files.py`](responses_with_files.py) -- Responses API server using `create_files_router(...)` with local-disk persistence for demos of:

- `client.files.create(...)`
- `input_file.file_id` in `client.responses.create(...)`

Uploaded files are stored in `examples/.uploaded_files/`.

```bash
pip install fastapi-openai-compat "fastapi[standard]"

python examples/responses_with_files.py
```

**E2e client** ([`responses_with_files_client.py`](responses_with_files_client.py)) -- uploads a file and sends it with a text instruction, all via the official OpenAI Python client:

```bash
pip install openai
python examples/responses_with_files_client.py
```

Or test with curl (requires jq):

```bash
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
```

---

## API documentation

Once the server is running, FastAPI automatically serves interactive API docs:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Using the OpenAI Python client

All examples are compatible with the official OpenAI Python client:

```bash
pip install openai
```

**Chat Completions:**

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

**Responses API:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Non-streaming
response = client.responses.create(
    model="responses-echo",
    input="Hello!",
)
print(response.output[0].content[0].text)

# Streaming
stream = client.responses.create(
    model="responses-stream",
    input="Hello from streaming",
    stream=True,
)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print()
```
