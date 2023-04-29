# Serve LM

Serve `transformers` language models using `litestar`. Mimicks `openai.create` API.

## Install

```bash
poetry use <python-path>
poetry install
```

## Run the API

```bash
poetry run api --model=databricks/dolly-v1-6b --device=cpu
```

### Call the API

```python
import serve_lm
output = serve_lm.generate("What is the capital of France?")
```
