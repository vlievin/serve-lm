# Serve LM

A simple `litestar` wrapper to serve a `transformer` language model as an API. The interface mimicks OpenAI one.

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
