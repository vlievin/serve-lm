# Serve LM

Serve `transformers` language models using `litestar`. Mimicks `openai.create` API.

## Install

```bash
pip install git+https://github.com/vlievin/serve-lm.git
```

## Run the API

```bash
python -m serve_lm.api --model=databricks/dolly-v1-6b --device=cpu
```

### Call the API

```python
import serve_lm
output = serve_lm.generate("What is the capital of France?")
```
