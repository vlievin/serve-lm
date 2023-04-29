import time
from typing import Any
import pydantic

import torch
import litestar
import transformers
import uvicorn
from loguru import logger
from serve_lm import arguments, models
import uuid


class Args(pydantic.BaseModel):
    """Arguments for the API."""

    model: str = "sshleifer/tiny-gpt2"
    device: str = "cpu"
    compile: bool = False
    host: str = "localhost"
    port: int = 8000
    reload_api: bool = False


# init the model
args = arguments.parse_model(Args)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model)
model.eval()
model.to(args.device)
if args.compile:
    model = torch.compile(model)


@litestar.get("/")
async def ping() -> dict[str, str]:
    """Ping the server."""
    return {"status": "OK"}


@litestar.get("/generate")
async def generate(query: models.GenerateRequest) -> models.GenerateResponse:
    """Generate text."""
    logger.info(f"Generating text for query: {query.prompt}")
    input_ids: torch.Tensor = tokenizer.encode(query.prompt, return_tensors="pt")  # type: ignore
    outputs = _generate(input_ids, query)
    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {
        "id": str(uuid.uuid4()),
        "created": int(time.time()),
        "model": args.model,
        "choices": [
            {"text": completion, "index": i, "logprobs": "", "finish_reason": "length"}
            for i, completion in enumerate(completions)
        ],
        "usage": _compute_usage(input_ids, outputs),
    }  # type: ignore


def _compute_usage(input_ids: torch.Tensor, outputs: torch.Tensor) -> dict[str, Any]:
    """Compute usage statistics."""
    return {
        "prompt_tokens": input_ids.shape[1],
        "completion_tokens": outputs.shape[0] * (outputs.shape[1] - input_ids.shape[1]),
        "total_tokens": outputs.shape[0] * outputs.shape[1],
    }


@torch.inference_mode()
def _generate(input_ids: torch.Tensor, query: models.GenerateRequest) -> torch.Tensor:
    input_ids.to(args.device)
    input_ids = input_ids.repeat(query.n, 1)
    outputs = model.generate(
        input_ids,
        max_length=query.max_tokens,
        do_sample=query.temperature > 0,
        temperature=query.temperature,
        top_p=query.top_p,
        top_k=query.n,
    )
    return outputs


def run() -> None:
    """Run the server."""
    logger.info("Starting server...")
    app = litestar.Litestar(route_handlers=[ping, generate])
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload_api)


if __name__ == "__main__":
    run()
