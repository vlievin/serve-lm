from typing import Any
import requests


def generate(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 1.0,
    n: int = 1,
    host: str = "localhost",
    port: int = 8000,
    timeout: float = 120,
    **kwargs,  # noqa: ANN
) -> dict[str, Any]:
    """Generate completions by calling the serve-lm API."""
    url = f"http://{host}:{port}/generate"
    data = dict(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
    )
    response = requests.get(url, params=data, timeout=timeout)
    response.raise_for_status()
    return response.json()
