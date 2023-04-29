from typing import Optional
import pydantic


class GenerateRequest(pydantic.BaseModel):
    """Request to the generate API."""

    prompt: str
    max_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 3


class GenerateChoice(pydantic.BaseModel):
    """Models a single generation sample."""

    text: str
    index: int
    logprobs: str
    finish_reason: str


class GenerateUsage(pydantic.BaseModel):
    """Models the token usage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(pydantic.BaseModel):
    """Response from the generate API."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[GenerateChoice]
    usage: Optional[GenerateUsage] = None
