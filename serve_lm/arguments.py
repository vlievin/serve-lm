from __future__ import annotations

import argparse
from typing import TypeVar

import pydantic
from typing_extensions import Type


M = TypeVar("M", bound=pydantic.BaseModel)


def parse_model(model: Type[M]) -> M:
    """Parse arguments using `argparse`."""
    parser = argparse.ArgumentParser()
    for field in model.__fields__.values():
        parser.add_argument(f"--{field.name}", type=field.type_, default=field.default)

    args = parser.parse_args()
    return model(**vars(args))
