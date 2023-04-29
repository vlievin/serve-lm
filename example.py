import rich
from serve_lm import generate, arguments, models


def run_completion(args: models.GenerateRequest) -> None:
    """Run a completion by calling the API."""
    completion = generate(**args.dict())
    rich.print(completion)


if __name__ == "__main__":
    args = arguments.parse_model(models.GenerateRequest)
    rich.print(args)
    run_completion(args)
