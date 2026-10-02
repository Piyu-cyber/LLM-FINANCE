import json
import time
from pathlib import Path
from typing import Any


LOG_DIR = Path("results")
LOG_FILE = LOG_DIR / "model_calls.jsonl"


def log_model_call(
    model: str,
    prompt: str,
    output: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: float | None = None,
    seed: int = 42,
) -> None:
    """Log one model call as a JSONL record."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "timestamp": time.time(),
        "model": model,
        "prompt": prompt,
        "output": output,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "seed": seed,
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    log_model_call(
        model="test-model",
        prompt="What is 2 + 2?",
        output="4",
        input_tokens=5,
        output_tokens=1,
        latency_ms=100.5,
    )

    print(f"Test log written to {LOG_FILE}")