import json
from pathlib import Path
from typing import Any

from finqa_loader import load_finqa
from chunker import create_chunks


OUTPUT_PATH = Path("data/finqa/train_chunks.jsonl")


def build_corpus() -> None:
    data = load_finqa()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for example in data:
            chunks = create_chunks(example)

            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"FinQA examples: {len(data)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_corpus()