import json
from pathlib import Path


INPUT_PATH = Path("data/finqa/train_chunks.jsonl")
OUTPUT_PATH = Path("data/finqa/contextual_chunks.jsonl")

WINDOW_SIZE = 2


def load_chunks():
    chunks = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def build_contextual_chunks(chunks):
    output = []

    # Group chunks belonging to the same FinQA example
    examples = {}

    for chunk in chunks:
        example_id = chunk["example_id"]

        if example_id not in examples:
            examples[example_id] = []

        examples[example_id].append(chunk)

    # Build sliding-window chunks
    for example_id, example_chunks in examples.items():

        for i in range(len(example_chunks)):

            start = max(0, i - WINDOW_SIZE + 1)
            end = min(len(example_chunks), i + 1)

            selected = example_chunks[start:end]

            text = " ".join(
                chunk["text"]
                for chunk in selected
            )

            output.append({
                "example_id": example_id,
                "chunk_id": f"context_{i}",
                "source": "contextual",
                "position": i,
                "text": text,
                "original_chunk_ids": [
                    chunk["chunk_id"]
                    for chunk in selected
                ],
            })

    return output


def save_chunks(chunks):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def main():

    print("Loading original chunks...")

    chunks = load_chunks()

    print(f"Original chunks: {len(chunks)}")

    print("Building contextual chunks...")

    contextual_chunks = build_contextual_chunks(chunks)

    print(f"Contextual chunks: {len(contextual_chunks)}")

    save_chunks(contextual_chunks)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()