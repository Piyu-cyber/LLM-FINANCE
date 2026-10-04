import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CORPUS_PATH = Path("data/finqa/train_chunks.jsonl")
EMBEDDINGS_PATH = Path("data/finqa/train_embeddings.npy")

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_corpus() -> list[dict]:
    chunks = []

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def main() -> None:
    print("Loading corpus...")

    chunks = load_corpus()

    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME, device="cuda")

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"Embedding shape: {embeddings.shape}")

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()