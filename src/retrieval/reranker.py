import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel


CORPUS_PATH = Path("data/finqa/train_chunks.jsonl")

MODEL_NAME = "colbert-ir/colbertv2.0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_QUERY_LENGTH = 64
MAX_DOC_LENGTH = 256


def load_corpus():
    chunks = []

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                chunks.append(json.loads(line))

    return chunks


class ColBERTReranker:
    def __init__(self):
        print(f"Loading reranker model: {MODEL_NAME}")
        print(f"Device: {DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            MODEL_NAME
        ).to(DEVICE)

        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, max_length):

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = self.model(**encoded)

        token_embeddings = outputs.last_hidden_state

        # L2 normalize token embeddings.
        token_embeddings = torch.nn.functional.normalize(
            token_embeddings,
            p=2,
            dim=-1,
        )

        return token_embeddings, encoded["attention_mask"]

    @torch.no_grad()
    def score(self, query, documents):

        query_embeddings, query_mask = self.encode(
            [query],
            MAX_QUERY_LENGTH,
        )

        document_embeddings, document_mask = self.encode(
            documents,
            MAX_DOC_LENGTH,
        )

        # Remove batch dimension from query.
        query_embeddings = query_embeddings[0]
        query_mask = query_mask[0]

        # Only real query tokens.
        query_embeddings = query_embeddings[
            query_mask.bool()
        ]

        scores = []

        for i in range(len(documents)):

            doc_embeddings = document_embeddings[i]
            doc_mask = document_mask[i]

            # Only real document tokens.
            doc_embeddings = doc_embeddings[
                doc_mask.bool()
            ]

            # Query-token × document-token similarity.
            similarity = torch.matmul(
                query_embeddings,
                doc_embeddings.transpose(0, 1),
            )

            # MaxSim:
            # for each query token, take its best
            # matching document token.
            max_similarity = similarity.max(
                dim=1
            ).values

            # Sum over query tokens.
            score = max_similarity.sum()

            scores.append(float(score))

        return scores


def rerank(query, candidates, reranker, top_k=5):

    documents = [
        candidate["text"]
        for candidate in candidates
    ]

    scores = reranker.score(
        query,
        documents,
    )

    reranked = []

    for candidate, score in zip(
        candidates,
        scores,
    ):

        result = candidate.copy()

        result["rerank_score"] = score

        reranked.append(result)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return reranked[:top_k]


def main():

    print("Loading corpus...")

    chunks = load_corpus()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    query = input(
        "\nQuery:\n"
    ).strip()

    # Temporary test:
    # take the first 10 chunks.
    #
    # Later this will be replaced by the
    # top-200 RRF candidates.
    candidates = chunks[:10]

    reranker = ColBERTReranker()

    print("\nReranking candidates...")

    results = rerank(
        query,
        candidates,
        reranker,
        top_k=5,
    )

    print("\nTop reranked results:")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print("\n---")
        print(f"Rank: {rank}")
        print(
            f"Rerank Score: "
            f"{result['rerank_score']:.4f}"
        )
        print(
            f"Example ID: "
            f"{result['example_id']}"
        )
        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )
        print(
            f"Source: "
            f"{result['source']}"
        )
        print(
            f"Text: "
            f"{result['text']}"
        )


if __name__ == "__main__":
    main()