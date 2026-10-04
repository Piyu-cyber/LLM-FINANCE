from typing import Any


def create_text_chunks(example: dict[str, Any]) -> list[dict[str, str]]:
    """Create chunks from pre_text and post_text."""

    chunks = []

    example_id = str(example["id"])

    # Pre-text chunks
    for i, text in enumerate(example["pre_text"]):
        if text.strip() and any(char.isalnum() for char in text):
            chunks.append(
                {
                    "example_id": example_id,
                    "chunk_id": f"text_{i}",
                    "source": "pre_text",
                    "position": i,
                    "text": text.strip(),
                }
            )

    # Post-text chunks
    for i, text in enumerate(example["post_text"]):
        if text.strip() and any(char.isalnum() for char in text):
            chunks.append(
                {
                    "example_id": example_id,
                    "chunk_id": f"post_text_{i}",
                    "source": "post_text",
                    "position": i,
                    "text": text.strip(),
                }
            )

    return chunks


def create_table_chunks(example: dict[str, Any]) -> list[dict[str, str]]:
    """Create one chunk per table row while preserving column headers."""

    table = example["table"]

    if not table:
        return []

    headers = table[0]
    chunks = []

    example_id = str(example["id"])

    for i, row in enumerate(table[1:]):
        values = []

        for header, value in zip(headers, row):
            if header:
                values.append(f"{header}: {value}")
            else:
                values.append(value)

        row_text = " | ".join(values)

        chunks.append(
            {
                "example_id": example_id,
                "chunk_id": f"table_{i}",
                "source": "table",
                "position": i,
                "text": row_text.strip(),
            }
        )

    return chunks


def create_chunks(example: dict[str, Any]) -> list[dict[str, str]]:
    """Create all retrievable chunks for a FinQA example."""

    text_chunks = create_text_chunks(example)
    table_chunks = create_table_chunks(example)

    return text_chunks + table_chunks

if __name__ == "__main__":
    from finqa_loader import load_finqa

    data = load_finqa()

    example = data[0]

    chunks = create_chunks(example)

    print(f"Number of chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n---")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text']}")