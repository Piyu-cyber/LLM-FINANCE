import json
from pathlib import Path
from typing import Any


FINQA_PATH = Path("data/finqa/dataset/train.json")


def load_finqa(path: Path = FINQA_PATH) -> list[dict[str, Any]]:
    """Load FinQA examples from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    data = load_finqa()

    print(f"Number of examples: {len(data)}")

    example = data[0]

    print("\nTop-level fields:")
    for key in example:
        print(f"  - {key}")

    qa = example["qa"]

    print("\nQuestion:")
    print(qa["question"])

    print("\nAnswer:")
    print(qa["answer"])

    print("\nProgram:")
    print(qa["program"])

    print("\nExecutable answer:")
    print(qa["exe_ans"])

    print("\nGold evidence:")
    print(qa["gold_inds"])


if __name__ == "__main__":
    main()