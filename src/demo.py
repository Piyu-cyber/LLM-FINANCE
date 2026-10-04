import sys

sys.path.insert(0, "src/retrieval")
sys.path.insert(0, "src/verification")

from rrf_retriever import (
    load_corpus,
    build_bm25,
    bm25_search,
    dense_search,
    reciprocal_rank_fusion,
    BM25_TOP_K,
    DENSE_TOP_K,
    RRF_K,
    EMBEDDINGS_PATH,
    MODEL_NAME,
)

from sentence_transformers import SentenceTransformer
import numpy as np

from pipeline import run_verification_pipeline


# ============================================================
# Demo configuration
# ============================================================

EXAMPLE_ID = "UNP/2009/page_34.pdf-1"

QUESTION = (
    "What is the change in interest expense "
    "from 2008 to 2009?"
)


# ============================================================
# Structured claim
#
# This is the claim structure we have already verified.
# Later, the LLM will generate this automatically.
# ============================================================

CLAIM = {
    "claims": [
        {
            "id": "claim_1",

            "claim_text": (
                "The change in interest expense "
                "from 2008 to 2009 was -89 million USD."
            ),

            "value": -89,

            "unit": "million USD",

            "citations": [
                "table_1"
            ],

            "operands": [
    {
        "id": "op_2009",
        "chunk_id": "table_1",
        "span_text": "2009: -600 ( 600 )",
        "context_text": (
            "millions of dollars: interest expense | "
            "2009: -600 ( 600 ) | "
            "2008: -511 ( 511 ) | "
            "2007: -482 ( 482 )"
        ),
        "normalized_value": -600,
        "unit": "million USD",
        "period": "2009",
    },

    {
        "id": "op_2008",
        "chunk_id": "table_1",
        "span_text": "2008: -511 ( 511 )",
        "context_text": (
            "millions of dollars: interest expense | "
            "2009: -600 ( 600 ) | "
            "2008: -511 ( 511 ) | "
            "2007: -482 ( 482 )"
        ),
        "normalized_value": -511,
        "unit": "million USD",
        "period": "2008",
    },
],

            "program": [
                {
                    "id": "step_1",

                    "operation": "subtract",

                    "operands": [
                        "op_2009",
                        "op_2008",
                    ],
                }
            ],

            "referent": {
                "entity": "interest expense",
                "period": "2009",
                "unit": "million USD",
            },

            "status": "answered",
        }
    ],

    "final_claim_id": "claim_1",
}


# ============================================================
# Display helpers
# ============================================================

def separator():
    print("=" * 65)


def section(title):
    print()
    print(f"[{title}]")
    print("-" * 65)


# ============================================================
# Main demo
# ============================================================

def main():

    separator()
    print("                    LLM-FINANCE DEMO")
    separator()

    print("\nQuestion:")
    print(QUESTION)

    # --------------------------------------------------------
    # 1. Load corpus
    # --------------------------------------------------------

    section("1. RETRIEVAL")

    print("Loading corpus...")

    chunks = load_corpus()

    print(f"Corpus chunks       : {len(chunks)}")

    # --------------------------------------------------------
    # 2. BM25
    # --------------------------------------------------------

    print("\nBuilding BM25 index...")

    bm25 = build_bm25(chunks)

    bm25_results = bm25_search(
        QUESTION,
        bm25,
        BM25_TOP_K,
    )

    print(f"BM25 candidates     : {len(bm25_results)}")

    # --------------------------------------------------------
    # 3. Dense retrieval
    # --------------------------------------------------------

    print("\nLoading dense embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Embedding shape     : {embeddings.shape}"
    )

    print(
        f"Loading model       : {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    dense_results = dense_search(
        QUESTION,
        model,
        embeddings,
        DENSE_TOP_K,
    )

    print(
        f"Dense candidates    : {len(dense_results)}"
    )

    # --------------------------------------------------------
    # 4. RRF
    # --------------------------------------------------------

    print("\nApplying Reciprocal Rank Fusion...")

    fused_results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        RRF_K,
    )

    print(
        f"RRF candidate pool  : {len(fused_results)}"
    )

    # --------------------------------------------------------
    # Display top evidence
    # --------------------------------------------------------

    print("\nTop retrieved evidence:")

    displayed = 0

    for rank, (index, score) in enumerate(
        fused_results,
        start=1,
    ):

        chunk = chunks[index]

        # For this demo, show evidence from our
        # target FinQA example.
        if chunk["example_id"] != EXAMPLE_ID:
            continue

        print("\n---")
        print(f"RRF Rank            : {rank}")
        print(f"RRF Score           : {score:.6f}")
        print(f"Example ID          : {chunk['example_id']}")
        print(f"Chunk ID            : {chunk['chunk_id']}")
        print(f"Source              : {chunk['source']}")
        print(f"Text                : {chunk['text']}")

        displayed += 1

        if displayed >= 3:
            break

    if displayed == 0:

        print(
            "\nTarget evidence was not found "
            "in the displayed RRF results."
        )

    # --------------------------------------------------------
    # 5. Structured claim
    # --------------------------------------------------------

    section("2. STRUCTURED CLAIM")

    final_claim = CLAIM["claims"][0]

    print("Claim:")
    print(final_claim["claim_text"])

    print("\nOperands:")

    for operand in final_claim["operands"]:

        print(
            f"  {operand['id']}: "
            f"{operand['normalized_value']} "
            f"{operand['unit']} "
            f"({operand['period']})"
        )

    print("\nOperation:")

    step = final_claim["program"][0]

    print(
        f"  {step['operation']}("
        f"{step['operands'][0]}, "
        f"{step['operands'][1]})"
    )

    # --------------------------------------------------------
    # 6. Verification
    # --------------------------------------------------------

    section("3. VERIFICATION")

    result = run_verification_pipeline(
        CLAIM,
        EXAMPLE_ID,
    )

    if not result["valid"]:

        print("✗ VERIFICATION FAILED")

        print(
            f"\nStatus : {result['status']}"
        )

        print(
            f"Reason : {result['reason']}"
        )

        return

    print("✓ Evidence verified")
    print("✓ Values verified")
    print("✓ Periods verified")
    print("✓ Units verified")

    # --------------------------------------------------------
    # 7. Deterministic calculation
    # --------------------------------------------------------

    section("4. DETERMINISTIC CALCULATION")

    print(
        "-600 - (-511) = -89 million USD"
    )

    print(
        "\nCalculated result:",
        result["calculation"]["result"],
    )

    # --------------------------------------------------------
    # 8. Final answer
    # --------------------------------------------------------

    section("5. FINAL RESULT")

    print("✓ VERIFIED_RESULT")

    print(
        "\nThe change in interest expense "
        "from 2008 to 2009 was:"
    )

    print(
        f"\n    {result['calculation']['result']:.0f} million USD"
    )

    separator()


if __name__ == "__main__":
    main()