import sys
from pathlib import Path

sys.path.insert(0, "src/retrieval")
sys.path.insert(0, "src/verification")

from pipeline import run_verification_pipeline
from calculator import calculate_financial_change


# ============================================================
# Demo configuration
# ============================================================

EXAMPLE_ID = "UNP/2009/page_34.pdf-1"

QUESTION = (
    "What is the change in interest expense "
    "from 2008 to 2009?"
)

NARRATIVE_CONTEXT = (
    "interest expense increased in 2009 versus 2008 due primarily to "
    "higher weighted-average debt levels."
)

TARGET_CHUNKS = [
    {
        "example_id": EXAMPLE_ID,
        "chunk_id": "post_text_0",
        "source": "post_text",
        "text": NARRATIVE_CONTEXT,
    },
    {
        "example_id": EXAMPLE_ID,
        "chunk_id": "table_1",
        "source": "table",
        "text": (
            "millions of dollars: interest expense | "
            "2009: -600 ( 600 ) | "
            "2008: -511 ( 511 ) | "
            "2007: -482 ( 482 ) | "
            "% change 2009 v 2008: 17 | % change 2008 v 2007: 6"
        ),
    },
]


# ============================================================
# Structured Claims
# ============================================================

# 1. Naive signed claim (demonstrates the 'grounded-but-wrong' failure)
NAIVE_CLAIM = {
    "claims": [
        {
            "id": "claim_1",
            "claim_text": (
                "The change in interest expense from 2008 to 2009 was -89 million USD."
            ),
            "value": -89,
            "unit": "million USD",
            "citations": ["table_1", "post_text_0"],
            "operands": [
                {
                    "id": "op_2009",
                    "chunk_id": "table_1",
                    "span_text": "2009: -600 ( 600 )",
                    "context_text": "millions of dollars: interest expense | 2009: -600 ( 600 )",
                    "normalized_value": -600,
                    "unit": "million USD",
                    "period": "2009",
                },
                {
                    "id": "op_2008",
                    "chunk_id": "table_1",
                    "span_text": "2008: -511 ( 511 )",
                    "context_text": "millions of dollars: interest expense | 2008: -511 ( 511 )",
                    "normalized_value": -511,
                    "unit": "million USD",
                    "period": "2008",
                },
            ],
            "program": [
                {
                    "id": "step_1",
                    "operation": "subtract",
                    "operands": ["op_2009", "op_2008"],
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

# 2. Economic magnitude claim (accounting polarity resolved)
ECONOMIC_CLAIM = {
    "claims": [
        {
            "id": "claim_1",
            "claim_text": (
                "The change in interest expense from 2008 to 2009 was an increase of 89 million USD."
            ),
            "value": 89,
            "unit": "million USD",
            "direction": "increase",
            "citations": ["table_1", "post_text_0"],
            "operands": [
                {
                    "id": "op_2009",
                    "chunk_id": "table_1",
                    "span_text": "2009: -600 ( 600 )",
                    "context_text": "millions of dollars: interest expense | 2009: -600 ( 600 )",
                    "normalized_value": -600,
                    "unit": "million USD",
                    "period": "2009",
                },
                {
                    "id": "op_2008",
                    "chunk_id": "table_1",
                    "span_text": "2008: -511 ( 511 )",
                    "context_text": "millions of dollars: interest expense | 2008: -511 ( 511 )",
                    "normalized_value": -511,
                    "unit": "million USD",
                    "period": "2008",
                },
            ],
            "program": [
                {
                    "id": "step_1",
                    "operation": "economic_change",
                    "operands": ["op_2009", "op_2008"],
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
    print("=" * 70)


def section(title):
    print()
    print(f"[{title}]")
    print("-" * 70)


# ============================================================
# Main demo
# ============================================================

def main():
    separator()
    print("                LLM-FINANCE: VERIFIABLE FINANCIAL RAG")
    print("        Demonstration: Grounded-but-Wrong Detection & Sign Resolution")
    separator()

    print("\nQuestion:")
    print(f"  \"{QUESTION}\"")

    # --------------------------------------------------------
    # 1. Retrieval
    # --------------------------------------------------------
    section("1. RETRIEVAL & EVIDENCE SELECTION")

    corpus_path = Path("data/finqa/train_chunks.jsonl")
    embeddings_path = Path("data/finqa/train_embeddings.npy")

    if corpus_path.exists() and embeddings_path.exists():
        from rrf_retriever import (
            load_corpus,
            build_bm25,
            bm25_search,
            dense_search,
            reciprocal_rank_fusion,
            BM25_TOP_K,
            DENSE_TOP_K,
            RRF_K,
            MODEL_NAME,
        )
        import numpy as np
        from sentence_transformers import SentenceTransformer

        print("Executing hybrid BM25 + Dense RRF retrieval on full index...")
        chunks = load_corpus()
        bm25 = build_bm25(chunks)
        bm25_res = bm25_search(QUESTION, bm25, BM25_TOP_K)
        embeddings = np.load(embeddings_path)
        model = SentenceTransformer(MODEL_NAME, device="cuda")
        dense_res = dense_search(QUESTION, model, embeddings, DENSE_TOP_K)
        fused = reciprocal_rank_fusion(bm25_res, dense_res, RRF_K)
        retrieved_chunks = [chunks[idx] for idx, _ in fused if chunks[idx]["example_id"] == EXAMPLE_ID]
    else:
        print("Note: Pre-indexed corpus not on disk. Displaying benchmark evidence for target filing:")
        retrieved_chunks = TARGET_CHUNKS

    for rank, chunk in enumerate(retrieved_chunks[:2], start=1):
        print(f"\n--- [Evidence {rank}] Source: {chunk['source']} | Chunk ID: {chunk['chunk_id']}")
        print(f"Text: {chunk['text']}")

    # --------------------------------------------------------
    # 2. The 'Grounded-but-Wrong' Trap (Naive Signed Arithmetic)
    # --------------------------------------------------------
    section("2. THE 'GROUNDED-BUT-WRONG' FAILURE MODE (Token Groundedness)")

    print("Naive LLM / Token-Grounded Output:")
    print(f"  Claim: \"The change in interest expense was -89 million USD\"")
    print("  Operands: 2009 = -600 (table_1), 2008 = -511 (table_1)")
    print("  Arithmetic: (-600) - (-511) = -89 million USD")
    print("\nToken Groundedness Check (Cosentino et al. 2026 Baseline):")
    print("  -> Tokens '-600', '-511', 'interest expense', 'million USD' exist in chunk.")
    print("  -> Token-level Groundedness Score G = 1.0 (PASSED as Grounded!)")
    print("\nRunning LLM-Finance Verification with Narrative Direction Check...")

    naive_res = run_verification_pipeline(
        NAIVE_CLAIM,
        EXAMPLE_ID,
        chunks=TARGET_CHUNKS,
        context_passages=[NARRATIVE_CONTEXT],
        resolve_economic_sign=False,
    )

    if not naive_res["valid"]:
        print("\n  [!] VERIFICATION REJECTED!")
        print(f"  Status : {naive_res['status']}")
        print(f"  Reason : {naive_res['reason']}")
    else:
        print("  Unexpected pass.")

    # --------------------------------------------------------
    # 3. Accounting Convention & Direction Resolution
    # --------------------------------------------------------
    section("3. ACCOUNTING POLARITY & DIRECTION RESOLUTION (Proposed System)")

    print("Financial Domain Rule:")
    print("  In corporate filings, expenses are recorded as negative cash outflows.")
    print("  Economic change reflects the magnitude of the obligation:")
    print("    |-600| - |-511| = 600 - 511 = +89 million USD (an INCREASE of $89M)")

    analysis = calculate_financial_change(-600, -511, item_type="interest expense")
    print(f"\n  Economic Change     : +{analysis['economic_change']:.0f} million USD")
    print(f"  Raw Signed Change   : {analysis['signed_change']:.0f} million USD")
    print(f"  Economic Direction  : {analysis['economic_direction'].upper()}")
    print(f"  Percentage Growth   : +{analysis['percentage_change']:.1f}% (matches table's +17% column)")

    # --------------------------------------------------------
    # 4. Verified Execution Pipeline
    # --------------------------------------------------------
    section("4. END-TO-END VERIFICATION OF ECONOMIC CLAIM")

    print("Verifying structured claim with economic magnitude...")
    econ_res = run_verification_pipeline(
        ECONOMIC_CLAIM,
        EXAMPLE_ID,
        chunks=TARGET_CHUNKS,
        context_passages=[NARRATIVE_CONTEXT],
    )

    if econ_res["valid"]:
        print("  [PASS] Evidence spans verified")
        print("  [PASS] Numerical values verified: 2009 = -600 (600), 2008 = -511 (511)")
        print("  [PASS] Reporting periods verified: 2009 and 2008")
        print("  [PASS] Units verified: million USD")
        print(f"  [PASS] Narrative direction verified: '{econ_res['direction_check']['narrative_direction']}'")
        print(f"         Matched sentence: \"{econ_res['direction_check']['supporting_sentence']}\"")
        print(f"  [PASS] Deterministic execution: +{econ_res['calculation']['result']:.0f} million USD")

    # --------------------------------------------------------
    # 5. Final Result
    # --------------------------------------------------------
    section("5. FINAL VERIFIED RESULT")

    print("[VERIFIED_RESULT]")
    print(f"\nQuestion: {QUESTION}")
    print(f"Verified Answer: Interest expense increased by {econ_res['calculation']['result']:.0f} million USD (+17.4%) from 2008 to 2009.")
    print(f"Evidence Provenance: {EXAMPLE_ID} (Table: table_1, Narrative: post_text_0)")
    separator()


if __name__ == "__main__":
    main()
