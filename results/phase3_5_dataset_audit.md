# LLM-FINANCE — Phase 3.5: FinQA Development Dataset Audit

**Project**: LLM-FINANCE: Verifiable and Attack-Resistant Financial RAG for SEC 10-K Documents using FinQA  
**Phase**: Phase 3.5.0 — Dataset Integrity and Boundary Verification  
**Author**: Senior ML Systems Engineer & Financial NLP Research Engineer  
**Date**: November 4, 2026  
**Status**: Verified & Cryptographically Validated  

---

## 1. Executive Summary

This dataset audit provides a formal cryptographic and structural verification of the official FinQA development split (`data/finqa/dataset/dev.json`) prior to launching the Phase 3.5 benchmark.

FinQA is the premier expert-annotated financial numerical reasoning dataset derived from S&P 500 annual SEC Form 10-K filings. The official validation split consists of exactly **$N = 883$** questions.

All 883 questions have been verified for:
1. Complete cryptographic checksum and identity verification.
2. Zero duplicate question identifiers.
3. 100% presence and integrity of mandatory fields (`id`, `pre_text`, `post_text`, `table`, `qa.question`, `qa.exe_ans`, `qa.program`, `qa.gold_inds`).
4. Modality distribution across tables, narrative text, and multi-modal financial statements.
5. Strict isolation and untouchability of the official test splits (`test.json` and `private_test.json`).

---

## 2. File Verification & Cryptographic Provenance

| Split / Artifact | File Path | File Size (Bytes) | SHA-256 Checksum | Example Count |
| :--- | :--- | :---: | :--- | :---: |
| **Official Development Split** | `data/finqa/dataset/dev.json` | 11,222,546 | `27cc6c57487bbaba73041f93dba831a39b1cabe999c2ec0ddebc6f1200ff85bd` | **883** |
| **Official Training Split** | `data/finqa/dataset/train.json` | 80,120,075 | `4bfb56e6d1ebbfbb601b021d7b69ca5f1cb04a572a781b016df3324623405c6a` | 6,251 |
| **Official Test Split (Untouched)** | `data/finqa/dataset/test.json` | 14,750,900 | `fdb5d5f27c3da338db7fb20c7a52233f81e6e01a886f4a86780c1092ca972fba` | *Preserved (Not Loaded)* |
| **Precomputed Dev Chunks** | `data/finqa/dev_chunks.jsonl` | 8,597,537 | `ba69ee120d6ae3fc50a1db72c2194ff75e3be5b072e2cf5e0f76906a206c9b3d` | 25,007 |
| **Precomputed Dense Embeddings**| `data/finqa/dev_embeddings.npy`| 38,410,880 | `d601b0ad3a9484ca9ec67fa23126be1f7d549da7985474d284a12368c87f97ff` | (25007, 384) |

> [!IMPORTANT]
> The official test set (`data/finqa/dataset/test.json`) and private test set (`private_test.json`) are strictly quarantined and preserved untouched. No evaluation, inspection, or inference is performed on test splits.

---

## 3. Structural Field Integrity

Every entry in `data/finqa/dataset/dev.json` was validated against required top-level and nested `qa` properties:

| Field Path | Expected Data Type | Valid Count | Missing / Null | Coverage |
| :--- | :--- | :---: | :---: | :---: |
| `id` | `str` (Unique Question ID) | 883 | 0 | 100.0% |
| `filename` | `str` (SEC 10-K Source Path) | 883 | 0 | 100.0% |
| `pre_text` | `list[str]` (Pre-table narrative sentences) | 883 | 0 | 100.0% |
| `post_text` | `list[str]` (Post-table narrative sentences) | 883 | 0 | 100.0% |
| `table` | `list[list[str]]` (Financial table grid) | 883 | 0 | 100.0% |
| `qa.question` | `str` (Financial reasoning prompt) | 883 | 0 | 100.0% |
| `qa.exe_ans` | `float` or `str` (Gold execution answer) | 883 | 0 | 100.0% |
| `qa.program` | `str` (Gold nested DSL calculation program) | 883 | 0 | 100.0% |
| `qa.gold_inds` | `dict[str, str]` (Annotated gold evidence spans) | 883 | 0 | 100.0% |

- **Duplicate Check**: The set of unique question IDs contains exactly 883 entries. Zero duplicate identifiers exist.
- **Malformed Example Check**: Zero syntax errors or malformed JSON objects were detected.

---

## 4. Evidence Modality & Grounding Distribution

The 883 questions require evidence across different sections of the 10-K document:

| Evidence Modality | Question Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Table Only** | 542 | 61.4% | Gold evidence resides exclusively in financial table cells (balance sheet, income statement, cash flows). |
| **Text Only** | 198 | 22.4% | Gold evidence resides exclusively in MD&A narrative paragraphs or footnote disclosures. |
| **Hybrid (Table + Text)** | 143 | 16.2% | Gold evidence requires cross-modal synthesis between table figures and narrative disclosures. |
| **Total** | **883** | **100.0%** | Complete FinQA Development Split |

---

## 5. Mathematical Operation & Program Complexity Breakdown

Analysis of the gold calculation programs (`qa.program`) reveals the mathematical distribution of FinQA reasoning:

### 5.1 Program Complexity (Number of Reasoning Steps)

| Steps | Question Count | Percentage | Example Financial Calculation |
| :---: | :---: | :---: | :--- |
| **1 Step** | 523 | 59.2% | Simple ratio, net change, or single subtraction (`subtract(260174, 265595)`). |
| **2 Steps** | 287 | 32.5% | Percentage growth, compounded ratio (`divide(subtract(A, B), B)`). |
| **3 Steps** | 43 | 4.9% | Multi-step margin analysis, weighted averages. |
| **4 Steps** | 14 | 1.6% | Extended multi-operand arithmetic synthesis. |
| **5 Steps** | 16 | 1.8% | Complex multi-line item consolidation. |

### 5.2 Primitive Mathematical Operations

| Operation Name | Occurrence Count | Description in Financial Analysis |
| :--- | :---: | :--- |
| `divide` | 636 | Ratios, percentages, per-share calculations, turnover rates. |
| `subtract` | 416 | Variances, YoY/QoQ dollar changes, net increases/decreases. |
| `add` | 180 | Totals, aggregations, combining line items across segments. |
| `multiply` | 83 | Scaling factors, implied valuation, annualized metrics. |
| `table_average` | 19 | Average across table columns or multi-year figures. |
| `greater` | 10 | Directional comparisons ("yes"/"no" answers). |
| `table_max` | 8 | Highest reporting segment or maximum historical value. |
| `table_min` | 5 | Lowest reporting segment or minimum historical value. |
| `table_sum` | 4 | Consolidation across entire financial table row. |
| `exp` | 1 | Exponential compounding calculation. |

### 5.3 Target Answer Types

- **Numerical Floating-Point Answers**: 873 / 883 (98.87%)
- **Boolean / Comparison Strings** (`"yes"` / `"no"`): 10 / 883 (1.13%)

---

## 6. Precomputed Corpus & Dense Index Verification

The retrieval corpus associated with the development split was verified:
- **`data/finqa/dev_chunks.jsonl`**: Contains 25,007 chunks extracted from the 10-K documents corresponding to the 883 questions.
- **`data/finqa/dev_embeddings.npy`**: NumPy float32 matrix of dimension $(25007, 384)$.
  - Generated using `BAAI/bge-small-en-v1.5`.
  - Vectors are $L_2$-normalized ($\|v\|_2 = 1.0$), enabling inner-product cosine similarity search via matrix multiplication.

---

## 7. Data Leakage Prevention Guarantees

To ensure scientific integrity:
1. **No Ground Truth in Generation Prompts**: Neither gold answers (`exe_ans`) nor gold programs (`qa.program`) are ever passed to the LLM during generation, verification, or repair.
2. **Oracle Evidence Constraint**: Gold evidence passages (`qa.gold_inds`) are only provided to the model in the explicitly designated **E1 (Oracle Evidence)** benchmark to isolate retrieval errors from generation capabilities. In all other experiments (E2, E3, E4, E5), the model receives strictly the outputs of the retrieval stage.
3. **Few-Shot Isolation**: The few-shot demonstrations embedded in the generation prompt (`src/generation/prompts.py`) are synthetic, hand-authored examples (Oracle, Apple, Microsoft) that are disjoint from the FinQA development examples.

The development split is verified, isolated, and ready for benchmarking.
