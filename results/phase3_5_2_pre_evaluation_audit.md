# LLM-FINANCE — Phase 3.5.2: Pre-Evaluation Audit Report

## 1. Executive Summary

This report establishes the preliminary research integrity verification for **Phase 3.5.2: Full FinQA Development Benchmark and Ablation Study**. All dataset splits, environment variables, local weights, and execution parameters have been audited, cryptographically hashed, and verified prior to launching any experimental runs.

---

## 2. Hardware and Environment Specification

| Parameter | Observed Value | Verification Status |
|---|---|---|
| **Python Binary** | `C:\Users\palak\AppData\Local\Programs\Python\Python313\python.exe` | Verified 64-bit |
| **Python Version** | `3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11)` | Supported |
| **PyTorch Version** | `2.6.0+cu124` | CUDA 12.4 Enabled |
| **CUDA Capability** | `8.9` (Ada Lovelace) | Verified |
| **GPU Model** | `NVIDIA GeForce RTX 4060 Laptop GPU` | 1 Physical GPU |
| **Physical VRAM** | `8,187.50 MB` (~7.996 GB) | Verified |
| **BF16 Hardware Support** | `True` (`torch.cuda.is_bf16_supported() == True`) | Verified |
| **Transformers Version** | `4.57.6` | Integrated |
| **Accelerate Version** | `1.15.0` | Integrated |
| **Test Suite Status** | 282 passed in 28.23s | 100% Passing (0 failures, 0 skips) |

---

## 3. Dataset Integrity & Cryptographic Audit

All FinQA dataset files located in `data/finqa/dataset/` were verified for example counts, file byte sizes, modification timestamps, and SHA-256 cryptographic hashes:

| Dataset File | Role in Phase 3.5.2 | Example Count | File Size (Bytes) | SHA-256 Checksum | Audit Status |
|---|---|---|---|---|---|
| **`dev.json`** | **Active Evaluation Target** | **883** | **11,222,546** | `27cc6c57487bbaba73041f93dba831a39b1cabe999c2ec0ddebc6f1200ff85bd` | **Verified Intact (883 examples)** |
| `train.json` | Training Reference | 6,251 | 80,120,075 | `9862301c1a28f78bfb59050203bf66b13719c4720cd18291a069331b21c63352` | Read-only reference |
| `test.json` | Official Blind Test Split | 1,147 | 14,750,900 | `37a6e6e198b7821dc080f079d43a4a7a9d3a7b562803b7e97875bf70cd61fb08` | **Untouched & Locked** |
| `private_test.json` | Private Evaluation Split | 919 | 5,697,753 | `aaf304a789d21d8451bf3281decdb495f9f9ddde9ce71c866bc177b6bd2a7d5a` | **Untouched & Locked** |

> [!IMPORTANT]
> **Research Protocol Guarantee**: Neither `test.json` nor `private_test.json` is modified, loaded, or inspected during Phase 3.5.2. Evaluation is restricted strictly to the 883 examples of `dev.json`.

---

## 4. Model Snapshot Verification

The model is pinned to the local snapshot directory without internet connectivity:

- **Model Identifier**: `Qwen/Qwen2.5-3B-Instruct`
- **Pinned Commit Hash**: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- **Absolute Local Directory**: `C:\Users\palak\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1`
- **Model Shards**:
  - `model-00001-of-00002.safetensors` (3,968,658,944 bytes)
  - `model-00002-of-00002.safetensors` (2,203,268,048 bytes)
  - `config.json` (661 bytes)
  - `tokenizer.json` (7,031,645 bytes)
  - `vocab.json` (2,776,833 bytes)
- **Offline Guarantee**: `local_files_only=True` is enforced via `llm_client.py`. Absolute snapshot directory resolution prevents Hugging Face Hub HTTP checks.

---

## 5. Retrieval Corpus and Configuration

The evaluation relies on the pre-processed development corpus established in Phase 2 and Phase 2.5:

- **Corpus File**: `data/finqa/dev_chunks.jsonl`
- **Total Chunks**: `25,007` chunks
- **Dense Embeddings File**: `data/finqa/dev_embeddings.npy`
- **Embeddings Dimension**: `(25007, 384)`, `float32`
- **Dense Embedding Model**: `BAAI/bge-small-en-v1.5` (snapshot `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` locally cached)
- **Primary Retrieval Strategy**:
  - Strategy: Dense Retrieval (`BAAI/bge-small-en-v1.5`)
  - Expansion: Parent-Child Context Expansion (`deduplicated_parent`)
  - Section Deduplication: Enabled
  - Metadata Filtering / Year Boost: **Disabled**
  - Candidate Depth: Top-5 chunks
- **Comparison Retrieval Strategy**:
  - Strategy: Weighted Reciprocal Rank Fusion (Weighted RRF)
  - Weights: `w_dense = 0.75`, `w_bm25 = 0.25`, `k = 60`
  - Expansion: Parent-Child Context Expansion (`deduplicated_parent`)
  - Candidate Depth: Top-5 chunks

---

## 6. Generation and Verification Invariants

- **Backend**: `HuggingFaceLocalClient`
- **Precision / Dtype**: `torch.bfloat16`
- **Batch Size**: `1`
- **Decoding Configuration**:
  - `temperature = 0.0` (Greedy deterministic decoding)
  - `do_sample = False`
- **Token Limits**:
  - Context Token Budget: `max_input_tokens = 2048` (context budget ~1,536 tokens, chunk-atomic truncation)
  - Maximum Output Tokens: `max_output_tokens = 512`
  - Generation Timeout: `60.0s`
- **Verification Engine**:
  - Schema: Strict `CLAIM_SCHEMA` JSON validation
  - Determinism: `run_verification_pipeline` symbolic arithmetic validation
  - Repair Policy: Maximum 1 repair attempt, strictly verified through the same pipeline
  - Abstention: Enabled for out-of-scope or ungrounded queries

---

## 7. Official FinQA Evaluation Conventions

Per official `evaluate.py`:
- **Answer Format**: Float scalar (`qa["exe_ans"]`)
- **Metric Definitions**:
  - **Exact Numerical Accuracy**: `pred == gold`
  - **Tolerance Accuracy**: `math.isclose(pred, gold, rel_tol=1e-3, abs_tol=1e-3)`
  - **Answer Coverage**: Answered / Total (883)
  - **Answered Accuracy**: Correct / Answered

---

## 8. Audit Conclusion

All integrity requirements have been verified. Zero regressions or discrepancies were found. Pre-evaluation audit is **APPROVED**. The system is ready to proceed to Phase 1 (Runtime Pilot).
