# LLM-FINANCE — Phase 3.5: Pre-Evaluation Audit and Benchmark Specification

**Project**: LLM-FINANCE: Verifiable and Attack-Resistant Financial RAG for SEC 10-K Documents using FinQA  
**Phase**: Phase 3.5.0 — Pre-Evaluation Audit, Hardware Validation, and Protocol Design  
**Author**: Senior ML Systems Engineer & Financial NLP Research Engineer  
**Date**: November 4, 2026  
**Status**: Pre-Evaluation Audit Complete  

---

## 1. Executive Summary

This pre-evaluation audit establishes the experimental parameters, environment specifications, model selection analysis, evaluation metrics, and stage-by-stage methodology for **Phase 3.5: FinQA Development Evaluation and RTX 4060 Profiling**.

The goal of Phase 3.5 is to transition from deterministic unit mock testing to **real neural language model inference** over the official FinQA development set ($N = 883$), rigorously quantifying:
1. Grounded financial execution accuracy.
2. The empirical value added by the deterministic symbolic verification engine (preventing hallucinated answers from being accepted).
3. The recovery rate and failure modes of single-turn diagnostic claim repair ($B_{\text{repair}} \le 1$).
4. System abstention fidelity when context is ungroundable.
5. Real GPU throughput, peak VRAM utilization, and latency profiling on the NVIDIA GeForce RTX 4060 Laptop GPU (8 GB).

---

## 2. Experimental Environment & Hardware Verification

The physical execution environment was inspected on the host system:

| Metric / Component | Verified Specification | Status / Compatibility |
| :--- | :--- | :---: |
| **Operating System** | Windows 11 Home (x86_64) | Fully Supported |
| **Primary Python** | Python 3.13.7 (`C:\Users\palak\AppData\Local\Programs\Python\Python313\python.exe`) | Active Environment |
| **PyTorch Version** | `2.6.0+cu124` | CUDA-Enabled |
| **CUDA Runtime** | CUDA 12.4 | Active |
| **GPU Model** | **NVIDIA GeForce RTX 4060 Laptop GPU** | Active |
| **GPU Architecture** | Ada Lovelace (Compute Capability **8.9**) | Native BF16 & FP16 |
| **Total VRAM** | 7.995 GB (8,187 MB) | Measured |
| **Free VRAM** | **6.930 GB** (7,096 MB) | Available for inference |
| **Bfloat16 Support** | `torch.cuda.is_bf16_supported() == True` | Supported |
| **Available Disk Space** | **131.43 GB Free** | Adequate for model caches |
| **Transformers Version** | `4.57.6` | Installed |
| **SentenceTransformers** | `5.3.0` | Installed |
| **Accelerate** | Not Installed | Standard PyTorch loader used |
| **BitsAndBytes** | Not Installed | Native BF16 precision selected |

---

## 3. Model Candidate Analysis & Inference Feasibility

### 3.1 Candidate Selection: `Qwen/Qwen2.5-3B-Instruct`
The designated primary candidate for Phase 3.5 is **Qwen2.5-3B-Instruct** (Alibaba Cloud):
- **Architecture**: Causal language model with Grouped-Query Attention (GQA).
- **Parameters**: 3.09 billion parameters.
- **Context Window**: 32,768 tokens (we constrain generation prompts to a chunk-atomic token budget of 2,048 tokens).
- **License**: Apache 2.0 (open research & commercial use).

### 3.2 VRAM Footprint & Budget Calculation
On the NVIDIA RTX 4060 Laptop GPU (6.93 GB free VRAM):

1. **Model Weights (BF16 Precision)**:
   $$M_{\text{weights}} = 3.09 \times 10^9 \times 2 \text{ bytes} \approx 6.18 \times 10^9 \text{ bytes} \approx 5.76 \text{ GB}$$
2. **KV Cache (Batch Size = 1, Sequence Length = 2,048 tokens)**:
   $$M_{\text{KV}} = 2 \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times D_{\text{head}} \times L_{\text{seq}} \times 2 \text{ bytes}$$
   For Qwen2.5-3B ($N_{\text{layers}} = 36$, $N_{\text{kv\_heads}} = 2$, $D_{\text{head}} = 128$, $L_{\text{seq}} = 2048$):
   $$M_{\text{KV}} = 2 \times 36 \times 2 \times 128 \times 2048 \times 2 \approx 75.5 \text{ MB}$$
3. **Activation Overhead (Batch Size = 1)**:
   $$\approx 150 - 250 \text{ MB}$$
4. **Total Peak VRAM Estimate**:
   $$M_{\text{peak}} = 5.76 + 0.08 + 0.20 \approx \mathbf{6.04 \text{ GB}}$$

**Conclusion**: The model fits comfortably within the **6.93 GB** free VRAM budget in native **bfloat16** precision without requiring external 4-bit quantization dependencies (`bitsandbytes`, which is not installed and historically prone to Windows DLL linking issues).

---

## 4. Dataset Integrity & Isolation Guarantees

- **Dataset File**: `data/finqa/dataset/dev.json` ($N = 883$ questions).
- **SHA-256 Checksum**: `27cc6c57487bbaba73041f93dba831a39b1cabe999c2ec0ddebc6f1200ff85bd`.
- **Precomputed Corpus**: `data/finqa/dev_chunks.jsonl` (25,007 chunks).
- **Precomputed Dense Embeddings**: `data/finqa/dev_embeddings.npy` (Shape: $25007 \times 384$, $L_2$-normalized float32 vectors generated with `BAAI/bge-small-en-v1.5`).
- **Test Set Isolation**: The test split (`data/finqa/dataset/test.json` and `private_test.json`) is strictly quarantined and will not be accessed or evaluated during Phase 3.5.
- **Data Leakage Safeguards**:
  - The model is never supplied gold programs or gold execution answers in prompts.
  - Few-shot prompts in `src/generation/prompts.py` use synthetic hand-authored SEC examples disjoint from the FinQA dev set.
  - Gold evidence chunks are supplied only in the explicitly designated **E1 (Oracle Evidence)** experiment.

---

## 5. Execution Accuracy Formulation & Comparison Rules

In accordance with official FinQA benchmark standards (`data/finqa/code/evaluate/evaluate.py`), execution accuracy measures whether the numerical value computed by the model matches the gold execution answer:

### 5.1 Official Evaluation Rules:
1. **Direct Equality**:
   $$\text{Match}(y_{\text{pred}}, y_{\text{gold}}) = \text{True} \iff y_{\text{pred}} == y_{\text{gold}}$$
2. **Floating-Point Rounding & Relative Tolerance**:
   Financial reports often round percentages and ratios to 2 or 3 decimal places:
   $$|y_{\text{pred}} - y_{\text{gold}}| \le 10^{-4} \quad \lor \quad \frac{|y_{\text{pred}} - y_{\text{gold}}|}{\max(|y_{\text{gold}}|, 10^{-4})} \le 10^{-3}$$
3. **Percentage / Basis Point Equivalence**:
   In FinQA, a model may calculate $0.154$ while the gold answer is recorded as $15.4$ (percent). If the prompt asks for a percentage or the referent is denominated in `%`, $y_{\text{pred}} \times 100 == y_{\text{gold}}$ or $y_{\text{pred}} / 100 == y_{\text{gold}}$ is treated as an equivalent match.
4. **Boolean Directional Answers**:
   For comparison questions (e.g. `greater`), string matches `"yes"` and `"no"` are evaluated directly.

---

## 6. Planned Benchmark Conditions (E1 to E5)

All experiments evaluate the same 883 FinQA development questions under identical generation parameters ($T = 0.0$, greedy decoding, $max\_output\_tokens = 512$, $max\_input\_tokens = 2048$):

| Exp ID | Condition Name | Retrieval Configuration | Evidence Context | Verification Engine | Single Repair ($B=1$) | Primary Evaluation Purpose |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **E1** | **Oracle Evidence** | Gold Passage Annotations | `qa.gold_inds` (Zero retrieval noise) | Disabled (Predicted Program Executed) | Disabled | Upper-bound generation and reasoning capacity of Qwen2.5-3B. |
| **E2** | **Retrieved (No Verif)** | Dense (`BGE-small`) + Parent-Child | Top-5 Retrieved Chunks + Parent Expansion | Disabled (Predicted Program Executed) | Disabled | Unverified RAG baseline; establishes hallucination and false acceptance rate. |
| **E3** | **Retrieved + Verif** | Dense (`BGE-small`) + Parent-Child | Top-5 Retrieved Chunks + Parent Expansion | **Enabled** (5-Stage Verifier Acceptance Gate) | Disabled | Measures verifier selectivity and rejection of ungrounded answers. |
| **E4** | **Retrieved + Verif + Repair** | Dense (`BGE-small`) + Parent-Child | Top-5 Retrieved Chunks + Parent Expansion | **Enabled** (5-Stage Verifier Acceptance Gate) | **Enabled** | Measures recovery rate on procedural errors (syntax, polarity, arithmetic). |
| **E5** | **Weighted RRF Comparison** | Weighted RRF (75% Dense / 25% BM25) + PC | Top-5 RRF Chunks + Parent Expansion | **Enabled** (5-Stage Verifier Acceptance Gate) | **Enabled** | Compares primary Dense+PC against hybrid lexical-dense retrieval. |

---

## 7. Metrics & Statistical Testing Plan

For paired comparisons ($E_2 \to E_3$, $E_3 \to E_4$, $E_4 \to E_5$):
- **Metrics**:
  - `Execution Accuracy`: $\frac{N_{\text{correct}}}{N_{\text{total}}}$
  - `Verified Answer Accuracy (Selective Precision)`: $\frac{N_{\text{correct \& verified}}}{N_{\text{verified}}}$
  - `Coverage`: $\frac{N_{\text{verified}}}{N_{\text{total}}}$
  - `Abstention Rate`: $\frac{N_{\text{abstained}}}{N_{\text{total}}}$
  - `False Acceptance Rate`: $\frac{N_{\text{incorrect \& verified}}}{N_{\text{verified}}}$
  - `Tokens Per Second`: $\frac{\text{Generated Tokens}}{\text{Generation Latency (s)}}$
  - `Peak VRAM Allocation`: Maximum allocated GPU memory in MB.
- **Statistical Tests**:
  - McNemar's paired test for binary classification outcomes.
  - Paired Wilcoxon signed-rank test for continuous latency differences.
  - Wilson score 95% confidence intervals on accuracy and coverage.

---

## 8. Stage 3.5.0 Deliverables Completed

- [x] `results/phase3_5_dataset_audit.md`
- [x] `results/phase3_5_pre_evaluation_audit.md`

We are now ready to proceed to **Stage 3.5.1: Real Model Loading and GPU Smoke Test** upon user approval.
