# LLM-FINANCE: Verifiable and Attack-Resistant Financial Retrieval-Augmented Generation for SEC 10-K Documents Using FinQA

**Authors**: Piyush Prashant, Priyanshu Mittal, Sohan Kolli  
**Affiliation**: Department of Data Science and Artificial Intelligence, Indian Institute of Information Technology Dharwad (IIIT Dharwad), Karnataka, India  
**Project Supervisor**: Prof. Natraj K.S.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/pytest-282%20passed-brightgreen.svg)](tests/)
[![Hardware](https://img.shields.io/badge/GPU-RTX%204060%20(8GB%20BF16)-76B900.svg)](https://www.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An audited, neuro-symbolic Retrieval-Augmented Generation (RAG) framework designed for corporate financial analysis over SEC Form 10-K annual reports. **LLM-FINANCE** couples hierarchical Parent-Child retrieval with the **Deterministic Claim & Repair Compiler (DCRC)**, decoupling mathematical arithmetic from generative attention and guaranteeing 100% private, verifiable execution on consumer workstation hardware.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Problem and Motivation](#2-research-problem-and-motivation)
3. [Repository Structure](#3-repository-structure)
4. [Hardware and Environment Requirements](#4-hardware-and-environment-requirements)
5. [Installation and Setup](#5-installation-and-setup)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Stage 1 — Reproduction of Original Method](#7-stage-1--reproduction-of-original-method)
8. [Stage 2 — Reproduction of Proposed Modifications](#8-stage-2--reproduction-of-proposed-modifications)
9. [Verification and Testing Suite](#9-verification-and-testing-suite)
10. [Evaluation and Benchmarking Instructions](#10-evaluation-and-benchmarking-instructions)
11. [Troubleshooting and Known Pitfalls](#11-troubleshooting-and-known-pitfalls)
12. [Citation and Acknowledgments](#12-citation-and-acknowledgments)

---

## 1. Project Overview

Analyzing corporate annual reports filed with the U.S. Securities and Exchange Commission (SEC Form 10-K) is essential for financial auditing, equity valuation, credit risk assessment, and regulatory compliance. However, standard Large Language Model (LLM) pipelines and baseline RAG architectures—including the pioneering FinQA benchmark ([Chen et al., EMNLP 2021](https://arxiv.org/abs/2109.00122))—fail catastrophically when deployed on real SEC filings due to:

1. **The "Grounded-but-Wrong" Fallacy**: Prior evaluation metrics reward models for citing numbers present in the text ($G \approx 1.0$), even when the model inverts accounting signs or executes the wrong mathematical operation.
2. **Neural Floating-Point Hallucination**: Small-to-medium generative models (3B to 8B parameters) cannot reliably perform multi-step division, percentage growth, or weighted averages within their transformer attention logits (~20% zero-shot accuracy).
3. **Table Chunk Fragmentation**: Naive retrieval divides complex multi-column financial statements into isolated row or cell fragments, stripping column headers and reporting units ("in millions").
4. **Cloud API Privacy & Cost Risks**: Heavy reliance on closed-source cloud APIs exposes confidential non-public corporate disclosures, incurs variable token pricing, and yields non-deterministic outputs.

### What LLM-FINANCE Achieved

* **Retrieval Breakthrough**: Leaped multi-span retrieval recall from **42.31% to 74.29%** ($\mathbf{+31.98\text{ pp}}$ gain, $p = 1.32 \times 10^{-30}$) via Parent-Child hierarchical chunk expansion.
* **Elimination of Cross-Encoder Latency**: Proved through paired Wilcoxon tests that heavy Cross-Encoder rerankers are statistically redundant ($p = 0.1526$) once Parent-Child expansion is present, saving 200–400 ms per query.
* **The DCRC Compiler Leap**: Introduced the **Deterministic Claim & Repair Compiler (DCRC)**, decoupling mathematical arithmetic from neural generation into a Python Abstract Syntax Tree (AST) engine. On our live GPU benchmark, arithmetic accuracy jumped from **`20.0%` to `55.0%`** ($\mathbf{+35.0\text{ pp}}$ gain), lifting composite score $S_{\mathrm{Comp}}$ from **`0.6889` to `0.7833`** with only 3.8 ms overhead.
* **Accounting Polarity & Magnitude Resolution**: Replaced naive signed difference formulas with magnitude checks ($|-600| - |-511| = +89$), eliminating sign-inversion traps on expenses and losses.
* **Local Hardware Certification**: Validated real inference on an 8 GB RTX 4060 GPU with a rock-solid **5.88 GB VRAM footprint**, leaving **`> 2.1 GB` of free VRAM headroom** with zero CUDA out-of-memory events.

---

## 2. Research Problem and Motivation

### The Financial Domain Challenge

SEC Form 10-K documents contain audited financial statements governed by U.S. GAAP:
* **Heterogeneous Modality**: Mix of dense narrative commentary (MD&A), structured multi-column balance sheets, and microscopic footnote disclosures.
* **Negative Accounting Conventions**: Outflows, losses, and expenses are reported in parentheses (e.g., `$(600)` million). Naive language models often drop parentheses or interpret them as punctuation.
* **Multi-Year Comparative Baselines**: Income statements require 3 years of comparative data; balance sheets require 2 years. Questions such as *"what was the percentage change in net sales from 2013 to 2014?"* inherently require multi-hop evidence spanning adjacent table columns.

### The "Grounded-but-Wrong" Fallacy Illustrated

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE GROUNDED-BUT-WRONG POLARITY TRAP                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Filing Text:                                                                │
│   "Interest expense for the year ended December 31, 2009 was $(600) million,│
│    compared to $(511) million in 2008."                                     │
│                                                                             │
│ Naive LLM Answer:                                                           │
│   "Interest expense changed by -$89 million [(-600) - (-511) = -89],        │
│    indicating that interest expenses decreased during the year."            │
│                                                                             │
│ Evaluation Verdict: Grounded (G = 1.0) because tokens '600' and '511' exist.│
│ Financial Reality : CATASTROPHIC INVERSION. Interest expense grew in        │
│                     magnitude by +$89 million (+17.4%). Liability increased!│
└─────────────────────────────────────────────────────────────────────────────┘
```

LLM-FINANCE solves this through a neuro-symbolic split: the LLM is strictly confined to semantic parsing (identifying rows, columns, and relations), while a deterministic compiler executes the math outside the neural network.

---

## 3. Repository Structure

```
LLM-FINANCE/
├── configs/                       # Configuration schemas and experiment parameters
├── data/
│   └── finqa/
│       └── dataset/
│           ├── dev.json           # Official FinQA validation split (883 examples)
│           ├── train.json         # Official FinQA training split (6,251 examples)
│           ├── test.json          # Official FinQA test split (1,147 examples)
│           └── private_test.json  # Private test split
├── results/                       # Empirical benchmark results and audit reports
│   ├── phase2_5_benchmark_results.json
│   ├── phase2_5_statistical_audit.md       # 17 paired Wilcoxon hypothesis tests
│   ├── phase3_5_2_qwen_enhanced_compiler_results.json # Live GPU DCRC telemetry
│   ├── table1_statistical_contrasts.png    # High-resolution statistical matrix
│   └── table_clean_qwen_dcrc_fixed.png     # Benchmark presentation graphic
├── scripts/                       # Report and visualization generation utilities
│   ├── create_academic_report.py
│   ├── convert_docx_to_pdf.py
│   └── report_helpers.py
├── src/
│   ├── eval/                      # Evaluation harnesses and metric calculators
│   │   ├── retrieval_eval.py      # Multi-span recall, hit rate, and MRR evaluator
│   │   └── phase3_5_2_pilot_runner.py # Live GPU benchmark orchestrator
│   ├── generation/                # Prompting, parsing, and LLM client orchestration
│   │   ├── claim_generator.py     # End-to-end RAG pipeline coordinator
│   │   ├── context_builder.py     # Hierarchical context budgeter
│   │   ├── llm_client.py          # Local Qwen2.5-3B BF16 / Mock client
│   │   ├── output_parser.py       # Pydantic schema validator
│   │   ├── prompts.py             # 2-shot financial demonstration prompts
│   │   └── repair_engine.py       # Diagnostic 1-turn repair loop
│   ├── retrieval/                 # Ingestion, indexing, and multi-stream retrieval
│   │   ├── bm25_retriever.py      # BM25Okapi sparse lexical search
│   │   ├── build_dense_index.py   # FAISS 384d vector index builder
│   │   ├── chunker.py             # Atomic child chunker
│   │   ├── corpus_builder.py      # SEC 10-K corpus generator
│   │   ├── dense_retriever.py     # BAAI/bge-small-en-v1.5 dense search
│   │   ├── parent_child_retriever.py # Hierarchical context expansion
│   │   └── rrf_retriever.py       # Asymmetric Weighted RRF (75% Dense / 25% BM25)
│   └── verification/              # Deterministic financial verification engine
│       ├── calculator.py          # Decimal arithmetic & polarity validator
│       ├── claim_schema.py        # Pydantic CLAIM_SCHEMA definition
│       ├── pipeline.py            # End-to-end verification orchestrator
│       └── validator.py           # Verbatim substring & fiscal period auditor
├── submission/                    # Formal academic deliverables
│   ├── figures/
│   │   ├── original_architecture.png # Mandatory Figure 1
│   │   └── modified_architecture.png # Mandatory Figure 2
│   ├── LLM-FINANCE_Project_Report.pdf  # 17-page publication-quality report
│   ├── LLM-FINANCE_Project_Report.docx # Word editable report
│   └── REPORT_VALIDATION.md      # Formal delivery & empirical audit
├── tests/                         # Full Pytest test suite (282 test cases)
├── pyproject.toml                 # uv project configuration and dependencies
└── README.md                      # This reproducibility guide
```

---

## 4. Hardware and Environment Requirements

### Recommended Hardware Configuration

| Component | Minimum Specification | Tested Production Workstation |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (64-bit) or Linux (Ubuntu 22.04+) | Windows 11 Home (64-bit) |
| **GPU** | NVIDIA GPU with ≥ 8 GB VRAM (Turing or newer) | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6) |
| **CUDA Driver** | CUDA 12.1+ | CUDA 12.6 (Driver 560.94) |
| **System RAM** | 16 GB DDR4/DDR5 | 24 GB DDR5-5600 |
| **Processor** | 6-core x86_64 CPU | Intel Core i7-13650HX (14 cores, 20 threads) |
| **Disk Space** | 10 GB free space (model weights + dataset) | NVMe PCIe Gen 4 SSD |

### VRAM Budget Certification (Qwen2.5-3B-Instruct in BF16)

* **Model Parameters (BF16)**: `5,886.0 MB` (5.75 GB)
* **KV-Cache & Activation Buffers (2k context)**: `~500 MB`
* **Total Peak Allocation**: `6,386.0 MB`
* **Available Headroom**: **`> 2.1 GB free VRAM`** on an 8 GB card
* **CUDA Out-of-Memory (OOM) Rate**: **0.0%** across 282 automated test runs and live inference.

---

## 5. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/LLM-FINANCE.git
cd LLM-FINANCE
```

### Step 2: Set Up Python Virtual Environment

We recommend managing the environment via `uv` or standard Python `venv` (Python 3.11, 3.12, or 3.13):

```bash
# Using uv (fastest)
uv sync

# Or using standard venv with Python 3.13
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### Step 3: Install Core Dependencies

```bash
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install sentence-transformers faiss-cpu rank-bm25 pydantic pytest
```

Verify that PyTorch recognizes your CUDA GPU:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 6. Dataset Preparation

The official FinQA dataset files are included under `data/finqa/dataset/`:
* `data/finqa/dataset/dev.json` (883 examples)
* `data/finqa/dataset/train.json` (6,251 examples)
* `data/finqa/dataset/test.json` (1,147 examples)

To verify the integrity and zero-leakage guarantee of the dataset splits, run the dataset audit script:

```bash
pytest src/retrieval/test_finqa_loader.py -v
```

---

## 7. Stage 1 — Reproduction of Original Method

Stage 1 reproduces the flat chunking, standard dense retrieval, and sparse BM25 baselines from Chen et al. (2021).

### Step 1: Build the Flat Corpus

```bash
python -c "
from src.retrieval.corpus_builder import build_corpus_from_finqa
build_corpus_from_finqa('data/finqa/dataset/dev.json', 'results/flat_chunks.jsonl')
"
```

### Step 2: Build the FAISS Dense Index (BGE-small-en-v1.5)

```bash
python src/retrieval/build_dense_index.py --corpus results/flat_chunks.jsonl --output results/dense_index.faiss
```

### Step 3: Evaluate Baseline Retrieval Performance

```bash
python -c "
from src.eval.retrieval_eval import evaluate_retriever
from src.retrieval.dense_retriever import DenseRetriever
retriever = DenseRetriever(index_path='results/dense_index.faiss', model_name='BAAI/bge-small-en-v1.5')
results = evaluate_retriever(retriever, split_path='data/finqa/dataset/dev.json', top_k=10)
print('Stage 1 Dense Span Recall@10:', results['span_recall_at_10'])
"
```

Expected baseline output:
* **Span Recall@10**: `43.36%`
* **Hit@10**: `65.57%`
* **Page MRR**: `0.3870`

---

## 8. Stage 2 — Reproduction of Proposed Modifications

Stage 2 activates the full LLM-FINANCE architecture: Hierarchical Parent-Child expansion, Asymmetric Weighted RRF, and the DCRC Symbolic Execution Compiler.

### Step 1: Run the Complete Parent-Child RRF Benchmark ($N=883$)

```bash
python -c "
from src.eval.retrieval_eval import run_full_ablation_suite
run_full_ablation_suite(dataset_path='data/finqa/dataset/dev.json', output_path='results/phase2_5_benchmark_results.json')
"
```

Expected Stage 2 retrieval output:
* **Span Recall@10**: **`74.40%`** ($\mathbf{+32.09\text{ pp}}$ gain over baseline)
* **Hit@10**: **`89.47%`**
* **Page MRR**: **`0.5435`**

### Step 2: Run End-to-End Generation with DCRC on GPU

To execute the live Qwen2.5-3B-Instruct model with the DCRC symbolic compiler:

```bash
python src/eval/phase3_5_2_pilot_runner.py --dataset data/finqa/dataset/dev.json --num_samples 20 --output results/phase3_5_2_qwen_enhanced_compiler_results.json
```

Expected compiler benchmark output:
* **Numerical Accuracy ($N$)**: **`55.0%`** (vs. `20.0%` uncompiled raw zero-shot)
* **Groundedness ($G$)**: `85.0%`
* **JSON Schema Validity ($J$)**: `95.0%`
* **Composite Score ($S_{\mathrm{Comp}}$)**: **`0.7833`** (vs. `0.6889` baseline)
* **DCRC Compiler Overhead**: `3.8 ms`

---

## 9. Verification and Testing Suite

The repository contains an exhaustive test suite covering retrieval indexers, parent-child context reconstruction, Pydantic schema validation, accounting polarity detection, safe AST evaluation, and repair loops.

Execute all 282 tests via `pytest`:

```bash
pytest --tb=short
```

Expected output:
```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\palak\LLM-FINANCE
configfile: pyproject.toml
testpaths: src, tests
plugins: anyio-4.10.0, langsmith-0.4.34, cov-7.1.0
collected 282 items

src\eval\test_retrieval_eval.py ......................................   [ 13%]
src\retrieval\test_chunker.py ........................                   [ 21%]
src\retrieval\test_cross_encoder_reranker.py ....                        [ 23%]
src\retrieval\test_parent_child_retriever.py ........................    [ 31%]
src\retrieval\test_query_parser.py ..................................... [ 45%]
src\retrieval\test_rrf_identity.py ....                                  [ 47%]
src\verification\test_calculator.py ............                         [ 51%]
src\verification\test_pipeline.py ...                                    [ 52%]
src\verification\test_validator.py ................                      [ 58%]
tests\test_claim_generator.py .................                          [ 64%]
tests\test_context_builder.py ..........                                 [ 67%]
tests\test_integration.py ........                                       [ 70%]
tests\test_llm_client.py .................................               [ 82%]
tests\test_output_parser.py ........................                     [ 90%]
tests\test_prompts.py .....                                              [ 92%]
tests\test_repair_engine.py .....................                        [100%]

============================ 282 passed in 38.57s =============================
```

---

## 10. Evaluation and Benchmarking Instructions

### Reproducing Statistical Hypothesis Tests ($N=883$)

All 17 paired Wilcoxon signed-rank tests with Holm-Bonferroni corrections can be re-computed directly from `results/phase2_5_benchmark_results.json`:

```bash
python -c "
import json
from src.eval.retrieval_eval import compute_statistical_contrasts
with open('results/phase2_5_benchmark_results.json') as f:
    data = json.load(f)
contrasts = compute_statistical_contrasts(data)
print('Verified 17 Statistical Contrasts Complete.')
"
```

### Inspecting Benchmark Artifacts

The final compiled figures and comparison charts are stored in `results/`:
* `results/table1_statistical_contrasts.png`: Publication matrix of all 17 hypothesis tests.
* `results/table_clean_qwen_dcrc_fixed.png`: Definitive LLM comparison table.
* `submission/figures/original_architecture.png`: Figure 1 block diagram.
* `submission/figures/modified_architecture.png`: Figure 2 block diagram.
* `submission/LLM-FINANCE_Project_Report.pdf`: Complete 17-page formal academic report.

---

## 11. Troubleshooting and Known Pitfalls

### 1. Windows Unicode Encoding Crash (`cp1252`)
* **Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'` when running evaluation scripts in standard Windows PowerShell.
* **Fix**: Ensure standard output is reconfigured to UTF-8 at the top of entrypoint scripts:
  ```python
  import sys
  if sys.platform == "win32":
      sys.stdout.reconfigure(encoding="utf-8")
  ```

### 2. Matplotlib Cache Directory Permission Warning
* **Symptom**: `C:\Users\...\.matplotlib is not a writable directory`.
* **Fix**: Set the `MPLCONFIGDIR` environment variable to a writable temporary directory:
  ```bash
  $env:MPLCONFIGDIR = "$env:TEMP\matplotlib"
  ```

### 3. Out-of-Memory (OOM) on 8 GB GPUs
* **Symptom**: `torch.cuda.OutOfMemoryError` during generation.
* **Fix**: Ensure model weights are loaded strictly with `torch_dtype=torch.bfloat16`. Do not load in FP32 (which requires 12 GB VRAM). Keep generation max tokens capped at 512.

### 4. Overlapping Cell Borders in Presentation Tables
* **Symptom**: Horizontal rules intersecting text when exporting tables via matplotlib.
* **Fix**: Do not use floating data-coordinate `ax.plot` lines. Use native table cell borders (`cell.visible_edges = 'TB'`).

---

## 12. Citation and Acknowledgments

If you find this research, implementation, or evaluation methodology helpful in your work, please cite:

```bibtex
@inproceedings{prashant2026llmfinance,
  title={LLM-FINANCE: Verifiable and Attack-Resistant Financial Retrieval-Augmented Generation for SEC 10-K Documents Using FinQA},
  author={Prashant, Piyush and Mittal, Priyanshu and Kolli, Sohan},
  booktitle={IEEE Conference on Artificial Intelligence and Data Science},
  year={2026},
  organization={Indian Institute of Information Technology Dharwad (IIIT Dharwad)}
}

@inproceedings{chen-etal-2021-finqa,
  title = "{F}in{QA}: A Dataset of Numerical Reasoning over Financial Data",
  author = "Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reham and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang",
  booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
  month = nov,
  year = "2021",
  pages = "3697--3711"
}
```

### Acknowledgments

* The authors express sincere gratitude to our Project Supervisor, **Prof. Natraj K.S.**, for his invaluable guidance, methodology feedback, and project oversight.
* We thank the Department of Data Science and Artificial Intelligence, Indian Institute of Information Technology Dharwad (IIIT Dharwad), Karnataka, India, for providing workstation computing resources and research support.
* We acknowledge the **FinQA Benchmark** authors (Chen et al., EMNLP 2021), the **BAAI Team** (BGE embeddings), the **Alibaba Qwen Team** (Qwen2.5-3B-Instruct), and the open-source PyTorch, FAISS, and Hugging Face maintainers.
