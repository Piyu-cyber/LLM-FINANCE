# LLM-FINANCE — Phase 2.5: Parent-Child Context Length Verification Report

**Project**: LLM-FINANCE: Verifiable and Attack-Resistant Financial RAG for SEC 10-K Documents  
**Phase**: 2.5 — Scientific Validation, Ablation Audit, and GPU Profiling Verification  
**Evaluation Scope**: Token-Level Length Measurement Using `bert-base-uncased` Across 883 Queries  
**Date**: November 3, 2026  
**Status**: Verified & Reproducible  

---

## 1. Executive Summary

In Phase 2, Parent-Child retrieval was identified as the single most impactful architectural enhancement, lifting Span Recall@10 from ~43% to ~65%. However, a critical open question remained: **what is the exact context length footprint of expanded parent sections, and does it risk overflowing LLM context windows or inflating generator inference cost in Phase 3?**

In Phase 2, a rough character-based approximation estimated the expanded top-5 parent context at ~2,300 tokens. In Phase 2.5, we implemented exact subword tokenization using `transformers.AutoTokenizer.from_pretrained('bert-base-uncased')` across all 25,007 corpus chunks and all 883 validation queries.

### Key Empirical Findings:
1. **Child Chunks are Extremely Compact**:
   - Single child chunks average **154.0 characters** and **40.3 subword tokens** (median: 36 tokens, P95: 82 tokens).
   - An assembled top-5 child-only prompt consumes on average only **201.7 tokens** (median: 182 tokens, P95: 377.9 tokens).
2. **Parent Sections Average Under 400 Tokens**:
   - Single parent documents (the complete table schedule or narrative section) average **1,530.1 characters** and **379.6 subword tokens** (median: 345 tokens, P95: 712 tokens, max: 1,839 tokens).
3. **Deduplication Saves Significant Context Space**:
   - Top-5 child chunks frequently originate from the *same* parent table or page.
   - Without deduplication, naively concatenating 5 expanded parents yields a mean of **1,897.9 tokens** (P95: 3,158.8 tokens).
   - When duplicate parent section IDs are deduplicated, the top-5 assembled context drops to **1,656.2 tokens** (median: 1,589 tokens, P95: 2,936.9 tokens), representing a **12.7% context savings** and completely eliminating redundant table rows from the prompt.
4. **LLM Context Window Feasibility**:
   - At a mean of 1,656 tokens and P95 of 2,937 tokens, the deduplicated parent context fits comfortably within standard 4,096-token context windows, and leaves ample room for multi-step reasoning chains in 8k, 16k, or 32k models.

---

## 2. Granular Token & Character Distributions

All measurements were calculated across the complete 25,007-chunk corpus and 883 FinQA validation queries using `bert-base-uncased` with fast Rust tokenization.

### A. Single Chunk / Section Level

| Entity Level | Characters (Mean) | Characters (Median) | Tokens (Mean) | Tokens (Median) | Tokens (P95) | Tokens (Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Child Chunk** (Row/Sentence) | 154.0 | 127.0 | 40.3 | 36.0 | 82.0 | 207 |
| **Parent Section** (Full Table/Text) | 1,530.1 | 1,380.0 | 379.6 | 345.0 | 712.0 | 1,839 |

### B. Per-Query Assembled Top-5 Context Budget ($N=883$ Validation Queries)

| Context Assembly Strategy | Tokens (Mean) | Tokens (Median) | Tokens (P95) | Tokens (Max) | Compression Ratio vs Child |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Child-Only Top-5** | 201.7 | 182.0 | 377.9 | 950 | $1.00 \times$ (Baseline) |
| **Parent Raw Top-5** (No Dedup) | 1,897.9 | 1,765.0 | 3,158.8 | 5,136 | $9.41 \times$ |
| **Parent Deduplicated Top-5** | **1,656.2** | **1,589.0** | **2,936.9** | **5,136** | **$8.21 \times$** |

---

## 3. Reconciling Phase 2 Estimates with Exact Phase 2.5 Measurement

In the Phase 2 report, section 6 stated:
> *"Parent-Child expansion expands retrieved child rows to their full section text, lifting multi-span recall from 43% to 65%, with an estimated top-5 context length of ~2,300 tokens."*

### Why the Phase 2 Estimate Differed:
1. **Character-to-Token Ratio Assumption**:
   Phase 2 used a rough heuristic of $1 \text{ token} \approx 4 \text{ characters}$. In SEC 10-K financial tables, however, financial text contains extensive whitespace, semicolons, dollar signs, and structured delimiters (e.g., `date the 31-dec-2012 of citi is 100.0 ;`). The true measured ratio in `bert-base-uncased` is **$4.03 \text{ characters per token}$** for parents and **$3.82 \text{ characters per token}$** for child chunks.
2. **Failure to Account for Cross-Child Co-location (Deduplication)**:
   In Phase 2, the ~2,300 token estimate assumed 5 distinct parent sections ($5 \times \sim 460$ tokens). In practice, because Dense and RRF retrieval retrieve multiple child rows from the *same* high-scoring table schedule, an average top-5 retrieved set contains only **$3.2$ unique parent sections**.
3. **True Deduplicated Burden**:
   Deduplication reduces the context from the naive ~2,300 estimate down to **1,656.2 tokens**, well within safe limits for generation.

---

## 4. Architectural Implications for Phase 3 LLM Generation

### 1. Context Window Budget Allocation (for 4,096-Token Models)
Even for smaller LLMs with strict 4k context limits (e.g., `Llama-3-8B-4k` or `Mistral-7B-Instruct-v0.2`):
- **System Prompt & Verification Schema**: ~400 tokens
- **Deduplicated Retrieved Context (Top-5 Parent)**: ~1,656 tokens (Mean) / ~2,937 tokens (P95)
- **Question & Few-Shot Demonstrations**: ~300 tokens
- **Available Generation Budget (AST & Calculation)**: **~1,740 tokens (Mean) / ~459 tokens (P95)**
At P95, the context comfortably fits without truncation. Only in 3 out of 883 queries ($0.34\%$) does raw context exceed 4,096 tokens, which can be capped cleanly by taking top-3 parents instead of top-5 when character length exceeds 12,000 characters.

### 2. Context Window Budget Allocation (for 8k / 32k / 128k Models)
For modern models with 8k+ windows (e.g., `Qwen2.5-7B-Instruct-32k` or `Llama-3.1-8B-128k`):
- Top-5 deduplicated parent context consumes only **$5.2\%$ of a 32k window**, completely eliminating context length pressure.
- The generator receives complete, pristine accounting tables with all surrounding footnote disclosures, providing the necessary verbatim grounding spans for operand verification.

### 3. Conclusion & Implementation Recommendation
- **Always deduplicate parent sections** before assembling the LLM context prompt.
- **Retain Parent-Child expansion**: The $+22$ pp gain in Span Recall@10 is achieved at a manageable context cost of ~1,656 tokens.
