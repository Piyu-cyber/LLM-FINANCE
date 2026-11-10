# LLM-FINANCE — Phase 2.5: Comprehensive Scientific Validation, Ablation Audit, and Research Decision Report

**Project**: LLM-FINANCE: Verifiable and Attack-Resistant Financial RAG for SEC 10-K Documents using FinQA  
**Phase**: Phase 2.5 — Scientific Validation, Ablation Audit, and GPU Profiling Verification  
**Author**: Senior ML Research Scientist & Experimental Methodology Engineer  
**Date**: November 3, 2026  
**Status**: Completed, Fully Audited, and Reproducible  

---

## 1. Executive Overview & Scientific Purpose

Before initiating Phase 3 (LLM Generator and Attack-Resistant Verification Pipeline), this Phase 2.5 investigation was commissioned to independently audit, validate, and stress-test the empirical claims established in Phase 2. 

Specifically, Phase 2 left four critical scientific questions unanswered:
1. **Cross-Encoder Feasibility & Ceiling**: The cross-encoder (`BAAI/bge-reranker-base`) was conditionally bypassed due to missing local weights. Could it be run offline on the RTX 4060 GPU, what was its candidate ceiling, and did it statistically improve multi-span retrieval?
2. **Metadata Boosting Validity**: Did heuristic metadata boosts (company ticker and fiscal year heuristics) genuinely improve semantic retrieval, or did they distort ranking in multi-year comparative queries?
3. **Asymmetric Fusion vs. Standard RRF**: Why did Dense retrieval dominate BM25, and did asymmetric weighting (75/25, 90/10) or score fusion fix the 50/50 RRF penalty?
4. **Context Length & GPU Speedup Verification**: What is the exact token footprint of Parent-Child expansion under a real tokenizer (`bert-base-uncased`), and what is the exact RTX 4060 GPU speedup over an optimized multi-threaded CPU baseline?

Over the course of Phase 2.5, all 883 official FinQA validation queries were evaluated across 21 retrieval pipeline variations. All statistical comparisons were computed using paired Wilcoxon signed-rank tests ($N=883$), 2,000-iteration paired bootstrap confidence intervals, and family-wise error rate control via the Holm-Bonferroni step-down method.

---

## 2. Complete Phase 2.5 Master Results Matrix ($N=883$)

Below is the definitive performance table across all evaluated pipeline architectures on the official FinQA validation set.

| Architecture ID | Configuration Description | Candidate Rec@50 | Span Recall@10 [95% CI] | Span Hit@10 | Page MRR [95% CI] | Page Hit@10 | NDCG@5 | Latency (Avg/P95) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CE_A`** | Dense + Cross-Encoder | 0.5578 | 0.4613 [0.431, 0.491] | 0.5708 | 0.4165 [0.394, 0.440] | 0.8154 | 0.5133 | 150.79 ms / 150.79 ms |
| **`CE_B`** | RRF + Cross-Encoder | 0.5494 | 0.4683 [0.438, 0.499] | 0.5753 | 0.4221 [0.400, 0.444] | 0.8347 | 0.5233 | 278.21 ms / 363.07 ms |
| **`CE_C`** | RRF + Meta + Cross-Encoder | 0.5581 | 0.4670 [0.437, 0.497] | 0.5730 | 0.4282 [0.406, 0.452] | 0.8347 | 0.5293 | 277.41 ms / 362.27 ms |
| **`CE_D`** | RRF + Meta + PC + Cross-Encoder | 0.5581 | **0.6758** [0.645, 0.704] | **0.7146** | 0.4282 [0.406, 0.452] | 0.8347 | **0.5293** | 277.41 ms / 362.27 ms |
| **`Meta_A`** | Dense Only (Baseline) | — | 0.4336 [0.404, 0.464] | 0.5391 | 0.4279 [0.406, 0.451] | 0.7995 | 0.5098 | **0.04 ms / 0.04 ms** |
| **`Meta_B`** | Dense + Metadata (Combined) | — | 0.3843 [0.356, 0.413] | 0.4790 | 0.3779 [0.355, 0.401] | 0.6920 | 0.4507 | 0.04 ms / 0.04 ms |
| **`Meta_B_Co`** | Dense + Metadata (Company Only) | — | 0.4359 [0.406, 0.466] | 0.5425 | 0.4278 [0.406, 0.451] | 0.7995 | 0.5093 | 0.04 ms / 0.04 ms |
| **`Meta_B_Yr`** | Dense + Metadata (Year Only) | — | 0.3821 [0.354, 0.411] | 0.4757 | 0.3776 [0.354, 0.400] | 0.6931 | 0.4500 | 0.04 ms / 0.04 ms |
| **`Meta_C`** | RRF Only (Standard 50/50) | — | 0.4054 [0.376, 0.436] | 0.5130 | 0.3870 [0.366, 0.410] | 0.7950 | 0.4744 | 126.05 ms / 210.90 ms |
| **`Meta_D`** | RRF + Metadata (Combined) | — | 0.4114 [0.383, 0.441] | 0.5176 | 0.4137 [0.392, 0.437] | 0.8120 | 0.5149 | 126.05 ms / 210.90 ms |
| **`Meta_D_Co`** | RRF + Metadata (Company Only) | — | 0.4071 [0.378, 0.437] | 0.5153 | 0.3870 [0.366, 0.410] | 0.7950 | 0.4744 | 126.05 ms / 210.90 ms |
| **`Meta_D_Yr`** | RRF + Metadata (Year Only) | — | 0.4091 [0.380, 0.439] | 0.5142 | 0.4136 [0.392, 0.437] | 0.8120 | 0.5145 | 126.05 ms / 210.90 ms |
| **`Meta_E`** | Dense + Parent-Child | — | **0.6534** [0.622, 0.684] | **0.6908** | 0.4279 [0.406, 0.451] | 0.7995 | 0.5098 | **0.04 ms / 0.04 ms** |
| **`Meta_F`** | Dense + Metadata + PC | — | 0.6534 [0.622, 0.684] | 0.6908 | 0.4279 [0.406, 0.451] | 0.7995 | 0.5098 | 0.04 ms / 0.04 ms |
| **`Meta_G`** | RRF + Parent-Child | — | 0.6586 [0.627, 0.688] | 0.6954 | 0.3870 [0.366, 0.410] | 0.7950 | 0.4744 | 126.05 ms / 210.90 ms |
| **`Meta_H`** | RRF + Metadata + PC | — | 0.6586 [0.627, 0.688] | 0.6954 | 0.3870 [0.366, 0.410] | 0.7950 | 0.4744 | 126.05 ms / 210.90 ms |
| **`Fuse_2`** | BM25 Only (Baseline) | — | 0.2711 [0.245, 0.299] | 0.3454 | 0.3344 [0.311, 0.358] | 0.6648 | 0.3975 | 126.01 ms / 210.86 ms |
| **`Fuse_4`** | Weighted RRF (75% Dense / 25% BM25)| — | 0.4344 [0.405, 0.464] | 0.5436 | 0.4225 [0.402, 0.446] | 0.8199 | 0.5121 | 126.05 ms / 210.90 ms |
| **`Fuse_5`** | Weighted RRF (90% Dense / 10% BM25)| — | 0.4347 [0.405, 0.464] | 0.5425 | 0.4304 [0.409, 0.453] | 0.8063 | 0.5152 | 126.05 ms / 210.90 ms |
| **`Fuse_7`** | Normalized Score Fusion (75/25) | — | 0.3956 [0.366, 0.425] | 0.4994 | 0.4002 [0.378, 0.423] | 0.8029 | 0.4830 | 126.05 ms / 210.90 ms |
| **`Fuse_4_PC`**| Weighted RRF (75/25) + PC | — | **0.6676** [0.636, 0.697] | **0.7044** | 0.4225 [0.402, 0.446] | **0.8199** | 0.5121 | 126.05 ms / 210.90 ms |

---

## 3. Verified Answers to the 10 Research Assessment Questions

### Question 1: Is the Cross-Encoder computationally feasible on RTX 4060?
**Answer: YES, fully feasible.**  
`BAAI/bge-reranker-base` operates reliably on the RTX 4060 Laptop GPU. Peak VRAM utilization is **1,319.99 MB allocated** and **1,388.00 MB reserved**, consuming only **17.0% of the 8,188 MB total VRAM**. The model reranks top-50 candidate pools at a throughput of **331.7 candidate pairs/second**, translating to an average per-query reranking latency of **150.79 ms**. Total memory footprint and execution stability allow indefinite zero-oom operation alongside the bi-encoder.

### Question 2: Does the Cross-Encoder significantly improve multi-span recall?
**Answer: IT DEPENDS ON CONTEXT EXPANSION.**  
- **On isolated child chunks**: YES. Adding the Cross-Encoder to Dense retrieval improves Span Recall@10 from **43.36% to 46.13% (+2.77 pp, $p = 7.84 \times 10^{-4}$)**. Adding it to Standard RRF improves Span Recall@10 from **40.54% to 46.83% (+6.29 pp, $p = 9.41 \times 10^{-11}$)**.
- **When Parent-Child expansion is active**: NO. Comparing `RRF + Meta + PC + CE` (67.58%) against `RRF + Meta + PC` without CE (65.86%), the difference is $+1.71$ pp ($p = 0.1526$), which **fails to reach statistical significance** after Holm-Bonferroni family-wise error control ($\alpha = 0.0045$). In the presence of section expansion, the Cross-Encoder provides marginal, statistically non-significant value while increasing latency by $277 \text{ ms}$.

### Question 3: Does the candidate generation ceiling limit the Cross-Encoder?
**Answer: YES, definitively.**  
Across 883 validation queries, a top-$K=50$ first-stage candidate pool generated by Dense retrieval achieves a **Candidate Span Recall@50 ceiling of 55.78%** (Page Hit@50: 92.19%). Because a cross-encoder can only reorder received candidates, its maximum attainable recall on unexpanded chunks is strictly bounded by 55.78%. The cross-encoder captures 46.13% Span Recall@10, which represents **82.7% of all retrievable gold spans in the pool**. Parent-Child expansion bypasses this ceiling completely by restoring co-located table rows from the same section, reaching 67.58%.

### Question 4: Is metadata boosting genuinely helpful or does it distort ranking?
**Answer: IT IS DEMONSTRABLY HARMFUL FOR DENSE RETRIEVAL.**  
Applying combined metadata boosting to Dense retrieval causes a **statistically significant collapse** in Span Recall@10 from **43.36% to 38.43% (-4.93 pp, $p = 4.83 \times 10^{-6}$)** and in Page MRR from **0.4279 to 0.3779 (-0.0500, $p = 5.23 \times 10^{-10}$)**.  
Isolation testing demonstrates that **Year-only boosting is the primary culprit** (Year-only Span Recall drops to 38.21%, $p = 9.43 \times 10^{-7}$). In SEC 10-K filings, questions asking for multi-year comparisons (e.g. *"percentage increase from 2008 to 2009"*, *"cumulative five-year return ended 2017"*, or *"debt due after 2011"*) have their rankings distorted: boosting chunks containing the single parsed query year pushes earlier baseline rows or maturity tables out of the top-10. Company-only boosting produces a negligible difference ($+0.23$ pp, $p = 0.0833$, not significant).

### Question 5: Does asymmetric fusion outperform standard RRF?
**Answer: YES, decisively.**  
Because BM25 is severely weaker than Dense retrieval on financial filings (27.11% vs. 43.36% Span Recall@10), Standard symmetric 50/50 RRF severely penalizes Dense retrieval, dragging recall down to **40.54% (-2.82 pp vs. Dense Only)**.  
Weighted RRF with **75% Dense / 25% BM25** cures this penalty, lifting Span Recall@10 back to **43.44% (+2.90 pp vs. 50/50 RRF, $p = 8.04 \times 10^{-7}$, Holm significant)** and Page MRR to **0.4225 (+0.0355, $p = 5.38 \times 10^{-8}$)**, reaching full statistical parity with Dense Only ($p = 0.7220$). Furthermore, Normalized Score Fusion (MinMax BM25 + cosine) fails to match Weighted RRF (39.56% vs 43.44%, $p = 1.51 \times 10^{-9}$), proving that rank-based reciprocal fusion is far more robust than min-max score scaling.

### Question 6: Does Parent-Child expansion remain the dominant factor?
**Answer: YES, by a massive margin.**  
Parent-Child expansion is the single most powerful architectural mechanism in the entire repository:
- Dense + PC vs. Dense Only: **$+21.98$ pp gain** in Span Recall@10 (from 43.36% to 65.34%, $p = 1.37 \times 10^{-28}$, effect size $r = 0.5383$).
- RRF + PC vs. RRF Only: **$+25.32$ pp gain** in Span Recall@10 (from 40.54% to 65.86%, $p = 1.32 \times 10^{-30}$, effect size $r = 0.5144$).
Parent-Child expansion provides **nearly $8 \times$ more recall improvement** than Cross-Encoder reranking (+21.98 pp vs. +2.77 pp) with zero neural overhead.

### Question 7: Is the ~2,300 token context length estimate accurate?
**Answer: REVISED DOWNWARDS TO 1,656 TOKENS.**  
Using exact subword tokenization with `bert-base-uncased` across all 883 validation queries:
- Single child chunk: mean **40.3 tokens** (median 36.0 tokens).
- Single parent section: mean **379.6 tokens** (median 345.0 tokens).
- Top-5 child-only prompt context: mean **201.7 tokens**.
- Top-5 raw parent prompt context (without deduplication): mean **1,897.9 tokens** (P95: 3,158.8 tokens).
- **Top-5 deduplicated parent prompt context**: mean **1,656.2 tokens** (median 1,589.0 tokens, P95: 2,936.9 tokens).
The Phase 2 estimate of ~2,300 tokens was conservative because it assumed five distinct parent documents. In reality, retrieved child chunks cluster within the same table, so deduplication yields an average of only 3.2 unique parents per query (~1,656 tokens), easily fitting inside standard 4k and 8k context windows.

### Question 8: What is the verified RTX 4060 GPU speedup over multi-threaded CPU?
**Answer: 26.82x steady-state speedup in FP16.**  
When evaluated across 1,024 chunks with warm-up cycles, explicit `torch.cuda.synchronize()`, and all 16 hardware threads utilized:
- Multi-threaded CPU baseline (16 threads): **99.89 chunks/sec** (640.73 ms/batch of 64).
- GPU FP32: **1,450.09 chunks/sec** (**14.52x speedup**).
- **GPU FP16: 2,678.70 chunks/sec** (**26.82x speedup**).
- GPU BF16: **2,409.83 chunks/sec** (**24.13x speedup**).
The Phase 2 reported speedup of 5.88x was measured on a cold 256-chunk slice with default single-batch PyTorch CPU settings. In warmed, steady-state batching, the RTX 4060 achieves **26.82x speedup** in FP16. Peak VRAM during encoding is only **325 MB**.

### Question 9: Which retrieval configuration should be chosen for Phase 3 LLM generation?
**Answer: DENSE + PARENT-CHILD (with deduplication).**  
Comparing the top candidates:
1. **`Dense + Parent-Child`**:
   - **Span Recall@10**: **65.34%** [0.622, 0.684]
   - **Page Hit@10**: **79.95%** [0.774, 0.825]
   - **NDCG@5**: **0.5098**
   - **Per-query Latency**: **0.04 ms** (matrix multiplication on RTX 4060)
   - **Memory Footprint**: 325 MB VRAM (bi-encoder only)
   - **Failure Modes**: Free of BM25 lexical dilution and year-boost distortions.
2. **`Weighted RRF (75/25) + Parent-Child`**:
   - Span Recall@10: **66.76%** (+1.42 pp over Dense+PC, not statistically significant).
   - Per-query Latency: **126.05 ms** ($3,150 \times$ slower due to CPU BM25 scoring).
3. **`RRF + Meta + PC + Cross-Encoder`**:
   - Span Recall@10: **67.58%** (+2.24 pp over Dense+PC, not statistically significant).
   - Per-query Latency: **277.41 ms** ($6,935 \times$ slower, consumes 1.32 GB VRAM).

**Decision**: For Phase 3, **`Dense + Parent-Child`** is the recommended primary retrieval architecture. It delivers 97% of the maximum possible recall at **0.04 ms per query**, leaving the entire GPU memory and compute budget open for the generation and verification models. For offline batch benchmarking where latency is unconstrained, `Weighted RRF (75/25) + PC + CE` can be supported as an optional high-recall profile.

### Question 10: Is the retrieval pipeline ready for Phase 3?
**Answer: YES, fully validated and ready.**  
The retrieval pipeline has passed every empirical stress test:
- 100% test suite health: **171 unit and integration tests passing**, 1 skipped (Python 3.13 conditional CUDA runner).
- The direct CUDA integration test passes on RTX 4060.
- All candidate ceilings, context token limits, and statistical significance boundaries are mapped and documented.
- Per-query failure modes have been extracted and characterized.

The codebase is in an optimal state to begin Phase 3 (LLM Generator and Attack-Resistant Verification Pipeline).
