# LLM-FINANCE — Phase 2.5: Asymmetric Fusion & RRF Ablation Report

**Project**: LLM-FINANCE: Verifiable and Attack-Resistant Financial RAG for SEC 10-K Documents  
**Phase**: 2.5 — Scientific Validation, Ablation Audit, and GPU Profiling Verification  
**Evaluation Scope**: Dense vs. BM25 Asymmetry, Standard RRF, Weighted RRF, and Normalized Score Fusion  
**Date**: November 3, 2026  
**Status**: Verified & Reproducible (Evaluated across all $N=883$ FinQA validation queries)  

---

## 1. Executive Summary

Standard Reciprocal Rank Fusion (RRF) assumes symmetric reliability between constituent retrieval models. In financial information retrieval over SEC 10-K filings, however, this assumption is strongly violated:
- **Dense Retrieval** (`BAAI/bge-small-en-v1.5`) achieves **43.36% Span Recall@10** and **0.4279 Page MRR**.
- **Lexical Retrieval** (BM25) achieves only **27.11% Span Recall@10** and **0.3344 Page MRR**.

Because BM25 is dramatically weaker than Dense retrieval (performing $-16.25$ pp lower in Span Recall@10), giving equal 50/50 voting weight to both retrievers in Standard RRF actively **drags down overall retrieval performance**: Standard RRF (50/50) drops Span Recall@10 to **40.54% (-2.82 pp vs Dense Only)** and Page MRR to **0.3870 (-0.0409 vs Dense Only)**.

To resolve this asymmetry, we evaluated:
1. **Weighted RRF** with asymmetric weighting:
   $$RRF(d) = w_{\text{dense}} \cdot \frac{1}{k + r_{\text{dense}}(d)} + w_{\text{bm25}} \cdot \frac{1}{k + r_{\text{bm25}}(d)}$$
   for weights $(75/25)$, $(90/10)$, and $(50/50)$.
2. **Normalized Score Fusion**: MinMax-normalized BM25 scores linearly combined with cosine Dense similarity scores.
3. Both fusion mechanisms augmented with **Parent-Child expansion**.

### Key Conclusions:
- **Weighted RRF (75/25) Cures the RRF Penalty**: Adjusting weights to 75% Dense and 25% BM25 significantly restores Span Recall@10 from **40.54% to 43.44% (+2.90 pp, $p = 8.04 \times 10^{-7}$, Holm significant)** and Page MRR from **0.3870 to 0.4225 (+0.0355, $p = 5.38 \times 10^{-8}$)**, matching Dense Only (**43.44% vs 43.36%, $p = 0.7220$, not significant**).
- **Weighted RRF (90/10)** further peaks Page MRR at **0.4304** and Span Recall@10 at **43.47%**.
- **Normalized Score Fusion Underperforms Weighted RRF**: Normalized Score Fusion (75/25) yields **39.56% Span Recall@10**, performing significantly worse than Weighted RRF 75/25 ($-3.89$ pp, $p = 1.51 \times 10^{-9}$, Holm significant). Unbounded BM25 score distributions distort rank order even after linear MinMax scaling.
- **Parent-Child Combination**: Combining Weighted RRF (75/25) with Parent-Child expansion achieves **66.76% Span Recall@10** (Page Hit@10 = **81.99%**), outperforming Dense+PC (**65.34%**, $+1.42$ pp).

---

## 2. Comprehensive Fusion Benchmark Matrix

The table below contrasts all fusion architectures across the official 883 FinQA validation queries.

| Configuration ID | Architecture Description | Span Recall@10 [95% CI] | Span Hit@10 | Page MRR [95% CI] | Page Hit@10 | NDCG@5 | Latency (avg/p95) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Fuse_1`** | Dense Only (Baseline) | 0.4336 [0.404, 0.464] | 0.5391 | 0.4279 [0.406, 0.451] | 0.7995 | 0.5098 | 0.04 ms / 0.04 ms |
| **`Fuse_2`** | BM25 Only (Baseline) | 0.2711 [0.245, 0.299] | 0.3454 | 0.3344 [0.311, 0.358] | 0.6648 | 0.3975 | 126.01 ms / 210.86 ms |
| **`Fuse_3`** | Standard RRF ($k=60$, 50/50) | 0.4054 [0.376, 0.436] | 0.5130 | 0.3870 [0.366, 0.410] | 0.7950 | 0.4744 | 126.05 ms / 210.90 ms |
| **`Fuse_4`** | Weighted RRF (75% Dense / 25% BM25) | 0.4344 [0.405, 0.464] | 0.5436 | 0.4225 [0.402, 0.446] | 0.8199 | 0.5121 | 126.05 ms / 210.90 ms |
| **`Fuse_5`** | Weighted RRF (90% Dense / 10% BM25) | 0.4347 [0.405, 0.464] | 0.5425 | 0.4304 [0.409, 0.453] | 0.8063 | 0.5152 | 126.05 ms / 210.90 ms |
| **`Fuse_6`** | Weighted RRF (50% Dense / 50% BM25) | 0.4054 [0.376, 0.436] | 0.5130 | 0.3870 [0.366, 0.410] | 0.7950 | 0.4744 | 126.05 ms / 210.90 ms |
| **`Fuse_7`** | Normalized Score Fusion (75/25) | 0.3956 [0.366, 0.425] | 0.4994 | 0.4002 [0.378, 0.423] | 0.8029 | 0.4830 | 126.05 ms / 210.90 ms |
| **`Fuse_4_PC`** | Weighted RRF (75/25) + Parent-Child | **0.6676** [0.636, 0.697] | **0.7044** | 0.4225 [0.402, 0.446] | **0.8199** | 0.5121 | 126.05 ms / 210.90 ms |
| **`Fuse_7_PC`** | Normalized Score Fusion + Parent-Child | 0.6508 [0.620, 0.680] | 0.6886 | 0.4002 [0.378, 0.423] | 0.8029 | 0.4830 | 126.05 ms / 210.90 ms |

---

## 3. Why Dense Dominates BM25 in Financial Filings

In general-domain IR benchmarks (e.g., MS MARCO, TREC-COVID), BM25 frequently performs within 5–10% of dense models because exact keyword matches (names, entities, codes) are highly informative. In SEC 10-K financial documents, however, BM25 fails severely for three structural reasons:

1. **High Vocabulary Uniformity Across Filings**:
   Financial reports use identical standardized terminology across hundreds of companies and thousands of pages: *"operating expenses"*, *"depreciation and amortization"*, *"net income"*, *"interest expense"*, *"income tax provision"*. Because term frequency (TF) and inverse document frequency (IDF) for these phrases saturate across the entire corpus, BM25 assigns low discriminative weight to the core financial accounting concepts.
2. **Tabular Syntactic Sparsity**:
   FinQA table rows are short, highly condensed text strings (e.g., `"table_1: cost of goods sold 2009: 1425 ; 2008: 1210"`). Queries phrased in natural language (e.g., *"What was the percentage increase in the cost of sales during 2009?"*) contain multiple query tokens (*"percentage"*, *"increase"*, *"during"*, *"sales"*) that do not appear verbatim in the table row. Dense bi-encoders map *"increase in cost of sales"* to the vector neighborhood of *"cost of goods sold"*, whereas BM25 assigns a near-zero score due to missing tokens.
3. **Spurious Lexical Matches on Numerical Tokens**:
   BM25 matches isolated numbers (e.g., `"2008"`, `"10"`, `"1"`) indiscriminately across footnotes and boilerplate risk factors, creating high-ranking false positives that drown out the true financial schedule.

---

## 4. Statistical Significance of Fusion Contrasts

All pairwise tests were conducted via two-sided paired Wilcoxon signed-rank tests ($N=883$) with Holm-Bonferroni correction across the experimental family.

```
Contrast 1: Weighted RRF (75/25) vs Standard RRF (50/50)
  - Span Recall@10 : +0.0290 (W = 13,644.0, p = 8.04e-07, Holm sig: YES, r = 0.8057)
  - Page MRR       : +0.0355 (W = 134,521.5, p = 5.38e-08, Holm sig: YES, r = 0.2691)

Contrast 2: Weighted RRF (75/25) vs Dense Only
  - Span Recall@10 : +0.0008 (W = 26,436.5, p = 0.7220, Holm sig: NO, r = 0.7489)
  - Page MRR       : -0.0054 (W = 100,988.0, p = 0.5236, Holm sig: NO, r = 0.4180)

Contrast 3: Weighted RRF (90/10) vs Dense Only
  - Span Recall@10 : +0.0011 (W = 6,997.0, p = 0.8071, Holm sig: NO, r = 0.8352)
  - Page MRR       : +0.0025 (W = 60,816.5, p = 0.3069, Holm sig: NO, r = 0.5963)

Contrast 4: Normalized Score Fusion (75/25) vs Weighted RRF (75/25)
  - Span Recall@10 : -0.0389 (W = 10,122.0, p = 1.51e-09, Holm sig: YES, r = 0.8213)
  - Page MRR       : -0.0223 (W = 93,346.5, p = 1.10e-04, Holm sig: YES, r = 0.4519)
```

### Interpretation:
- Standard 50/50 RRF significantly hurts performance ($p = 8.04 \times 10^{-7}$).
- Weighted RRF with 75% or 90% Dense weighting fully eliminates the penalty and reaches parity with Dense Only, while providing slight complementary coverage for rare queries with distinctive entity names (e.g., ticker symbols or specific accounting footnote numbers).
- Rank-based reciprocal rank fusion (RRF) is vastly superior to score-based linear normalization ($p = 1.51 \times 10^{-9}$), because BM25 scores have no fixed upper bound and vary wildly between short table rows and long narrative paragraphs.

---

## 5. Architectural Verdict

1. **Never use unweighted 50/50 RRF** when pairing Dense bi-encoders with BM25 on financial 10-K corpora.
2. If hybrid fusion is used, **Weighted RRF ($75/25$ or $90/10$)** is the only statistically sound formulation.
3. However, from an operational efficiency perspective:
   - Dense Only executes in **0.04 ms/query** on the RTX 4060 GPU.
   - Any RRF configuration requires executing BM25 over the entire corpus, which consumes **~126 ms/query** (a $3,150 \times$ latency increase).
   - Because Weighted RRF ($75/25$) provides statistically indistinguishable Span Recall@10 over Dense Only ($+0.08$ pp, $p = 0.7220$), the computational cost of BM25 is not justified for latency-critical production paths unless specifically targeting queries with rare corporate identifiers.
