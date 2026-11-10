# LLM-FINANCE — Phase 2: FinQA Validation Dataset Audit Report

**Dataset Path**: `data/finqa/dataset/dev.json`  
**Evaluation Split**: Official Validation Split (`dev`)  
**Date**: November 3, 2026  
**Auditor**: Machine Learning Research Engineer  
**Status**: Verified Clean and Ready for Benchmarking  

---

## 1. Dataset Overview and Structural Integrity

| Property | Value | Notes / Verification Method |
| :--- | :--- | :--- |
| **File Existence** | Verified | File located at `data/finqa/dataset/dev.json` |
| **Total Examples** | 883 | Exact count from JSON parsing |
| **Unique Example IDs** | 883 | 100% unique identifiers; 0 duplicates |
| **Unique Companies / Filings** | 224 | Distinct `Company/Year` 10-K report boundaries |
| **Unique Questions** | 874 | 9 duplicate question strings across different companies |
| **Train / Dev ID Leakage** | 0 | Strict disjoint intersection with `train.json` (6,251 IDs) |
| **Eligible Evaluation Examples**| 883 | All 883 examples have parseable gold evidence |
| **Excluded Examples** | 0 | No examples discarded or omitted |

---

## 2. Evidence Annotation Breakdown

FinQA questions require evidence extracted from financial tables (`table_*`) and narrative text passages (`pre_text_*`, `post_text_*`).

- **Total Gold Evidence Items**: 1,513 annotations across 883 examples
- **Evidence Source Distribution**:
  - Financial Table Rows: **1,071** (70.8%)
  - Narrative Text Passages: **442** (29.2%)
- **Mean Evidence Spans per Query**: **1.71 spans**

### Gold Span Count Distribution per Example
| Number of Spans Required | Frequency (Examples) | Percentage of Dev Split |
| :---: | :---: | :---: |
| 1 span | 416 | 47.1% |
| 2 spans | 356 | 40.3% |
| 3 spans | 80 | 9.1% |
| 4 spans | 22 | 2.5% |
| 5 spans | 2 | 0.2% |
| 6 spans | 3 | 0.3% |
| 7 spans | 3 | 0.3% |
| 8 spans | 1 | 0.1% |

> [!IMPORTANT]
> **52.9% of all validation queries require two or more evidence spans** to answer the financial question. This underscores the necessity of the Phase 1 metric correction: measuring true multi-span Recall@K ($\text{covered} / \text{total}$) rather than Hit@K (which triggers on a single match).

---

## 3. Duplicate Question Analysis

The 9 repeated question strings occur across distinct SEC 10-K filings and represent standardized accounting questions:
1. *"what percentage of the total oil and gas mmboe comes from canada?"* (2 filings)
2. *"what is the percent change in annual long-term debt maturities from 2016 to 2017?"* (2 filings)
3. *"by how much did the average price per share increase from 2010 to 2011?"* (2 filings)
4. *"what was average net sales for space systems in millions from 2013 to 2015?"* (2 filings)
5. *"what percent of assets acquired by the acquisition are non-tangible assets?"* (2 filings)
6. *"what was the percentage change in net sales from 2008 to 2009?"* (2 filings)
7. *"what was the percentage change in operating income from 2015 to 2016?"* (2 filings)
8. *"what is the ratio of operating expenses to total revenues in 2017?"* (2 filings)
9. *"what is the effective tax rate in 2012?"* (2 filings)

Because each instance refers to a different company report (`id`), all 883 examples represent distinct queries and are retained for evaluation.

---

## 4. Query ID Alignment Guarantee

To ensure scientific validity across all Phase 2 benchmark configurations (BM25, Dense, RRF, Cross-Encoder, Metadata Boosting, Parent-Child Expansion):
- Every retrieval method will be evaluated against the exact identical list of **883 query IDs**.
- All evaluation results will serialize the full array of evaluated query IDs in the output JSON.
