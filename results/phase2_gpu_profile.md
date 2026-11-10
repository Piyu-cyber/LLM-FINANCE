# Phase 2 — RTX 4060 GPU Profiling Report

**Hardware Target**: NVIDIA GeForce RTX 4060 Laptop GPU (8,187.5 MB Total VRAM)  
**Host Architecture**: Windows 11, Compute Capability 8.9 (Ada Lovelace), 24 GB Host System RAM  
**PyTorch / CUDA**: PyTorch 2.14.1+cu126, CUDA Runtime 12.6, cuDNN Enabled  
**Embedding Model**: `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors)  
**Execution Environment**: Sandboxed Python Virtualenv (`.venv`, Python 3.11)  
**Date**: November 2026

---

## 1. Executive Summary

This report documents comprehensive GPU profiling and performance characterization for the **LLM-FINANCE** dense retrieval infrastructure executed on a local **NVIDIA GeForce RTX 4060 Laptop GPU**. All measurements were captured under controlled, reproducible conditions using the official FinQA validation dataset (25,007 chunks; 883 queries).

### Key Architectural Findings:
1. **Exceptional Throughput**: The RTX 4060 achieves **1,529.0 chunks/sec** at batch size 64 (scaling to **1,604.1 chunks/sec** at batch size 128), embedding the entire 25,007 validation chunk corpus in just **16.53 seconds**.
2. **Minimal VRAM Footprint**: Full corpus encoding at batch size 64 requires only **260.0 MB** peak allocated VRAM and **356.0 MB** reserved VRAM (< 4.4% of total 8 GB VRAM capacity), leaving over 7.5 GB of VRAM available for future LLM generation or cross-encoder reranking.
3. **GPU Acceleration Speedup**: GPU encoding provides a **5.88x speedup** over CPU execution on the same host.
4. **FP32 vs FP16 Fidelity**: FP16 inference produces **100% top-10 rank agreement** and an average cosine similarity of **1.000000** with FP32 embeddings, with maximum absolute deviation of only $1.74 \times 10^{-4}$.
5. **Interactive Single-Query Latency**: Single-query dense encoding latency averages **19.89 ms** (median **19.72 ms**, 95th percentile **26.08 ms**), supporting interactive sub-50ms financial RAG workflows.

---

## 2. Batch Size Scaling & Memory Profiling

Embedding generation throughput and CUDA memory utilization were systematically profiled across standard transformer batch sizes using PyTorch CUDA memory tracking (`torch.cuda.max_memory_allocated` and `torch.cuda.max_memory_reserved`).

| Batch Size | Duration (1,024 chunks) | Throughput (chunks/sec) | Peak Allocated VRAM (MB) | Peak Reserved VRAM (MB) | Full Corpus Projection (25,007 chunks) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **32** | 0.8407 s | 1,218.01 | 197.70 MB | 260.00 MB | 20.53 s |
| **64** *(Optimal)* | 0.6697 s | 1,529.00 | 259.98 MB | 356.00 MB | 16.36 s |
| **128** | 0.6384 s | 1,604.13 | 380.99 MB | 572.00 MB | 15.59 s |

### Throughput Analysis:
- Moving from batch size 32 to 64 yields a **+25.5% throughput gain** (1,218.0 $\to$ 1,529.0 chunks/sec) due to improved Tensor Core occupancy on the Ada Lovelace SMs.
- Increasing from batch size 64 to 128 yields a modest **+4.9% gain** (1,529.0 $\to$ 1,604.1 chunks/sec) at the expense of a **+60.7% increase in reserved VRAM** (356 MB $\to$ 572 MB).
- **Recommendation**: Batch size **64** represents the sweet spot for throughput versus memory safety, preventing VRAM fragmentation when collocated with dense indices.

---

## 3. Numerical Precision: FP32 vs. FP16 Evaluation

To assess whether half-precision (FP16) inference can be safely deployed without degrading retrieval accuracy, we compared FP32 and FP16 embeddings across 1,024 validation chunks.

| Metric | FP32 (Single Precision) | FP16 (Half Precision) | Delta / Impact |
|:---|:---:|:---:|:---|
| **Peak Allocated VRAM** | 259.98 MB | 223.07 MB | -36.91 MB (-14.2%) |
| **Duration (1,024 chunks)** | 0.6571 s | 0.8425 s | Latency overhead from type conversions |
| **Mean Cosine Similarity** | 1.000000 | 1.000000 | Identical angular direction |
| **Minimum Cosine Similarity** | 1.000000 | 1.000000 | Perfect preservation |
| **Max Absolute Element Error** | — | $1.74 \times 10^{-4}$ | Within FP16 mantissa resolution |
| **Top-10 Retrieval Rank Agreement** | **100.0%** (1.0000) | **100.0%** (1.0000) | Zero rank order distortions |

> [!NOTE]
> On the RTX 4060 Laptop GPU, pure PyTorch FP16 inference for small transformer models (`bge-small-en-v1.5`, 33M parameters) exhibits a slight kernel launch / down-casting overhead when not compiled via TensorRT or TorchScript, making FP32 marginally faster in raw wall-clock time for batch encoding. However, **100% top-10 rank agreement** proves FP16 can be deployed whenever VRAM conservation is paramount.

---

## 4. Single-Query Latency Distribution

Online financial question answering requires low-latency retrieval. We profiled the end-to-end latency of encoding individual query strings across 100 randomly sampled FinQA validation questions.

| Metric | Latency (ms) | Target SLA | Compliance |
|:---|:---:|:---:|:---:|
| **Minimum Latency** | 14.52 ms | < 50 ms | Passed |
| **Average Latency** | 19.89 ms | < 50 ms | Passed |
| **Median (P50) Latency** | 19.72 ms | < 50 ms | Passed |
| **95th Percentile (P95) Latency** | 26.08 ms | < 50 ms | Passed |
| **99th Percentile (P99) Latency** | 31.01 ms | < 75 ms | Passed |
| **Maximum Latency** | 33.32 ms | < 100 ms | Passed |

All single-query latencies are under 35 ms, leaving ample budget (> 450 ms) for upstream token parsing and downstream generative verification within standard sub-second SLAs.

---

## 5. CPU vs. GPU Acceleration Speedup

We benchmarked CPU execution (Intel 13th Gen multi-core host processor with PyTorch AVX2/AVX-512 vectorization) against the NVIDIA RTX 4060 Laptop GPU on identical chunk batches.

| Processor Device | Execution Time (256 Chunks) | Throughput (chunks/sec) | Measured Speedup |
|:---|:---:|:---:|:---:|
| **Host CPU** (Torch CPU) | 2.3512 s | 108.9 c/s | Baseline (1.00x) |
| **RTX 4060 GPU** (CUDA 12.6) | 0.3998 s | 640.4 c/s | **5.88x Faster** |
| **RTX 4060 Full Pipeline** (BS=64) | — | 1,529.0 c/s | **14.04x Faster** |

### Computational Implications:
- On CPU, generating embeddings for the 25,007 FinQA validation chunks would require approximately **3 minutes 50 seconds**.
- On RTX 4060 GPU, the entire validation corpus is embedded in **16.53 seconds** — an operational efficiency gain of over 92%.

---

## 6. Full Corpus Verification & Data Artifacts

- **Output File**: `data/finqa/dev_embeddings.npy`
- **Matrix Dimension**: `(25007, 384)`
- **Data Type**: `float32` (38,410,752 floating-point values)
- **File Size on Disk**: 36.63 MB
- **Unit Normalization Check**: Verified $\ell_2$ norm $= 1.00000 \pm 10^{-6}$ across all 25,007 rows.
- **Data Integrity**: 0 NaN, 0 Inf, 0 null vectors.

---

## 7. Conclusions & GPU Deployment Guidelines for Phase 3

1. **Retain Batch Size 64**: Use batch size 64 for all offline corpus indexing and evaluation passes.
2. **GPU Batched Retrieval**: Retain PyTorch matrix multiplication (`torch.matmul`) and `torch.topk` on CUDA for dense scoring, which retrieves top-100 candidates for all 883 validation queries in just **179.67 ms**.
3. **VRAM Headroom**: With only 356 MB peak reserved VRAM during dense retrieval, the remaining **7.8 GB VRAM** is fully available for quantized LLM generation (e.g. 4-bit / 8-bit quantized models such as Mistral-7B or Qwen-2.5-7B) in Phase 3.
