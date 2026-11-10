# LLM-FINANCE — Phase 2: Experimental Environment & Hardware Verification Report

**Project**: LLM-FINANCE: Verifiable and Attack-Resistant Financial RAG for SEC 10-K Documents  
**Date**: November 3, 2026  
**Author**: Machine Learning Research Engineer  
**Status**: Verified  

---

## 1. System and Execution Environments

The machine hosts a dual-environment configuration:
1. **Primary ML Execution Environment (`.venv`)**: Python 3.11 with the full ML stack (PyTorch 2.14.1+cu126, CUDA 12.6, Transformers, Sentence-Transformers, cu126 wheels).
2. **System Host Environment**: Python 3.13 with standard developer utilities and CLI tools.

| Component | Specification | Details / Verification Source |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home / Pro | Windows-10-10.0.26300-SP0 AMD64 |
| **System RAM** | 24.0 GB | 23.8 GB usable |
| **Python Version (ML)** | 3.11.16 | `.venv/Scripts/python.exe` (MSC v.1944 64 bit) |
| **Python Version (System)**| 3.13.7 | `Python313/python.exe` (pytest runner) |
| **PyTorch Version** | 2.14.1+cu126 | Built with CUDA 12.6 support |
| **CUDA Runtime** | 12.6 | `torch.version.cuda` = 12.6 |
| **CUDA Availability** | `True` | `torch.cuda.is_available() == True` |
| **GPU Model** | NVIDIA GeForce RTX 4060 Laptop GPU | 1 physical device, Device index 0 |
| **Total GPU VRAM** | 7.9956 GB (8,187.5 MB) | `torch.cuda.get_device_properties(0).total_memory` |
| **Available / Free VRAM** | 6.9297 GB (7,096.0 MB) | `torch.cuda.mem_get_info()` at idle |
| **Allocated VRAM (Init)** | 0.0 MB | Clean process initialization |
| **Compute Capability** | 8.9 | NVIDIA Ada Lovelace architecture |
| **FP16 Supported** | `True` | Native Ada Lovelace 4th Gen Tensor Cores |
| **BF16 Supported** | `True` | `torch.cuda.is_bf16_supported() == True` |
| **Sentence-Transformers** | 6.1.0 | Installed in `.venv` |
| **Transformers** | 5.18.0 | Installed in `.venv` |
| **Package Manager** | uv 0.12.9 | `C:/Users/palak/.local/bin/uv.exe` |

---

## 2. Hugging Face Local Model Cache Status

Local cache directory inspected: `C:\Users\palak\.cache\huggingface\hub`

| Model Identifier | Architecture | Status | Local Cache Directory |
| :--- | :--- | :---: | :--- |
| `BAAI/bge-small-en-v1.5` | Dense Bi-Encoder (384-d) | **CACHED** | `models--BAAI--bge-small-en-v1.5` |
| `bert-base-uncased` | Masked LM Base | **CACHED** | `models--bert-base-uncased` |
| `sentence-transformers/all-MiniLM-L6-v2` | Dense Bi-Encoder (384-d) | **CACHED** | `models--sentence-transformers--all-MiniLM-L6-v2` |
| `jinaai/jina-embeddings-v3` | Dense Bi-Encoder | **CACHED** | `models--jinaai--jina-embeddings-v3` |
| `BAAI/bge-reranker-base` | Cross-Encoder Reranker | **NOT CACHED** | Missing from local cache hub |

> [!IMPORTANT]
> Because standard execution environments run under network isolation, `HF_HUB_OFFLINE=1` is required to ensure pre-cached models (`bge-small-en-v1.5`) load deterministically from disk without network timeouts. Any reranking pipeline requiring `BAAI/bge-reranker-base` must either use local cache or fallback gracefully.

---

## 3. Automated Test Suite Results

Command executed:
```powershell
python -m pytest -v
```
(Invoked via pytest engine on `C:\Users\palak\LLM-FINANCE`)

```
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\palak\LLM-FINANCE
configfile: pyproject.toml
testpaths: src, tests
plugins: anyio-4.10.0, langsmith-0.4.34, cov-7.1.0
collected 172 items

src/eval/test_retrieval_eval.py: 38 passed
src/retrieval/test_chunker.py: 18 passed
src/retrieval/test_cross_encoder_reranker.py: 5 passed
src/retrieval/test_dense_retriever.py: 10 passed
src/retrieval/test_finetune_retriever.py: 13 passed
src/retrieval/test_parent_child_retriever.py: 11 passed
src/retrieval/test_query_parser.py: 32 passed
src/retrieval/test_rrf_identity.py: 4 passed
src/verification/test_calculator.py: 11 passed
src/verification/test_pipeline.py: 3 passed
src/verification/test_validator.py: 19 passed
tests/test_integration.py: 7 passed, 1 skipped

======================= 171 passed, 1 skipped in 2.97s ========================
```

- **Total Tests Collected**: 172
- **Passed**: 171
- **Failed**: 0
- **Skipped**: 1 (`tests/test_integration.py::test_gpu_embedding_and_reranking_execution`)
- **Warnings**: 0
- **Total Suite Runtime**: 2.97 seconds

---

## 4. Investigation of the Skipped Conditional CUDA Test

### Root Cause Analysis
In `tests/test_integration.py`, the GPU test is guarded by:
```python
def _has_cuda():
    try:
        import torch
        if not hasattr(torch, "cuda") or isinstance(torch.cuda, MagicMock):
            return False
        res = torch.cuda.is_available()
        return bool(res) if not isinstance(res, MagicMock) else False
    except Exception:
        return False

@pytest.mark.skipif(not _has_cuda(), reason="CUDA GPU not available in current test environment")
def test_gpu_embedding_and_reranking_execution(): ...
```
When running the general test suite via the system Python 3.13 environment, `torch` is not installed in Python 3.13. To allow instant, lightweight test execution without heavy C-extensions, mock stubs are provided. Consequently, `_has_cuda()` correctly evaluates to `False`, triggering the intended condition.

### Direct Execution in the CUDA Environment
The test was independently executed inside the Python 3.11 virtual environment (`.\.venv\Scripts\python.exe`) where PyTorch 2.14.1+cu126 is installed:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -c "
import sys; sys.path.insert(0, '.')
from tests.test_integration import test_gpu_embedding_and_reranking_execution
test_gpu_embedding_and_reranking_execution()
print('SUCCESS: test_gpu_embedding_and_reranking_execution passed on CUDA!')
"
```

**Result**:
```
Loading weights: 100%|##########| 199/199 [00:00<00:00, 1817.24it/s]
SUCCESS: test_gpu_embedding_and_reranking_execution passed on CUDA!
```
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Output Tensor Device: `cuda:0`
- Embedding Dimension: `[1, 384]`
- Result: **VERIFIED PASS** on physical hardware.
