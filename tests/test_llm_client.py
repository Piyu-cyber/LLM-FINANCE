"""test_llm_client.py — Unit and integration tests for the LLM generation interface.

Tests:
1. GenerationConfig validation and parameter boundaries.
2. MockLLMClient deterministic generation, token counting, latency, failure, and timeout simulation.
3. HuggingFaceLocalClient dependency validation, CUDA checking, missing model handling, and clean unloading.
4. ConfigurableAPIClient credential checking, mocked request execution, usage parsing, and error handling.
5. Conditional local model inference (skipped unless explicitly requested and cached).
"""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure test paths are available
for p in ["src", "src/generation"]:
    if p not in sys.path:
        sys.path.insert(0, p)

from generation_config import GenerationConfig
from llm_client import (
    GenerationResult,
    LLMClient,
    MockLLMClient,
    HuggingFaceLocalClient,
    ConfigurableAPIClient,
)


def _has_working_torch():
    try:
        import torch
        from transformers import AutoTokenizer
        return True
    except Exception:
        return False


# ===========================================================================
# 1. GenerationConfig Tests
# ===========================================================================

class TestGenerationConfig:
    """Test configuration validation, bounds checking, and API key management."""

    def test_default_mock_config(self):
        cfg = GenerationConfig()
        assert cfg.backend == "mock"
        assert cfg.temperature == 0.0
        assert cfg.max_output_tokens == 512
        assert cfg.max_repair_attempts == 1
        assert cfg.to_dict()["backend"] == "mock"

    def test_valid_custom_config(self):
        cfg = GenerationConfig(
            backend="local",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            device="cuda",
            dtype="bfloat16",
            temperature=0.1,
            max_output_tokens=256,
            generation_timeout=30.0,
        )
        assert cfg.backend == "local"
        assert cfg.device == "cuda"
        assert cfg.dtype == "bfloat16"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid backend 'tpu'"):
            GenerationConfig(backend="tpu")

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="Invalid device 'metal'"):
            GenerationConfig(device="metal")

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="Invalid dtype 'int8'"):
            GenerationConfig(dtype="int8")

    def test_invalid_quantization_raises(self):
        with pytest.raises(ValueError, match="Invalid quantization '1bit'"):
            GenerationConfig(quantization="1bit")

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            GenerationConfig(temperature=-0.5)

    def test_non_positive_max_output_tokens_raises(self):
        with pytest.raises(ValueError, match="max_output_tokens must be positive"):
            GenerationConfig(max_output_tokens=0)

    def test_non_positive_max_input_tokens_raises(self):
        with pytest.raises(ValueError, match="max_input_tokens must be positive"):
            GenerationConfig(max_input_tokens=-10)

    def test_non_positive_timeout_raises(self):
        with pytest.raises(ValueError, match="generation_timeout must be positive"):
            GenerationConfig(generation_timeout=0.0)

    def test_quantized_cpu_raises(self):
        with pytest.raises(ValueError, match="Quantized execution .* is not supported on CPU"):
            GenerationConfig(backend="local", device="cpu", quantization="4bit")

    def test_api_backend_missing_provider_or_endpoint_raises(self):
        with pytest.raises(ValueError, match="either 'api_provider' or 'api_endpoint' must be specified"):
            GenerationConfig(backend="api")

    def test_get_api_key_reads_environment(self, monkeypatch):
        cfg = GenerationConfig(
            backend="api",
            api_provider="openai",
            api_key_env_var="TEST_LLM_KEY",
        )
        # Not set
        monkeypatch.delenv("TEST_LLM_KEY", raising=False)
        assert cfg.get_api_key() is None
        assert cfg.to_dict()["api_key_configured"] is False

        # Set
        monkeypatch.setenv("TEST_LLM_KEY", "sk-secret-12345")
        assert cfg.get_api_key() == "sk-secret-12345"
        assert cfg.to_dict()["api_key_configured"] is True
        # Ensure raw secret is not dumped in to_dict()
        assert "sk-secret-12345" not in str(cfg.to_dict())


# ===========================================================================
# 2. MockLLMClient Tests
# ===========================================================================

class TestMockLLMClient:
    """Test deterministic offline mock generation client."""

    def test_mock_lifecycle(self):
        client = MockLLMClient()
        assert not client.is_loaded
        client.load()
        assert client.is_loaded
        client.unload()
        assert not client.is_loaded

    def test_context_manager(self):
        client = MockLLMClient()
        with client as c:
            assert c.is_loaded
        assert not client.is_loaded

    def test_deterministic_generation(self):
        client = MockLLMClient(default_response='{"status": "answered", "value": 100.0}')
        res = client.generate("Calculate percentage increase:")
        assert res.is_success
        assert res.status == "SUCCESS"
        assert res.text == '{"status": "answered", "value": 100.0}'
        assert res.prompt_tokens > 0
        assert res.completion_tokens > 0
        assert res.total_tokens == res.prompt_tokens + res.completion_tokens
        assert res.latency_ms > 0.0
        assert res.backend == "mock"

    def test_deterministic_token_counting(self):
        client = MockLLMClient()
        # 'Interest: -600 million USD.' has: Interest, :, -, 600, million, USD, . -> 7 tokens
        count = client.count_tokens("Interest: -600 million USD.")
        assert count == 7
        assert client.count_tokens("") == 0

    def test_response_map(self):
        response_map = {
            "interest": '{"claim": "interest expense", "value": -600}',
            "tax": '{"claim": "income tax", "value": 250}',
        }
        client = MockLLMClient(response_map=response_map)
        res_interest = client.generate("What is the interest expense?")
        assert res_interest.text == '{"claim": "interest expense", "value": -600}'

        res_tax = client.generate("What is the tax provision?")
        assert res_tax.text == '{"claim": "income tax", "value": 250}'

    def test_response_queue(self):
        queue = [
            '{"step": 1, "value": 10}',
            '{"step": 2, "value": 20}',
        ]
        client = MockLLMClient(response_queue=queue)
        r1 = client.generate("first")
        assert r1.text == '{"step": 1, "value": 10}'
        r2 = client.generate("second")
        assert r2.text == '{"step": 2, "value": 20}'
        # Falls back to default when queue is exhausted
        r3 = client.generate("third")
        assert r3.text == client.default_response

    def test_simulated_failure(self):
        client = MockLLMClient(simulate_failure=True)
        res = client.generate("test prompt")
        assert not res.is_success
        assert res.status == "FAILURE"
        assert "Simulated mock generation failure" in (res.error_message or "")
        assert res.text == ""

    def test_simulated_timeout(self):
        client = MockLLMClient(simulate_timeout=True)
        res = client.generate("test prompt")
        assert not res.is_success
        assert res.status == "TIMEOUT"
        assert "timed out" in (res.error_message or "")

    def test_get_model_info(self):
        client = MockLLMClient()
        info = client.get_model_info()
        assert info["backend"] == "mock"
        assert info["model_name"] == "mock-financial-llm"


# ===========================================================================
# 3. HuggingFaceLocalClient Tests (Unit & Dependency Validation)
# ===========================================================================

class TestHuggingFaceLocalClientUnit:
    """Test local Hugging Face client validation and error paths without downloading."""

    def test_quantization_missing_bitsandbytes_raises(self):
        cfg = GenerationConfig(
            backend="local",
            model_name="mock-model",
            device="cuda",
            quantization="4bit",
        )
        client = HuggingFaceLocalClient(cfg)

        with patch.dict("sys.modules", {"bitsandbytes": None}):
            # Attempting load should detect missing bitsandbytes
            with pytest.raises(ImportError, match="requires 'bitsandbytes'"):
                client.load()

    def test_cuda_device_resolution_when_unavailable(self):
        cfg = GenerationConfig(
            backend="local",
            model_name="mock-model",
            device="cuda:99",
        )
        client = HuggingFaceLocalClient(cfg)
        with pytest.raises(RuntimeError):
            client.load()

    def test_missing_model_raises_runtime_error(self):
        cfg = GenerationConfig(
            backend="local",
            model_name="nonexistent/fake-model-repo-id-404",
            device="cpu",
        )
        client = HuggingFaceLocalClient(cfg)
        mock_tf = sys.modules.get("transformers")
        if mock_tf is not None and hasattr(mock_tf, "AutoTokenizer"):
            with patch.object(mock_tf.AutoTokenizer, "from_pretrained", side_effect=OSError("Model not found")):
                with pytest.raises(RuntimeError, match="Failed to load Hugging Face model"):
                    client.load()
        else:
            with pytest.raises(RuntimeError):
                client.load()

    def test_token_counting_fallback_when_unloaded(self):
        cfg = GenerationConfig(backend="local", model_name="mock-model")
        client = HuggingFaceLocalClient(cfg)
        # When unloaded, count_tokens uses safe fallback
        assert client.count_tokens("USD 500 million") == 3

    def test_unload_lifecycle(self):
        cfg = GenerationConfig(backend="local", model_name="mock-model", device="cpu")
        client = HuggingFaceLocalClient(cfg)
        client.model = MagicMock()
        client.tokenizer = MagicMock()
        client._is_loaded = True

        client.unload()
        assert client.model is None
        assert client.tokenizer is None
        assert not client.is_loaded


# ===========================================================================
# 4. ConfigurableAPIClient Tests
# ===========================================================================

class TestConfigurableAPIClient:
    """Test API client credential management, request dispatch, and error handling."""

    def test_missing_api_key_raises(self, monkeypatch):
        cfg = GenerationConfig(
            backend="api",
            api_provider="openai",
            api_key_env_var="NONEXISTENT_KEY_ENV",
        )
        monkeypatch.delenv("NONEXISTENT_KEY_ENV", raising=False)
        client = ConfigurableAPIClient(cfg)
        with pytest.raises(ValueError, match="Missing API key"):
            client.load()

    def test_mocked_openai_api_call_success(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-mock-key")
        cfg = GenerationConfig(
            backend="api",
            api_provider="openai",
            api_key_env_var="TEST_API_KEY",
            model_name="gpt-4o-mini",
        )
        client = ConfigurableAPIClient(cfg)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"status": "answered", "value": 42}'}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }

        with patch("requests.post", return_value=mock_response):
            res = client.generate("Calculate ratio:")
            assert res.is_success
            assert res.status == "SUCCESS"
            assert res.text == '{"status": "answered", "value": 42}'
            assert res.prompt_tokens == 15
            assert res.completion_tokens == 10
            assert res.total_tokens == 25
            assert res.backend == "api"

    def test_mocked_anthropic_api_call_success(self, monkeypatch):
        monkeypatch.setenv("TEST_ANTHROPIC_KEY", "ant-mock-key")
        cfg = GenerationConfig(
            backend="api",
            api_provider="anthropic",
            api_key_env_var="TEST_ANTHROPIC_KEY",
            model_name="claude-3-haiku-20240307",
        )
        client = ConfigurableAPIClient(cfg)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": '{"result": 123}'}],
            "usage": {"input_tokens": 20, "output_tokens": 8},
        }

        with patch("requests.post", return_value=mock_response):
            res = client.generate("Calculate:")
            assert res.is_success
            assert res.text == '{"result": 123}'
            assert res.prompt_tokens == 20
            assert res.completion_tokens == 8

    def test_api_http_error_handling(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-mock-key")
        cfg = GenerationConfig(
            backend="api",
            api_provider="openai",
            api_key_env_var="TEST_API_KEY",
        )
        client = ConfigurableAPIClient(cfg)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized: Invalid API key"

        with patch("requests.post", return_value=mock_response):
            res = client.generate("test")
            assert not res.is_success
            assert res.status == "FAILURE"
            assert "HTTP 401" in (res.error_message or "")

    def test_api_timeout_handling(self, monkeypatch):
        import requests
        monkeypatch.setenv("TEST_API_KEY", "sk-mock-key")
        cfg = GenerationConfig(
            backend="api",
            api_provider="openai",
            api_key_env_var="TEST_API_KEY",
            generation_timeout=5.0,
        )
        client = ConfigurableAPIClient(cfg)

        with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
            res = client.generate("test")
            assert not res.is_success
            assert res.status == "TIMEOUT"
            assert "timed out after 5.0s" in (res.error_message or "")


# ===========================================================================
# 5. Conditional Real Local Model Tests (Skipped Offline)
# ===========================================================================

try:
    import torch
    _cuda_avail = torch.cuda.is_available()
except Exception:
    _cuda_avail = False

_qwen_snap_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
LOCAL_MODEL_CACHED = (
    os.path.exists(os.path.join(_qwen_snap_dir, "model-00001-of-00002.safetensors"))
    and os.path.exists(os.path.join(_qwen_snap_dir, "model-00002-of-00002.safetensors"))
    and _cuda_avail
)

@pytest.mark.skipif(not LOCAL_MODEL_CACHED, reason="Qwen2.5-3B-Instruct model is not cached locally")
def test_real_qwen_local_generation():
    """Optional end-to-end integration test executing real forward pass on cached model."""
    cfg = GenerationConfig(
        backend="local",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        device="cuda",
        dtype="bfloat16",
        max_output_tokens=64,
        temperature=0.0,
    )
    client = HuggingFaceLocalClient(cfg)
    with client:
        res = client.generate("What is 100 divided by 5? Output JSON.")
        assert res.is_success
        assert len(res.text) > 0
        assert res.latency_ms > 0.0
        assert "vram_allocated_mb" in res.metadata
