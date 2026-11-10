"""llm_client.py — Common LLM generation interface and backend implementations.

Provides:
- GenerationResult: Standardized structured result across all backends.
- LLMClient: Abstract base class for all generation clients.
- MockLLMClient: Deterministic, offline mock client for automated testing and CI.
- HuggingFaceLocalClient: Local PyTorch/Transformers inference engine on GPU/CPU.
- ConfigurableAPIClient: Provider-agnostic REST API client for hosted LLM endpoints.
"""

from __future__ import annotations

import abc
import gc
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock

try:
    from .generation_config import GenerationConfig
except (ImportError, ValueError):
    from generation_config import GenerationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standardized Generation Result
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Standardized output of a generation call, independent of backend or provider.

    Attributes:
        text: Generated completion string (empty on failure).
        model_name: Name or path of the model used.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated in the completion.
        total_tokens: Total tokens consumed (prompt + completion).
        latency_ms: Wall-clock generation time in milliseconds.
        backend: Execution backend identifier ('mock', 'local', 'api').
        status: Execution status ('SUCCESS', 'FAILURE', 'TIMEOUT').
        error_message: Detailed diagnostic error message if status is not 'SUCCESS'.
        metadata: Provider- or device-specific metadata (e.g. GPU memory, finish reason).
    """

    text: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    backend: str = "mock"
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Return True if generation succeeded."""
        return self.status == "SUCCESS"


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class LLMClient(abc.ABC):
    """Abstract interface defining the common contract for all LLM backends."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._is_loaded = False
        self._load_latency_ms = 0.0

    @abc.abstractmethod
    def load(self) -> None:
        """Initialize and load the model/client resources into memory."""
        pass

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Generate a completion for the given prompt string.

        Args:
            prompt: Text prompt passed to the model.
            **kwargs: Dynamic overrides for temperature, max_output_tokens, etc.

        Returns:
            GenerationResult: Standardized result object.
        """
        pass

    @abc.abstractmethod
    def unload(self) -> None:
        """Release any loaded model weights, memory, or network sessions."""
        pass

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the provided text for this model/backend."""
        pass

    @abc.abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return diagnostic metadata describing the model, backend, and hardware."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Return True if the client is currently loaded and ready for inference."""
        return self._is_loaded

    def __enter__(self) -> LLMClient:
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload()


# ---------------------------------------------------------------------------
# MockLLMClient (Offline & Deterministic)
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Deterministic offline mock client for automated testing and CI.

    Supports:
    - Fixed default responses
    - Pattern-based response mapping
    - Sequential response queues
    - Simulated failures and timeouts
    - Deterministic documented token counting: len(re.findall(r'\\w+|[^\\w\\s]', text))
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        default_response: str = '{"status": "answered", "value": 42.0}',
        response_map: Optional[dict[str, str]] = None,
        response_queue: Optional[list[str]] = None,
        simulate_failure: bool = False,
        simulate_timeout: bool = False,
        mock_latency_ms: float = 1.5,
    ) -> None:
        if config is None:
            config = GenerationConfig(backend="mock", model_name="mock-financial-llm")
        super().__init__(config)
        self.default_response = default_response
        self.response_map = response_map or {}
        self.response_queue = list(response_queue) if response_queue else []
        self.simulate_failure = simulate_failure
        self.simulate_timeout = simulate_timeout
        self.mock_latency_ms = mock_latency_ms
        self.call_history: list[dict[str, Any]] = []

    def load(self) -> None:
        """Simulate loading resources instantly without network access."""
        t0 = time.perf_counter()
        self._is_loaded = True
        self._load_latency_ms = (time.perf_counter() - t0) * 1000.0

    def unload(self) -> None:
        """Clear call history and reset state."""
        self._is_loaded = False
        self.call_history.clear()

    def count_tokens(self, text: str) -> int:
        """Deterministic approximation: regex word and punctuation token count.

        Formula: count of contiguous alphanumeric words plus distinct punctuation symbols.
        Example: 'Interest: -600 million USD.' -> 6 tokens.
        """
        if not text:
            return 0
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return len(tokens)

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Generate deterministic response or simulate failure/timeout."""
        t0 = time.perf_counter()
        prompt_tokens = self.count_tokens(prompt)

        # Record call
        self.call_history.append({"prompt": prompt, "kwargs": kwargs})

        # Simulated timeout
        if self.simulate_timeout:
            latency = self.mock_latency_ms
            return GenerationResult(
                text="",
                model_name=self.config.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
                latency_ms=latency,
                backend="mock",
                status="TIMEOUT",
                error_message=f"Generation timed out after {self.config.generation_timeout}s.",
            )

        # Simulated failure
        if self.simulate_failure:
            latency = self.mock_latency_ms
            return GenerationResult(
                text="",
                model_name=self.config.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
                latency_ms=latency,
                backend="mock",
                status="FAILURE",
                error_message="Simulated mock generation failure.",
            )

        # Determine response
        if self.response_queue:
            text = self.response_queue.pop(0)
        else:
            text = self.default_response
            for pattern, mapped_response in self.response_map.items():
                if pattern in prompt:
                    text = mapped_response
                    break

        completion_tokens = self.count_tokens(text)
        latency = self.mock_latency_ms + (time.perf_counter() - t0) * 1000.0

        return GenerationResult(
            text=text,
            model_name=self.config.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency,
            backend="mock",
            status="SUCCESS",
            metadata={"call_index": len(self.call_history)},
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "model_name": self.config.model_name,
            "is_loaded": self._is_loaded,
            "simulated_failure": self.simulate_failure,
            "simulated_timeout": self.simulate_timeout,
            "queued_responses": len(self.response_queue),
        }


# ---------------------------------------------------------------------------
# HuggingFaceLocalClient (Local GPU/CPU Inference)
# ---------------------------------------------------------------------------

class HuggingFaceLocalClient(LLMClient):
    """Local PyTorch / Transformers model inference engine.

    Enforces:
    - Lazy loading of model and tokenizer.
    - Explicit device assignment with CUDA verification.
    - Strict quantization validation (checks bitsandbytes presence before attempting 4-bit/8-bit).
    - Separation of model-load latency from token-generation latency.
    - Explicit GPU VRAM tracking (allocated and reserved memory).
    - Clean unloading via garbage collection and torch.cuda.empty_cache().
    """

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        self.model: Any = None
        self.tokenizer: Any = None
        self._target_device: str = "cpu"
        self._torch_dtype: Any = None

    def _resolve_device_and_dtype(self) -> tuple[str, Any]:
        """Verify device availability and resolve PyTorch dtype."""
        try:
            import torch
        except Exception as e:
            raise RuntimeError(f"PyTorch could not be loaded: {e}") from e

        # Device resolution
        device_str = self.config.device
        cuda_available = False
        if hasattr(torch, "cuda") and not isinstance(torch.cuda, MagicMock):
            try:
                res = torch.cuda.is_available()
                cuda_available = bool(res) if not isinstance(res, MagicMock) else False
            except Exception:
                cuda_available = False

        if device_str in ("cuda", "auto") or device_str.startswith("cuda:"):
            if not cuda_available:
                if device_str == "auto":
                    logger.warning("CUDA unavailable; falling back to CPU.")
                    target_device = "cpu"
                else:
                    raise RuntimeError(
                        f"Device '{device_str}' requested, but CUDA is not available."
                    )
            else:
                if device_str.startswith("cuda:"):
                    try:
                        idx = int(device_str.split(":")[-1])
                        count = torch.cuda.device_count()
                        if isinstance(count, int) and idx >= count:
                            raise RuntimeError(
                                f"Device '{device_str}' requested, but only {count} CUDA device(s) are available."
                            )
                    except ValueError:
                        pass
                target_device = device_str if device_str != "auto" else "cuda:0"
        else:
            target_device = "cpu"

        # Dtype resolution
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": torch.bfloat16 if (target_device.startswith("cuda") and torch.cuda.is_bf16_supported()) else (
                torch.float16 if target_device.startswith("cuda") else torch.float32
            ),
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float32)
        return target_device, torch_dtype

    def load(self) -> None:
        """Load the tokenizer and model onto the configured target device."""
        if self._is_loaded and self.model is not None:
            return

        t0 = time.perf_counter()

        # Check quantization prerequisites
        quant_config = None
        if self.config.quantization in ("4bit", "8bit"):
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig

                if self.config.quantization == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
            except (ImportError, Exception) as e:
                raise ImportError(
                    f"Quantization '{self.config.quantization}' requires 'bitsandbytes', which is not installed "
                    "or incompatible with the current environment."
                ) from e

        target_device, torch_dtype = self._resolve_device_and_dtype()
        self._target_device = target_device
        self._torch_dtype = torch_dtype

        if quant_config is not None and hasattr(quant_config, "bnb_4bit_compute_dtype"):
            quant_config.bnb_4bit_compute_dtype = torch_dtype

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_identifier = self.config.model_path or self.config.model_name
        local_files_only = self.config.extra_kwargs.get("local_files_only", True)

        # If model_identifier is a repo id but cached locally in hub snapshots, resolve to the snapshot path
        # to ensure transformers AutoTokenizer avoids remote API checks in offline sandboxed environments.
        if local_files_only and not os.path.exists(model_identifier):
            potential_snap = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{model_identifier.replace('/', '--')}/snapshots"
            )
            if os.path.exists(potential_snap) and os.listdir(potential_snap):
                snap_sub = os.path.join(potential_snap, sorted(os.listdir(potential_snap))[-1])
                if os.path.exists(os.path.join(snap_sub, "config.json")):
                    model_identifier = snap_sub

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_identifier,
                revision=self.config.model_revision,
                trust_remote_code=self.config.extra_kwargs.get("trust_remote_code", False),
                local_files_only=local_files_only,
            )

            # Load model
            model_kwargs: dict[str, Any] = {
                "revision": self.config.model_revision,
                "trust_remote_code": self.config.extra_kwargs.get("trust_remote_code", False),
                "local_files_only": local_files_only,
            }

            if quant_config is not None:
                model_kwargs["quantization_config"] = quant_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch_dtype
                if target_device.startswith("cuda"):
                    model_kwargs["device_map"] = target_device

            self.model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                **model_kwargs,
            )

            if quant_config is None and not hasattr(self.model, "hf_device_map"):
                self.model.to(target_device)

            self.model.eval()
            self._is_loaded = True
            self._load_latency_ms = (time.perf_counter() - t0) * 1000.0

            logger.info(
                f"Loaded local model '{model_identifier}' on {target_device} ({torch_dtype}) "
                f"in {self._load_latency_ms:.2f}ms."
            )

        except Exception as e:
            self.unload()
            raise RuntimeError(
                f"Failed to load Hugging Face model '{model_identifier}': {e}"
            ) from e

    def unload(self) -> None:
        """Release PyTorch model memory and clean CUDA cache."""
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's actual tokenizer."""
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Fallback approximation if tokenizer not yet loaded
        return len(re.findall(r"\w+|[^\w\s]", text))

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Execute local GPU/CPU forward inference with generation parameters."""
        if not self._is_loaded or self.model is None or self.tokenizer is None:
            self.load()

        import torch

        t0 = time.perf_counter()
        prompt_tokens = self.count_tokens(prompt)

        # Enforce prompt truncation if exceeding max_input_tokens
        max_in = kwargs.get("max_input_tokens", self.config.max_input_tokens)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_in)

        # Move to target device
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": kwargs.get("max_output_tokens", self.config.max_output_tokens),
            "do_sample": False if kwargs.get("temperature", self.config.temperature) == 0.0 else True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        temp = kwargs.get("temperature", self.config.temperature)
        if temp > 0.0:
            gen_kwargs["temperature"] = temp

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            # Extract generated tokens only (strip prompt prefix)
            generated_ids = output_ids[0][input_ids.shape[1]:]
            completion_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            completion_tokens = len(generated_ids)
            latency = (time.perf_counter() - t0) * 1000.0

            # Collect GPU memory metadata if on CUDA
            gpu_metadata = {}
            if torch.cuda.is_available() and self._target_device.startswith("cuda"):
                gpu_metadata["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
                gpu_metadata["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)

            return GenerationResult(
                text=completion_text,
                model_name=self.config.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency,
                backend="local",
                status="SUCCESS",
                metadata=gpu_metadata,
            )

        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            return GenerationResult(
                text="",
                model_name=self.config.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=prompt_tokens,
                latency_ms=latency,
                backend="local",
                status="FAILURE",
                error_message=str(e),
            )

    def get_model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "backend": "local",
            "model_name": self.config.model_name,
            "model_revision": self.config.model_revision,
            "device": self._target_device,
            "dtype": str(self._torch_dtype),
            "quantization": self.config.quantization,
            "is_loaded": self._is_loaded,
            "load_latency_ms": round(self._load_latency_ms, 2),
        }
        try:
            import torch
            if torch.cuda.is_available() and self._target_device.startswith("cuda"):
                info["cuda_device_name"] = torch.cuda.get_device_name()
                info["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
                info["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)
        except Exception:
            pass
        return info


# ---------------------------------------------------------------------------
# ConfigurableAPIClient (Remote Hosted Endpoints)
# ---------------------------------------------------------------------------

class ConfigurableAPIClient(LLMClient):
    """Provider-independent REST API client for hosted LLM services.

    Supports:
    - Standard OpenAI-compatible chat completion endpoints.
    - Anthropic and custom JSON endpoints via endpoint configuration.
    - Reading credentials strictly from environment variables.
    - Zero cost / zero network calls during automated testing (mock-tested).
    - Provider-reported token usage ingestion.
    """

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        self.api_key: Optional[str] = None
        self.endpoint_url: str = ""

    def load(self) -> None:
        """Validate API credentials and configure target URL."""
        t0 = time.perf_counter()
        key = self.config.get_api_key()
        if not key:
            raise ValueError(
                f"Missing API key. Environment variable '{self.config.api_key_env_var}' is not set or empty."
            )
        self.api_key = key

        # Resolve endpoint URL
        if self.config.api_endpoint:
            self.endpoint_url = self.config.api_endpoint
        elif self.config.api_provider == "openai":
            self.endpoint_url = "https://api.openai.com/v1/chat/completions"
        elif self.config.api_provider == "anthropic":
            self.endpoint_url = "https://api.anthropic.com/v1/messages"
        else:
            self.endpoint_url = "http://localhost:8000/v1/chat/completions"

        self._is_loaded = True
        self._load_latency_ms = (time.perf_counter() - t0) * 1000.0

    def unload(self) -> None:
        """Clear memory and credentials."""
        self.api_key = None
        self._is_loaded = False

    def count_tokens(self, text: str) -> int:
        """Approximate token count for API models when offline."""
        if not text:
            return 0
        return len(re.findall(r"\w+|[^\w\s]", text))

    def _build_payload(self, prompt: str, **kwargs: Any) -> tuple[dict[str, str], dict[str, Any]]:
        """Construct request headers and body according to provider."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        max_tokens = kwargs.get("max_output_tokens", self.config.max_output_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)

        if self.config.api_provider == "anthropic":
            headers["x-api-key"] = self.api_key or ""
            headers["anthropic-version"] = "2023-06-01"
            headers.pop("Authorization", None)
            payload = {
                "model": self.config.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            # Standard OpenAI / vLLM / Ollama chat format
            payload = {
                "model": self.config.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

        return headers, payload

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Dispatch HTTP request to remote endpoint and process response."""
        if not self._is_loaded:
            self.load()

        import requests

        t0 = time.perf_counter()
        prompt_tokens_est = self.count_tokens(prompt)
        headers, payload = self._build_payload(prompt, **kwargs)

        try:
            timeout = kwargs.get("generation_timeout", self.config.generation_timeout)
            resp = requests.post(
                self.endpoint_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
            latency = (time.perf_counter() - t0) * 1000.0

            if resp.status_code != 200:
                return GenerationResult(
                    text="",
                    model_name=self.config.model_name,
                    prompt_tokens=prompt_tokens_est,
                    completion_tokens=0,
                    total_tokens=prompt_tokens_est,
                    latency_ms=latency,
                    backend="api",
                    status="FAILURE",
                    error_message=f"API error HTTP {resp.status_code}: {resp.text[:200]}",
                )

            data = resp.json()

            # Parse completion and usage
            if self.config.api_provider == "anthropic":
                text = "".join([c.get("text", "") for c in data.get("content", [])])
                usage = data.get("usage", {})
                p_tok = usage.get("input_tokens", prompt_tokens_est)
                c_tok = usage.get("output_tokens", self.count_tokens(text))
            else:
                choices = data.get("choices", [])
                text = choices[0].get("message", {}).get("content", "") if choices else ""
                usage = data.get("usage", {})
                p_tok = usage.get("prompt_tokens", prompt_tokens_est)
                c_tok = usage.get("completion_tokens", self.count_tokens(text))

            return GenerationResult(
                text=text,
                model_name=self.config.model_name,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=p_tok + c_tok,
                latency_ms=latency,
                backend="api",
                status="SUCCESS",
                metadata={"provider": self.config.api_provider, "http_status": resp.status_code},
            )

        except requests.exceptions.Timeout:
            latency = (time.perf_counter() - t0) * 1000.0
            return GenerationResult(
                text="",
                model_name=self.config.model_name,
                prompt_tokens=prompt_tokens_est,
                completion_tokens=0,
                total_tokens=prompt_tokens_est,
                latency_ms=latency,
                backend="api",
                status="TIMEOUT",
                error_message=f"API request timed out after {self.config.generation_timeout}s.",
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            return GenerationResult(
                text="",
                model_name=self.config.model_name,
                prompt_tokens=prompt_tokens_est,
                completion_tokens=0,
                total_tokens=prompt_tokens_est,
                latency_ms=latency,
                backend="api",
                status="FAILURE",
                error_message=str(e),
            )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "backend": "api",
            "model_name": self.config.model_name,
            "provider": self.config.api_provider,
            "endpoint_url": self.endpoint_url,
            "is_loaded": self._is_loaded,
            "api_key_configured": bool(self.api_key),
        }
