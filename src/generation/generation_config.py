"""generation_config.py — Configuration dataclass for the LLM generation interface.

Defines GenerationConfig with validation for mock, local Hugging Face, and remote API
generation backends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


SUPPORTED_BACKENDS = ("mock", "local", "api")
SUPPORTED_DEVICES = ("cuda", "cpu", "auto")
SUPPORTED_DTYPES = ("float16", "bfloat16", "float32", "auto")
SUPPORTED_QUANTIZATIONS = (None, "4bit", "8bit", "awq", "gptq")


@dataclass
class GenerationConfig:
    """Validated configuration for LLM generation backends.

    Attributes:
        backend: Execution backend ('mock', 'local', 'api').
        model_name: Hugging Face repository ID, API model name, or mock identifier.
        model_revision: Optional specific commit hash or branch tag for model reproducibility.
        model_path: Optional explicit local directory path containing model weights.
        device: Target execution device ('cuda', 'cpu', 'auto').
        dtype: Numerical precision for local weights ('float16', 'bfloat16', 'float32', 'auto').
        quantization: Quantization format (None, '4bit', '8bit', 'awq', 'gptq').
        temperature: Sampling temperature (>= 0.0; 0.0 indicates greedy deterministic decoding).
        max_output_tokens: Maximum tokens the model is permitted to generate (> 0).
        max_input_tokens: Maximum prompt tokens allowed before deterministic truncation (> 0).
        generation_timeout: Timeout in seconds for a single generation call (> 0.0).
        batch_size: Batch size for batched generation (>= 1).
        max_repair_attempts: Maximum repair iterations permitted on verification failure (>= 0).
        stop_sequences: List of string sequences that halt generation.
        api_provider: Provider name for API backend ('openai', 'anthropic', 'gemini', 'generic').
        api_endpoint: Optional custom base URL or full endpoint URL for API requests.
        api_key_env_var: Name of the environment variable containing the secret API key.
        extra_kwargs: Additional backend-specific parameters (e.g. top_p, repetition_penalty).
    """

    backend: str = "mock"
    model_name: str = "mock-financial-llm"
    model_revision: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    quantization: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 512
    max_input_tokens: int = 2048
    generation_timeout: float = 60.0
    batch_size: int = 1
    max_repair_attempts: int = 1
    stop_sequences: list[str] = field(default_factory=list)
    api_provider: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key_env_var: str = "LLM_API_KEY"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values and reject incompatible settings."""
        self._validate()

    def _validate(self) -> None:
        # Backend validation
        backend_lower = str(self.backend).lower().strip()
        if backend_lower not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Invalid backend '{self.backend}'. Must be one of {SUPPORTED_BACKENDS}."
            )
        self.backend = backend_lower

        # Device validation
        device_lower = str(self.device).lower().strip()
        if device_lower not in SUPPORTED_DEVICES and not device_lower.startswith("cuda:"):
            raise ValueError(
                f"Invalid device '{self.device}'. Must be one of {SUPPORTED_DEVICES} or 'cuda:<index>'."
            )
        self.device = device_lower

        # Dtype validation
        dtype_lower = str(self.dtype).lower().strip()
        if dtype_lower not in SUPPORTED_DTYPES:
            raise ValueError(
                f"Invalid dtype '{self.dtype}'. Must be one of {SUPPORTED_DTYPES}."
            )
        self.dtype = dtype_lower

        # Quantization validation
        quant_lower = self.quantization.lower().strip() if self.quantization else None
        if quant_lower not in SUPPORTED_QUANTIZATIONS:
            raise ValueError(
                f"Invalid quantization '{self.quantization}'. Must be one of {SUPPORTED_QUANTIZATIONS}."
            )
        self.quantization = quant_lower

        # Numeric bounds validation
        if self.temperature < 0.0:
            raise ValueError(f"Temperature must be non-negative, got {self.temperature}.")

        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}.")

        if self.max_input_tokens <= 0:
            raise ValueError(f"max_input_tokens must be positive, got {self.max_input_tokens}.")

        if self.generation_timeout <= 0.0:
            raise ValueError(f"generation_timeout must be positive, got {self.generation_timeout}.")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")

        if self.max_repair_attempts < 0:
            raise ValueError(f"max_repair_attempts must be non-negative, got {self.max_repair_attempts}.")

        # Backend-specific validations
        if self.backend == "api":
            if not self.api_provider and not self.api_endpoint:
                raise ValueError(
                    "When backend is 'api', either 'api_provider' or 'api_endpoint' must be specified."
                )

        if self.backend == "local" and self.quantization and self.device == "cpu":
            raise ValueError("Quantized execution (e.g. 4-bit, 8-bit) is not supported on CPU.")

    def get_api_key(self) -> Optional[str]:
        """Retrieve the API key from environment variables.

        Never stores or hardcodes the key inside the config instance.
        Returns None if the environment variable is not set or empty.
        """
        if not self.api_key_env_var:
            return None
        val = os.environ.get(self.api_key_env_var, "").strip()
        return val if val else None

    def to_dict(self) -> dict[str, Any]:
        """Return a clean dictionary representation of the configuration.

        Ensures no secrets or sensitive environment credentials are exposed.
        """
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_input_tokens": self.max_input_tokens,
            "generation_timeout": self.generation_timeout,
            "batch_size": self.batch_size,
            "max_repair_attempts": self.max_repair_attempts,
            "stop_sequences": list(self.stop_sequences),
            "api_provider": self.api_provider,
            "api_endpoint": self.api_endpoint,
            "api_key_configured": bool(self.get_api_key()),
            "extra_kwargs": dict(self.extra_kwargs),
        }
