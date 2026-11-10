"""generation — Constrained LLM Generation Engine for LLM-FINANCE.

This package provides:
1. GenerationConfig: Validated configuration dataclass for local, mock, and API backends.
2. LLMClient: Abstract provider-agnostic interface for text generation.
3. MockLLMClient: Deterministic offline mock for CI and testing.
4. HuggingFaceLocalClient: Local PyTorch/Transformers model inference on CUDA/CPU.
5. ConfigurableAPIClient: Provider-independent REST API client for hosted models.
6. ContextBuilder & ContextBuildResult: Evidence prompt assembly and token budget capping.
7. Prompts: Constrained SEC 10-K financial prompts, few-shot demonstrations, and repair prompts.
8. OutputParser & ParseResult: Strict JSON extraction, schema validation, and referential integrity checks.
9. ClaimGenerator & ClaimGenerationResult: End-to-end retrieval, generation, and verification orchestrator.
"""

from .claim_generator import (
    ClaimGenerationResult,
    ClaimGenerator,
    weighted_reciprocal_rank_fusion,
)
from .context_builder import (
    ContextBuildResult,
    ContextBuilder,
    default_token_counter,
    format_child_chunk,
    format_parent_expanded_item,
)
from .generation_config import GenerationConfig
from .llm_client import (
    ConfigurableAPIClient,
    GenerationResult,
    HuggingFaceLocalClient,
    LLMClient,
    MockLLMClient,
)
from .output_parser import (
    ALLOWED_OPERATIONS,
    OutputParser,
    ParseResult,
    extract_json_string,
    is_numeric_value,
)
from .prompts import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_generation_prompt,
    build_repair_prompt,
    build_system_prompt,
    build_user_prompt,
)
from .repair_engine import (
    RepairAttemptAudit,
    RepairEngine,
    RepairabilityClassification,
    classify_repairability,
    compute_repair_metrics,
)

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "LLMClient",
    "MockLLMClient",
    "HuggingFaceLocalClient",
    "ConfigurableAPIClient",
    "ContextBuilder",
    "ContextBuildResult",
    "default_token_counter",
    "format_child_chunk",
    "format_parent_expanded_item",
    "SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "build_system_prompt",
    "build_user_prompt",
    "build_generation_prompt",
    "build_repair_prompt",
    "OutputParser",
    "ParseResult",
    "extract_json_string",
    "is_numeric_value",
    "ALLOWED_OPERATIONS",
    "ClaimGenerator",
    "ClaimGenerationResult",
    "weighted_reciprocal_rank_fusion",
    "RepairEngine",
    "RepairAttemptAudit",
    "RepairabilityClassification",
    "classify_repairability",
    "compute_repair_metrics",
]
