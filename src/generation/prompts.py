"""prompts.py — Constrained financial reasoning prompts for SEC 10-K claim generation.

Generates system, user, and diagnostic repair prompts enforcing:
- Strict evidence grounding (no hallucinated chunk IDs or values).
- Verbatim span citations matching chunk text.
- Explicit accounting polarity handling (expenses/outflows vs revenues/inflows).
- Distinction between raw signed differences and economic magnitude changes.
- Deterministic schema adherence matching CLAIM_SCHEMA.
- Explicit abstention when evidence is insufficient.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

SYSTEM_PROMPT = """You are a senior SEC 10-K financial analyst and symbolic reasoning engine.
Your task is to answer financial questions based strictly on the provided evidence chunks from SEC 10-K filings.

CRITICAL INVARIANTS:
1. Grounding: Answer using ONLY facts, numbers, and dates present in the provided evidence. Never estimate, speculate, or fabricate numbers, chunk IDs, or document sources.
2. Exact Span Citation: Every operand must cite an exact verbatim substring (`span_text`) from the specified `chunk_id`. The `chunk_id` MUST be copied verbatim from the `[Evidence Chunk: <id>]` header.
3. Accounting Polarity & Line Item Conventions:
   - Distinguish revenues/inflows from expenses/costs/losses/outflows.
   - For revenues, assets, or standard changes: use `operation: "subtract"` (current - previous).
   - For expenses, costs, or outflows reported as negative numbers (e.g. -600 and -511), financial narratives report growth in magnitude (an increase of 89). Use `operation: "economic_change"` or `operation: "magnitude_change"`.
   - For growth rates, use `operation: "percentage_change"`.
   - Allowed operations: ["add", "subtract", "multiply", "divide", "economic_change", "magnitude_change", "percentage_change", "abs"].
4. Referential Integrity:
   - In `program`, every referenced operand ID must be defined in `operands` or be the `id` of an earlier program step.
   - `final_claim_id` must match the `id` of the primary claim in `claims`.
5. Abstention:
   - If the provided evidence lacks the necessary data to answer the question, or if required operands cannot be grounded, you MUST abstain by returning `status: "insufficient_evidence"`, with `value: null`, `operands: []`, and `program: []`.
6. Output Format:
   - Output MUST be a single, valid JSON object adhering strictly to the schema below.
   - Do NOT wrap the JSON in commentary or conversational text.
"""

FEW_SHOT_EXAMPLES = [
    {
        "question": "What was the change in total net sales from 2018 to 2019?",
        "context": (
            "[Evidence Chunk: table_row_0] (Document: AAPL_2019, Source: table)\n"
            "Net sales | $ 260,174 | $ 265,595\n"
            "[Evidence Chunk: text_0] (Document: AAPL_2019, Source: pre_text)\n"
            "All monetary amounts in millions of US dollars for years ended September 28, 2019 and September 29, 2018."
        ),
        "output": {
            "claims": [
                {
                    "id": "c1",
                    "claim_text": "Total net sales decreased by 5,421 million USD from 2018 to 2019.",
                    "value": -5421.0,
                    "unit": "million USD",
                    "citations": ["table_row_0", "text_0"],
                    "operands": [
                        {
                            "id": "op_sales_2019",
                            "chunk_id": "table_row_0",
                            "span_text": "$ 260,174",
                            "context_text": "Net sales | $ 260,174 | $ 265,595",
                            "normalized_value": 260174.0,
                            "unit": "million USD",
                            "period": "2019",
                        },
                        {
                            "id": "op_sales_2018",
                            "chunk_id": "table_row_0",
                            "span_text": "$ 265,595",
                            "context_text": "Net sales | $ 260,174 | $ 265,595",
                            "normalized_value": 265595.0,
                            "unit": "million USD",
                            "period": "2018",
                        },
                    ],
                    "program": [
                        {
                            "id": "step_1",
                            "operation": "subtract",
                            "operands": ["op_sales_2019", "op_sales_2018"],
                        }
                    ],
                    "referent": {
                        "entity": "total net sales",
                        "period": "2019",
                        "unit": "million USD",
                    },
                    "status": "answered",
                }
            ],
            "final_claim_id": "c1",
        },
    },
    {
        "question": "By how much did interest expense increase from 2008 to 2009?",
        "context": (
            "[Evidence Chunk: table_row_4] (Document: MSFT_2009, Source: table)\n"
            "Interest expense | $ (600) | $ (511)\n"
            "[Evidence Chunk: text_2] (Document: MSFT_2009, Source: pre_text)\n"
            "Amounts in millions. For the fiscal years ended June 30, 2009 and 2008."
        ),
        "output": {
            "claims": [
                {
                    "id": "c1",
                    "claim_text": "Interest expense increased by 89 million USD from 2008 to 2009.",
                    "value": 89.0,
                    "unit": "million USD",
                    "citations": ["table_row_4", "text_2"],
                    "operands": [
                        {
                            "id": "op_interest_2009",
                            "chunk_id": "table_row_4",
                            "span_text": "(600)",
                            "context_text": "Interest expense | $ (600) | $ (511)",
                            "normalized_value": -600.0,
                            "unit": "million USD",
                            "period": "2009",
                        },
                        {
                            "id": "op_interest_2008",
                            "chunk_id": "table_row_4",
                            "span_text": "(511)",
                            "context_text": "Interest expense | $ (600) | $ (511)",
                            "normalized_value": -511.0,
                            "unit": "million USD",
                            "period": "2008",
                        },
                    ],
                    "program": [
                        {
                            "id": "step_1",
                            "operation": "economic_change",
                            "operands": ["op_interest_2009", "op_interest_2008"],
                        }
                    ],
                    "referent": {
                        "entity": "interest expense",
                        "period": "2009",
                        "unit": "million USD",
                    },
                    "status": "answered",
                }
            ],
            "final_claim_id": "c1",
        },
    },
    {
        "question": "What was the research and development expenditure for fiscal year 2012?",
        "context": (
            "[Evidence Chunk: table_row_1] (Document: ORCL_2015, Source: table)\n"
            "R&D expenses | $ 5,841 | $ 5,587 | $ 5,066\n"
            "[Evidence Chunk: text_0] (Document: ORCL_2015, Source: pre_text)\n"
            "Years ended May 31, 2015, 2014, and 2013. Amounts in millions."
        ),
        "output": {
            "claims": [
                {
                    "id": "c1",
                    "claim_text": "The provided evidence does not report research and development expenditures for fiscal year 2012.",
                    "value": None,
                    "unit": "million USD",
                    "citations": [],
                    "operands": [],
                    "program": [],
                    "referent": {
                        "entity": "research and development expenditure",
                        "period": "2012",
                        "unit": "million USD",
                    },
                    "status": "insufficient_evidence",
                }
            ],
            "final_claim_id": "c1",
        },
    },
]


def build_system_prompt() -> str:
    """Return the complete system prompt for financial claim generation."""
    return SYSTEM_PROMPT.strip()


def build_user_prompt(
    question: str,
    context_text: str,
    example_id: Optional[str] = None,
    include_few_shot: bool = True,
) -> str:
    """Build the user prompt combining query, context, and optional few-shot demonstrations."""
    sections = []

    if include_few_shot:
        sections.append("### EXAMPLES OF VALID CLAIMS AND PROGRAMS ###\n")
        for i, demo in enumerate(FEW_SHOT_EXAMPLES, start=1):
            sections.append(f"Example {i}:")
            sections.append(f"Question: {demo['question']}")
            sections.append(f"Evidence:\n{demo['context']}")
            sections.append(
                f"Generated Structured Claim JSON:\n{json.dumps(demo['output'], indent=2)}\n"
            )
        sections.append("### END OF EXAMPLES ###\n")

    sections.append("### RETRIEVED SEC 10-K EVIDENCE ###")
    if example_id:
        sections.append(f"Target Document ID: {example_id}")
    sections.append(context_text if context_text.strip() else "[NO EVIDENCE RETRIEVED]")
    sections.append("### END OF EVIDENCE ###\n")

    sections.append("### FINANCIAL QUESTION ###")
    sections.append(question.strip())
    sections.append("\nGenerate the structured claim JSON:")

    return "\n".join(sections)


def build_generation_prompt(
    question: str,
    context_text: str,
    example_id: Optional[str] = None,
    include_few_shot: bool = True,
) -> str:
    """Build a unified single prompt containing system instructions, demonstrations, context, and question."""
    system = build_system_prompt()
    user = build_user_prompt(
        question=question,
        context_text=context_text,
        example_id=example_id,
        include_few_shot=include_few_shot,
    )
    return f"{system}\n\n{'=' * 40}\n\n{user}"


def build_repair_prompt(
    question: str,
    context_text: str,
    previous_output: str,
    errors: Sequence[str],
    example_id: Optional[str] = None,
) -> str:
    """Build a targeted single-turn repair prompt giving diagnostic error feedback."""
    bullet_errors = "\n".join(f"- {err}" for err in errors)

    return (
        f"{build_system_prompt()}\n\n"
        f"### REPAIR REQUEST: VERIFICATION FAILED ###\n"
        f"Your previous output failed deterministic verification with the following error(s):\n"
        f"{bullet_errors}\n\n"
        f"### PREVIOUS GENERATION ###\n"
        f"{previous_output}\n\n"
        f"### RETRIEVED EVIDENCE ###\n"
        f"{context_text}\n\n"
        f"### QUESTION ###\n"
        f"{question}\n\n"
        f"Please repair the errors above and generate a corrected structured claim JSON adhering strictly to the schema."
    )
