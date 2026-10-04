from validator import (
    load_corpus,
    build_chunk_lookup,
    validate_claim,
)


chunks = load_corpus()

lookup = build_chunk_lookup(chunks)


claim = {
    "claims": [
        {
            "id": "claim_1",

            "claim_text":
                "The interest expense in 2009 was "
                "-600 million USD.",

            "value": -600,

            "unit": "million USD",

            "citations": [
                "table_1"
            ],

            "operands": [
                {
                    "id": "operand_1",

                    "chunk_id": "table_1",

                    "span_text":
                        "interest expense | 2009: -600",

                    "context_text":
                        "millions of dollars: interest expense | "
                        "2009: -600 | 2008: -511",

                    "normalized_value": -600,

                    "unit": "million USD"
                }
            ],

            "program": [],

            "referent": {
                "entity": "Union Pacific",
                "period": "2009",
                "unit": "million USD"
            },

            "status": "answered"
        }
    ],

    "final_claim_id": "claim_1"
}


result = validate_claim(
    claim,
    lookup,
    example_id="UNP/2009/page_34.pdf-1"
)

print(result)