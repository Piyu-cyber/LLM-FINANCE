from pipeline import run_verification_pipeline


EXAMPLE_ID = "UNP/2009/page_34.pdf-1"


claim = {
    "claims": [
        {
            "id": "claim_1",

            "claim_text": (
                "The change in interest expense "
                "from 2008 to 2009 was -89 million USD."
            ),

            "value": -89,

            "unit": "million USD",

            "citations": [
                "table_1"
            ],

            "operands": [

                # --------------------------------------------------
                # Operand 1: 2009 interest expense
                # --------------------------------------------------

                {
                    "id": "op_2009",

                    "chunk_id": "table_1",

                    "span_text": (
                        "interest expense | 2009: -600 ( 600 )"
                    ),

                    "context_text": (
                        "millions of dollars: interest expense | "
                        "2009: -600 ( 600 ) | "
                        "2008: -511 ( 511 ) | "
                        "2007: -482( 482 )"
                    ),

                    "normalized_value": -600,

                    "unit": "million USD",

                    "period": "2009",
                },

                # --------------------------------------------------
                # Operand 2: 2008 interest expense
                # --------------------------------------------------

                {
                    "id": "op_2008",

                    "chunk_id": "table_1",

                    "span_text": (
                        "2008: -511 ( 511 )"
                    ),

                    "context_text": (
                        "millions of dollars: interest expense | "
                        "2009: -600 ( 600 ) | "
                        "2008: -511 ( 511 ) | "
                        "2007: -482( 482 )"
                    ),

                    "normalized_value": -511,

                    "unit": "million USD",

                    "period": "2008",
                },
            ],

            # ------------------------------------------------------
            # Calculation program
            # ------------------------------------------------------

            "program": [
                {
                    "id": "step_1",

                    "operation": "subtract",

                    "operands": [
                        "op_2009",
                        "op_2008",
                    ],
                }
            ],

            # ------------------------------------------------------
            # Final claim referent
            # ------------------------------------------------------

            "referent": {
                "entity": "interest expense",
                "period": "2009",
                "unit": "million USD",
            },

            "status": "answered",
        }
    ],

    "final_claim_id": "claim_1",
}


if __name__ == "__main__":

    result = run_verification_pipeline(
        claim,
        EXAMPLE_ID,
    )

    print(result)