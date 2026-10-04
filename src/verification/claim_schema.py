CLAIM_SCHEMA = {
    "type": "object",

    "required": [
        "claims",
        "final_claim_id"
    ],

    "properties": {

        "claims": {
            "type": "array",

            "items": {
                "type": "object",

                "required": [
                    "id",
                    "claim_text",
                    "value",
                    "unit",
                    "citations",
                    "operands",
                    "program",
                    "referent",
                    "status",
                ],

                "properties": {

                    "id": {
                        "type": "string"
                    },

                    "claim_text": {
                        "type": "string"
                    },

                    "value": {},

                    "unit": {
                        "type": "string"
                    },

                    "citations": {
                        "type": "array",

                        "items": {
                            "type": "string"
                        },
                    },

                    "operands": {
                        "type": "array",

                        "items": {
                            "type": "object",

                            "required": [
                                "id",
                                "chunk_id",
                                "span_text",
                                "context_text",
                                "normalized_value",
                                "unit",
                                "period",
                            ],

                            "properties": {

                                "id": {
                                    "type": "string"
                                },

                                "chunk_id": {
                                    "type": "string"
                                },

                                "span_text": {
                                    "type": "string"
                                },

                                "context_text": {
                                    "type": "string"
                                },

                                "normalized_value": {},

                                "unit": {
                                    "type": "string"
                                },

                                "period": {
                                    "type": "string"
                                },
                            },
                        },
                    },

                    "program": {
                        "type": "array"
                    },

                    "referent": {
                        "type": "object",

                        "required": [
                            "entity",
                            "period",
                            "unit",
                        ],

                        "properties": {

                            "entity": {
                                "type": "string"
                            },

                            "period": {
                                "type": "string"
                            },

                            "unit": {
                                "type": "string"
                            },
                        },
                    },

                    "status": {
                        "type": "string",

                        "enum": [
                            "answered",
                            "insufficient_evidence",
                        ],
                    },
                },
            },
        },

        "final_claim_id": {
            "type": "string"
        },
    },
}