EVAL_DATASET = [
    # --- Original 5 (kept for continuity with prior eval runs) ---
    {
        "id": "sglt2_dka_relationship",
        "question": "What is the relationship between SGLT2 inhibitors and diabetic ketoacidosis?",
        "expected_sources": ["medi-105-e47272.pdf"],
        "expected_terms": [
            "SGLT2",
            "diabetic ketoacidosis",
            "euglycemic",
        ],
    },
    {
        "id": "diabetes_insipidus_sglt2",
        "question": "What does the literature say about SGLT2 inhibitors and diabetes insipidus?",
        "expected_sources": ["biol-2025-1194.pdf"],
        "expected_terms": [
            "SGLT2",
            "diabetes insipidus",
        ],
    },
    {
        "id": "wolfram_cost_burden",
        "question": "What economic burden is associated with Wolfram syndrome?",
        "expected_sources": ["13023_2019_Article_1149.pdf"],
        "expected_terms": [
            "Wolfram syndrome",
            "cost",
            "burden",
        ],
    },
    {
        "id": "pancreatic_cancer_privacy_ml",
        "question": "How is privacy-aware machine learning used in pancreatic cancer diagnosis?",
        "expected_sources": ["12911_2024_Article_2657.pdf"],
        "expected_terms": [
            "privacy",
            "machine learning",
            "pancreatic cancer",
        ],
    },
    {
        "id": "breast_cancer_field_cancerization",
        "question": "What is field cancerization in breast cancer?",
        "expected_sources": ["PATH-257-561.pdf"],
        "expected_terms": [
            "field cancerization",
            "breast cancer",
        ],
    },

    # --- Hypertension / diabetes case-report batch (verified against source text) ---
    {
        "id": "minoxidil_duration",
        "question": "How long does minoxidil's pharmacologic effect last after dosing?",
        "expected_sources": ["nihms837917.pdf"],
        "expected_terms": [
            "minoxidil",
            "3-4 day",
            "duration",
        ],
    },
    {
        "id": "hypertension_brain_chemical",
        "question": "Which brain chemical system links high blood pressure to effects on learning and memory?",
        "expected_sources": ["IJHT2012-701385.pdf"],
        "expected_terms": [
            "angiotensin",
            "renin-angiotensin",
            "learning",
        ],
    },
    {
        "id": "hypertension_comorbidities_china",
        "question": "What three comorbidities did the Chinese hypertension study examine?",
        "expected_sources": ["ijmsv14p0201.pdf"],
        "expected_terms": [
            "coronary heart disease",
            "diabetes",
            "hyperlipidemia",
        ],
    },
    {
        "id": "edka_incidence_rate",
        "question": "What is the incidence rate of euglycemic diabetic ketoacidosis?",
        "expected_sources": ["medi-105-e47272.pdf"],
        "expected_terms": [
            "0.8",
            "1.1%",
            "EDKA",
        ],
    },
    {
        "id": "hhs_insulin_dose",
        "question": "What insulin infusion rate was used to treat the hyperosmolar hyperglycemic state case?",
        "expected_sources": ["biol-2025-1194.pdf"],
        "expected_terms": [
            "0.04",
            "IU/kg/min",
            "insulin",
        ],
    },
    {
        "id": "shared_complication_case_reports",
        "question": "What diabetic complication is described in both the burns case report and the hyperosmolar hyperglycemic state case report?",
        "expected_sources": ["medi-105-e47272.pdf", "biol-2025-1194.pdf"],
        "expected_terms": [
            "central diabetes insipidus",
            "CDI",
        ],
    },

    # --- Infectious disease batch (verified against source text) ---
    {
        "id": "rat_sensitivity_value",
        "question": "What sensitivity value was found for rapid antigen tests in detecting COVID-19?",
        "expected_sources": ["srx-22-1939.pdf"],
        "expected_terms": [
            "67.1%",
            "sensitivity",
        ],
    },
    {
        "id": "neutralizing_antibody_target",
        "question": "Which domain of the SARS-CoV-2 spike protein do neutralizing antibodies primarily target?",
        "expected_sources": ["11596_2021_Article_2470.pdf"],
        "expected_terms": [
            "RBD",
            "ACE2",
        ],
    },
    {
        "id": "fluconazole_resistant_candida_burden",
        "question": "How many fluconazole-resistant Candida infections occur annually in the United States?",
        "expected_sources": ["kvir-08-02-1196300.pdf"],
        "expected_terms": [
            "3,400",
            "fluconazole",
        ],
    },
    {
        "id": "resistance_vs_tolerance_definition",
        "question": "What is the difference between antimicrobial resistance and tolerance in terms of measurement?",
        "expected_sources": ["ijms-22-02224.pdf"],
        "expected_terms": [
            "MIC",
            "MDK",
            "tolerance",
        ],
    },
]


def get_eval_dataset() -> list[dict]:
    return EVAL_DATASET