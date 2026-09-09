"""Central configuration for the Smart Inquiry Triage Assistant.

Everything tunable lives here so that the pipeline itself stays readable and
so that model choices can be overridden with environment variables without
touching code.
"""

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAST_CASES_CSV = DATA_DIR / "past_cases.csv"
TAXONOMY_JSON = DATA_DIR / "taxonomy.json"

CHROMA_DIR = ROOT / ".chroma"
COLLECTION_NAME = "past_cases"

# --- Models --------------------------------------------------------------
# Both run locally through Ollama. See README for why local models were
# chosen over a hosted API for this use case.

CHAT_MODEL = os.getenv("TRIAGE_CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("TRIAGE_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Low temperature: triage is a classification task, not a creative one.
CHAT_TEMPERATURE = float(os.getenv("TRIAGE_TEMPERATURE", "0.0"))

# --- Pipeline defaults ---------------------------------------------------

DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

PRIORITY_LEVELS = ("low", "medium", "high")
PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2}

# --- Confidence weights --------------------------------------------------
# The three signals that make up the confidence score. Rationale for the
# split is documented in README.md under "How confidence is computed".

W_CROSSCHECK = 0.40   # do the classifier and the neighbours agree?
W_AGREEMENT = 0.35    # how strongly do the neighbours support the label?
W_SIMILARITY = 0.25   # how close are the neighbours at all?

# A high-priority neighbour this similar pulls the predicted priority up one
# level. Under-triaging a safety issue costs more than over-triaging a routine
# one, so the bias is deliberately asymmetric.
#
# 0.858 is the 75th percentile of in-domain top-1 similarity, measured by
# leave-one-out retrieval over all 300 past cases. An earlier value of 0.75
# sat below the *minimum* in-domain similarity (0.775), so it fired on 61% of
# cases against a true high-priority rate of 18%. At 0.858 it fires on 9%.
# See README for the sweep.
SAFETY_BIAS_SIMILARITY = 0.858

# --- Similarity normalisation -------------------------------------------
# Raw cosine similarity from nomic-embed-text is compressed: in-domain
# inquiries score 0.77-0.93 while unrelated text still scores 0.70-0.76.
# Fed into the confidence score raw, the term is almost constant and adds no
# discrimination. These bounds stretch the useful band across [0, 1] and were
# read off the measured distribution (floor just under the off-topic ceiling,
# ceiling at roughly the in-domain 95th percentile).
SIMILARITY_FLOOR = 0.72
SIMILARITY_CEILING = 0.90
