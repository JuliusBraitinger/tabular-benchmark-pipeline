"""This file is the central configuration file for the rules in this pipeline. 
It contains all the thresholds and parameters that are used in the rules.
 A flowchart of each rule can be seen in the readme."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Dimension thresholds (A4) ---
MIN_ROWS = 150
MIN_FEATURES = 1000
# Upper bound: a 16 GB box can't hold a dense float32 matrix much larger than
# this once concat / pyarrow / joblib copies are factored in. Microarray
# studies above this cap (typically methylation/CNV with 450k-850k probes)
# are skipped at candidate listing time to avoid OOM kills.
MAX_FEATURES = 250_000

# --- Kaggle loader thresholds ---
KAGGLE_MIN_BYTES = 5_000_000      # 5 MB: rough A4 lower bound on zip size
KAGGLE_MAX_BYTES = 2_000_000_000  # 2 GB: stay under MAX_FEATURES OOM territory
KAGGLE_TARGET_NAMES = (
    "target", "label", "class", "y", "outcome", "survived", "diagnosis",
)
KAGGLE_TARGET_SUFFIXES = ("_label", "_target", "_class")

# --- Licence allow-list (A5) ---
ALLOWED_LICENCES = frozenset({
    "cc0-1.0", "cc0", "public-domain",
    "cc-by-4.0", "cc-by-sa-4.0", "cc-by-sa-3.0",  # ChEMBL is CC BY-SA 3.0
    "mit", "apache-2.0",
    "bsd-2-clause", "bsd-3-clause",
    "odbl-1.0",
    "public",
})

# --- Accepted task types (A1) ---
ACCEPTED_TASKS = frozenset({
    "classification", "regression",
    "supervised_classification", "supervised_regression",
})

# AHP start weights for the soft rules. Only used by the CRITIC method
# to derive objective weights - the final scoring combines AHP + CRITIC.
AHP_WEIGHTS = {"S1": 10, "S2": 10, "S3": 15, "S4": 20, "S6": 5}
TOTAL_POINTS = 60
PASS_THRESHOLD = 0.6  # 36/60
DIVERGENCE_THRESHOLD = 3  # |AHP_pts - CRITIC_pts| > 3 -> diverge
AHP_PART = 0.6  # weight of AHP in the final score; 0.4 is CRITIC
WEIGHTS_PATH = "data/critic_weights.json"

# --- API settings ---
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
GDC_BASE_URL = "https://api.gdc.cancer.gov"
ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_DELAY = 0.4  # seconds between API calls
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
