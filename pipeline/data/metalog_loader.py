# Metalog metagenomics loader (EMBL Bork Group, https://metalog.embl.de).
#
# What gets loaded (one dataset PER disease, like metagenomics_loader):
#   rows     = gut samples: all healthy people + one disease's patients
#   features = microbial taxa (mOTUs3.0 / MetaPhlAn4 relative abundances)
#   target   = healthy vs <disease>  ->  binary classification
#
# NO API (the site sits behind an anti-bot wall), so the gut DB is downloaded
# once by hand into METALOG_DIR as two tables that share a sample-id index:
#   profiles.parquet  rows=samples, cols=taxa
#   metadata.parquet  rows=samples, cols=curated fields (incl. disease status)

import logging
import os
from pathlib import Path

import pandas as pd

from pipeline.data.base import CandidateInfo
from pipeline import stats
from pipeline.hard_rules import runner as hard_rules

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("METALOG_DIR", "data/metalog"))
DISEASE_COL = "subject_disease_status"                     # target column in metadata.parquet
MIN_CASES = 100                                            # skip diseases with too few patients
HEALTHY = {"healthy", "control", "no"}  # values counted as healthy


def build_dataset(X_file, metadata_file):
    #TODO implement
    return X_file, metadata_file

def list_candidates(max_candidates=50):
    X, disease = build_dataset()

    candidates = []
    for label, n_cases in disease.value_counts().items():
        if label in HEALTHY or n_cases < MIN_CASES:
            continue
        n_rows = int((disease.isin(HEALTHY) | (disease == label)).sum())  # healthy + this disease
        results = hard_rules.run_metadata_checks(
            n_samples=n_rows, n_features=X.shape[1], task_type="classification",
            licence="odbl-1.0", source="metalog", name=label
        )
        if not hard_rules.all_passed(results):
            stats.record(f"metalog-{label}", "metalog",
                         f"Metalog gut (healthy vs {label})", results, "biological")
            continue

        candidates.append(CandidateInfo(
            id=f"metalog-{label}",
            source="metalog",
            name=f"Metalog (healthy vs {label})",
            n_samples=n_rows,
            n_features=X.shape[1],
            task_type="classification",
            licence="odbl-1.0",
            url="https://metalog.embl.de",
            metadata={"disease": label},         
            domain="biological",
        ))
        if len(candidates) >= max_candidates:
            break

    return candidates
