# Microbiome loader — human gut taxonomic profiles from curatedMetagenomicData
# (Bioconductor, Waldron/Segata). One dataset PER disease:
#   rows     = gut samples: healthy controls + one disease's patients
#   features = MetaPhlAn relative-abundance taxa
#   target   = healthy vs <disease>  ->  binary classification
#
# No live API: the data is staged offline once (scripts/export_cmd.R -> scripts/cmd_prep.py)
# into CMD_DIR as two tables that share a sample_id index:
#   profiles.parquet  rows=samples, cols=taxa
#   metadata.parquet  rows=samples, cols=curated fields (incl. study_condition)

import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

from pipeline.data.base import CandidateInfo, Dataset
from pipeline import stats
from pipeline.hard_rules import runner as hard_rules

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("CMD_DIR", "data/cmd"))
DISEASE_COL = "study_condition"                            # curatedMetagenomicData condition column
MIN_CASES = 100                                            # skip diseases with too few patients
HEALTHY = {"control"}  # curatedMetagenomicData healthy label
SKIP_LABELS = {"fmt"}  # study_condition values that aren't diseases (interventions etc.)
MAX_CONTROL_RATIO = 3  # cap controls at this multiple of cases, else the pooled set is ~95% healthy
SEED = 0               # deterministic control subsampling: list_candidates count == fetch data


def build_dataset(X_file=DATA_DIR / "profiles.parquet", metadata_file=DATA_DIR / "metadata.parquet"):
    profiles = pd.read_parquet(X_file)
    metadata = pd.read_parquet(metadata_file)
    metadata = metadata[~metadata.index.duplicated(keep="first")]  # sample_id isn't globally unique
    shared = profiles.index.intersection(metadata.index)
    X = profiles.loc[shared]
    disease = metadata.loc[shared, DISEASE_COL].str.strip().str.lower()
    disease.name = "disease"
    return X, disease

def list_candidates(max_candidates=50):
    X, disease = build_dataset()
    n_controls = int(disease.isin(HEALTHY).sum())

    candidates = []
    for label, n_cases in disease.value_counts().items():
        if label in HEALTHY or label in SKIP_LABELS or n_cases < MIN_CASES:
            continue
        kept_controls = min(n_controls, MAX_CONTROL_RATIO * n_cases)       # cap controls per disease
        n_rows = int(n_cases + kept_controls)
        results = hard_rules.run_metadata_checks(
            n_samples=n_rows, n_features=X.shape[1], task_type="classification",
            licence="odbl-1.0", source="cmd", name=label
        )
        if not hard_rules.all_passed(results):
            stats.record(f"cmd-{label}", "cmd",
                         f"curatedMetagenomicData gut (healthy vs {label})", results, "biological")
            continue

        candidates.append(CandidateInfo(
            id=f"cmd-{label}",
            source="cmd",
            name=f"curatedMetagenomicData (healthy vs {label})",
            n_samples=n_rows,
            n_features=X.shape[1],
            task_type="classification",
            licence="odbl-1.0",
            url="https://waldronlab.io/curatedMetagenomicData/",
            metadata={"disease": label},
            domain="biological",
        ))
        if len(candidates) >= max_candidates:
            break

    return candidates

def fetch(candidate):
    X, disease = build_dataset()
    target = candidate.metadata["disease"]

    case_ids = disease.index[disease == target]
    control_ids = disease.index[disease.isin(HEALTHY)]
    n_keep = min(len(control_ids), MAX_CONTROL_RATIO * len(case_ids))      # cap controls per disease
    control_ids = control_ids.to_series().sample(n=n_keep, random_state=SEED).index

    X = pd.concat([X.loc[case_ids], X.loc[control_ids]])
    y = pd.Series(["disease"] * len(case_ids) + ["healthy"] * len(control_ids), name="target")
    X = np.log1p(X).reset_index(drop=True)

    results = hard_rules.run_data_checks(X=X, y=y, task_type="classification")
    if not hard_rules.all_passed(results):
        return None, results

    return Dataset(
        id=candidate.id,
        source="cmd",
        name=candidate.name,
        X=X,
        y=y,
        task_type="classification",
        metadata={"licence": "odbl-1.0", "disease": target, "url": candidate.url},
        domain="biological",
    ), results
