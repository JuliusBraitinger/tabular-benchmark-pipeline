# curatedMetagenomicData loader — turns the gut-microbiome parquet (staged by
# scripts/export_cmd.R + scripts/cmd_prep.py into CMD_DIR) into ML datasets.
#   profiles.parquet = samples x taxa (the features)
#   metadata.parquet = curated fields (study_condition, study_name, age, BMI, ...)
#
#   classification — healthy vs one disease. Controls come from the disease's OWN studies
#                    (removes cross-study batch effects) and are balanced 1:1 with the cases
#                    (so the A3 signal test isn't fooled by the bigger class).
#   regression     — predict a continuous trait (age, BMI, a lab value)

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data.base import CandidateInfo, Dataset
from pipeline import stats
from pipeline.hard_rules import runner as hard_rules

DATA_DIR = Path(os.environ.get("CMD_DIR", "data/cmd"))
URL = "https://waldronlab.io/curatedMetagenomicData/"

# classification (one dataset per disease)
DISEASE_COL = "study_condition"
STUDY_COL = "study_name"
HEALTHY = {"control"}
SKIP_LABELS = {"fmt"}      # a value that isn't a disease
MIN_CASES = 50
MIN_CONTROLS = 50
SEED = 0

# regression (one dataset per continuous trait)
TARGETS = ["age", "BMI", "hba1c", "ldl", "hdl", "cholesterol", "triglycerides",
           "hscrp", "creatinine", "systolic_p", "dyastolic_p", "albumine",
           "gestational_age", "birth_weight"]


def build_dataset():
    # load both tables and keep only samples that appear in both
    X = pd.read_parquet(DATA_DIR / "profiles.parquet")
    meta = pd.read_parquet(DATA_DIR / "metadata.parquet")
    meta = meta[~meta.index.duplicated(keep="first")]   # sample_id isn't globally unique
    shared = X.index.intersection(meta.index)
    return X.loc[shared], meta.loc[shared]


def list_candidates(max_candidates=50):
    X, meta = build_dataset()
    P = X.shape[1]
    disease = meta[DISEASE_COL].str.strip().str.lower()
    study = meta[STUDY_COL]
    candidates = []

    # classification: healthy vs each disease
    for label, n_cases in disease.value_counts().items():
        if label in HEALTHY or label in SKIP_LABELS or n_cases < MIN_CASES:
            continue
        studies = study[disease == label].unique()
        n_controls = int((disease.isin(HEALTHY) & study.isin(studies)).sum())
        if n_controls < MIN_CONTROLS:
            continue
        n_rows = 2 * min(n_cases, n_controls)   # balanced 1:1
        name = f"curatedMetagenomicData (healthy vs {label})"
        results = hard_rules.run_metadata_checks(
            n_samples=n_rows, n_features=P, task_type="classification",
            licence="odbl-1.0", source="cmd", name=label)
        if not hard_rules.all_passed(results):
            stats.record(f"cmd-{label}", "cmd", name, results, "biological")
            continue
        candidates.append(CandidateInfo(
            id=f"cmd-{label}", source="cmd", name=name, n_samples=n_rows, n_features=P,
            task_type="classification", licence="odbl-1.0", url=URL,
            metadata={"disease": label}, domain="biological"))

    # regression: predict each continuous trait from the taxa
    for col in TARGETS:
        if col not in meta.columns:
            continue
        y = meta[col].dropna()
        name = f"curatedMetagenomicData (predict {col})"
        results = hard_rules.run_metadata_checks(
            n_samples=len(y), n_features=P, task_type="regression",
            licence="odbl-1.0", source="cmd", name=col)
        if not hard_rules.all_passed(results):
            stats.record(f"cmd_reg-{col}", "cmd", name, results, "biological")
            continue
        candidates.append(CandidateInfo(
            id=f"cmd_reg-{col}", source="cmd", name=name, n_samples=len(y), n_features=P,
            task_type="regression", licence="odbl-1.0", url=URL,
            metadata={"target": col}, domain="biological"))

    return candidates[:max_candidates]


def fetch(candidate):
    X, meta = build_dataset()

    if "target" in candidate.metadata:
        # regression: every sample that has a value, predict it from the taxa
        col = candidate.metadata["target"]
        y = meta[col].dropna()
        rows = X.index.intersection(y.index)
        y = y.loc[rows].astype(float)
        task = "regression"
    else:
        # classification: this disease's cases + within-study controls, balanced 1:1
        label = candidate.metadata["disease"]
        disease = meta[DISEASE_COL].str.strip().str.lower()
        study = meta[STUDY_COL]
        studies = study[disease == label].unique()
        cases = disease.index[disease == label]
        controls = disease.index[disease.isin(HEALTHY) & study.isin(studies)]
        n = min(len(cases), len(controls))
        cases = cases.to_series().sample(n=n, random_state=SEED).index
        controls = controls.to_series().sample(n=n, random_state=SEED).index
        rows = cases.append(controls)
        y = pd.Series(["disease"] * n + ["healthy"] * n, index=rows)
        task = "classification"

    X = np.log1p(X.loc[rows]).reset_index(drop=True)
    y = y.reset_index(drop=True)
    y.name = "target"

    results = hard_rules.run_data_checks(X=X, y=y, task_type=task)
    if not hard_rules.all_passed(results):
        return None, results

    return Dataset(
        id=candidate.id,
        source="cmd",
        name=candidate.name,
        X=X,
        y=y,
        task_type=task,
        metadata={**candidate.metadata, "licence": "odbl-1.0", "url": candidate.url},
        domain="biological"), results
