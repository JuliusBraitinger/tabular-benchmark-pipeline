# Human gut metagenomics loader.
#
# What gets loaded (one dataset PER disease):
#   rows     = gut samples: all healthy people + one disease's patients
#   features = ~34k microbe markers, each 0 or 1 = is that microbe present
#   target   = healthy vs <disease>  ->  binary classification
# Source: Pasolli's MetAML marker table, downloaded automatically.

import bz2
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pipeline.data.base import CandidateInfo, Dataset
from pipeline import stats
from pipeline.hard_rules import runner as hard_rules

logger = logging.getLogger(__name__)

MARKER_URL = "https://raw.githubusercontent.com/segatalab/metaml/master/data/marker_presence.txt.bz2"
CACHE_DIR = Path(os.environ.get("PIPELINE_CACHE", "/tmp")) / "metagenomics_cache"
MIN_PREVALENCE = 0.10                   # drop markers present in <10% of samples
MIN_CASES = 50                          
HEALTHY = {"n", "nd", "n_relative"}     # disease codes counted as healthy


def build_dataset():
    # download the marker file once (cached), parse it, keep only prevalent
    # markers. returns X (samples x markers) + the disease label per sample.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = CACHE_DIR / "marker_presence.txt.bz2"
    if not raw.exists():
        logger.info("downloading MetAML marker matrix (~23MB)...")
        resp = requests.get(MARKER_URL, timeout=180)
        resp.raise_for_status()
        raw.write_bytes(resp.content)

    disease, markers, names, n_samples = None, [], [], None
    with bz2.open(raw, "rt") as f:
        for line in f:
            name, _, rest = line.partition("\t")
            values = rest.rstrip("\n").split("\t")
            if n_samples is None:
                n_samples = len(values)
            if name == "disease":
                disease = values
                continue
            ones = values.count("1")
            if ones + values.count("0") != n_samples:    # a metadata (text) row, not a marker
                continue
            if ones >= MIN_PREVALENCE * n_samples:        # keep only prevalent markers
                markers.append(np.array(values, dtype=np.int8))
                names.append(name)

    X = pd.DataFrame(np.array(markers).T, columns=names)  # rows = samples, cols = markers
    disease = pd.Series([d.strip() for d in disease], name="disease")  # one label per sample
    return X, disease


def list_candidates(max_candidates=50):
    X, disease = build_dataset()

    # one candidate per disease that has enough patients (skip the healthy labels)
    candidates = []
    for label, n_cases in disease.value_counts().items():
        if label in HEALTHY or n_cases < MIN_CASES:
            continue
        n_rows = int((disease.isin(HEALTHY) | (disease == label)).sum())  # healthy + this disease

        results = hard_rules.run_metadata_checks(
            n_samples=n_rows, n_features=X.shape[1], task_type="classification",
            licence="cc-by-4.0", source="metagenomics", name=label,
        )
        if not hard_rules.all_passed(results):
            stats.record(f"gut-{label}", "metagenomics",
                         f"Human gut markers (healthy vs {label})", results, "biological")
            continue

        candidates.append(CandidateInfo(
            id=f"gut-{label}",
            source="metagenomics",
            name=f"Human gut markers (healthy vs {label})",
            n_samples=n_rows,
            n_features=X.shape[1],
            task_type="classification",
            licence="cc-by-4.0",
            url="https://github.com/segatalab/metaml",
            metadata={"disease": label},          # fetch needs to know which disease
            domain="biological",
        ))
        if len(candidates) >= max_candidates:
            break

    return candidates


def fetch(candidate):
    X, disease = build_dataset()
    target = candidate.metadata["disease"]

    # keep the healthy samples + this disease's patients, then label them
    keep = disease.isin(HEALTHY) | (disease == target)
    X = X[keep].reset_index(drop=True)
    y = pd.Series(["healthy" if d in HEALTHY else "disease" for d in disease[keep]], name="target")

    results = hard_rules.run_data_checks(X=X, y=y, task_type="classification")
    if not hard_rules.all_passed(results):
        return None, results

    return Dataset(
        id=candidate.id,
        source="metagenomics",
        name=candidate.name,
        X=X,
        y=y,
        task_type="classification",
        metadata={"licence": "cc-by-4.0", "disease": target, "url": candidate.url},
        domain="biological",
    ), results
