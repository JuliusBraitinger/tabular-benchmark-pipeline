# Human gut metagenomics loader.
#
# What gets loaded (one dataset):
#   rows     = people / gut samples (~3600)
#   features = ~34k microbe markers, each 0 or 1 = is that microbe present
#   target   = healthy vs disease  ->  binary classification
# Source: Pasolli's MetAML marker table, downloaded automatically.

import bz2
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules

logger = logging.getLogger(__name__)

MARKER_URL = "https://raw.githubusercontent.com/segatalab/metaml/master/data/marker_presence.txt.bz2"
CACHE_DIR = Path(os.environ.get("PIPELINE_CACHE", "/tmp")) / "metagenomics_cache"
MIN_PREVALENCE = 0.10                   # drop markers present in <10% of samples
HEALTHY = {"n", "nd", "n_relative"}     # disease codes counted as healthy
DATASET_ID = "gut-markers-disease"


def build_dataset():
    # download the marker file once (cached), parse it, keep only prevalent
    # markers, and build X (samples x markers) + y (healthy vs disease).
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
    y = pd.Series(["healthy" if d.strip() in HEALTHY else "disease" for d in disease], name="target")
    return X, y


def list_candidates(max_candidates=50): #for now just run metadata checks because not target variable
    # download the marker matrix once, then reuse the cached copy
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = CACHE_DIR / "marker_presence.txt.bz2"
    if not raw.exists():
        logger.info("downloading MetAML marker matrix (~23MB)...")
        resp = requests.get(MARKER_URL, timeout=180)
        resp.raise_for_status()
        raw.write_bytes(resp.content)

    # parse: rows = features (+ a few metadata rows), columns = samples
    disease, names, rows, n = None, [], [], None
    with bz2.open(raw, "rt") as f: #open the compressed marker matrix and read it line by line
        for line in f:
            label, _, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            if n is None:
                n = rest.count("\t") + 1            # number of samples (#tabs + 1)
                min_present = int(MIN_PREVALENCE * n)
            if label == "disease":
                disease = rest.split("\t")
                continue
            vals = "\t" + rest
            ones, zeros = vals.count("\t1"), vals.count("\t0")
            if ones + zeros != n:                   # skip non-binary (metadata) rows
                continue
            if ones >= min_present:                 # keep only prevalent markers
                rows.append(np.array(rest.split("\t"), dtype=np.int8))
                names.append(label)

    idx = [f"s{i}" for i in range(n)]
    X = pd.DataFrame(np.array(rows).T, index=idx, columns=names)
    n_samples, n_features = X.shape

    results = hard_rules.run_metadata_checks(
        n_samples=n_samples, n_features=n_features, task_type="classification",
        licence="cc-by-4.0", source="metagenomics", name=DATASET_ID,
    )
    if not hard_rules.all_passed(results):
        logger.info("metagenomics rejected: %s",
                    [r.reason for r in hard_rules.failed_rules(results)])
        return []

    return [CandidateInfo(
        id=DATASET_ID,
        source="metagenomics",
        name="Human gut markers (healthy vs disease)",
        n_samples=n_samples,
        n_features=n_features,
        task_type="classification",
        licence="cc-by-4.0",
        url="https://github.com/segatalab/metaml",
        metadata={},
        domain="biological",
    )]


def fetch(candidate):
    # download the marker file once (cached)
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
    y = pd.Series(["healthy" if d.strip() in HEALTHY else "disease" for d in disease], name="target")

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
        metadata={"licence": "cc-by-4.0"},
        domain="biological",
    ), results
