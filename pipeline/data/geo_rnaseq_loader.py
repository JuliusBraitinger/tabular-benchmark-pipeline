# Loads RNA-seq exports made by the Nextflow pipeline (build_tabular.py).
# Each study = 3 files: <ACC>_X.parquet, <ACC>_metadata.parquet, <ACC>_info.json

import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data.base import CandidateInfo, Dataset
from pipeline import stats   
from pipeline.hard_rules import runner as hard_rules  # hard rule checks


# folder to scan for exports, under the project data dir like the other datasets
# (override with GEO_RNASEQ_DIR env var)
EXPORT_DIR = Path(os.environ.get("GEO_RNASEQ_DIR", "data/rnaseq"))
CACHE_DIR = Path("/tmp/geo_rnaseq_cache")  


# pick the target column: classification with most labelled samples, else first regression
def pick_task(tasks):
    clf = [t for t in tasks if t["type"] == "classification"] 
    if clf:
        return max(clf, key=lambda t: sum(t["classes"].values())) #picks task that has most labelled sample
    reg = [t for t in tasks if t["type"] == "regression"]
    return reg[0] if reg else None


# find every export under EXPORT_DIR, unpacking tarballs that hold an info.json
def find_studies(folder):
    studies = {}
    for info in folder.rglob("*_info.json"):
        acc = info.name[:-len("_info.json")] # extract accession from filename for finding study
        studies.setdefault(acc, info.parent)
    for archive in folder.glob("*.tar.gz"): # datasets are exported as tar.gz files 
        acc = archive.name[:-len(".tar.gz")]
        if acc in studies:
            continue
        dest = CACHE_DIR / acc
        if not (dest / f"{acc}_info.json").exists():
            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()
                if not any(n.endswith(f"{acc}_info.json") for n in names): # sanity check that this actually contains the expected files
                    continue  # not an RNA-seq export, skip
                dest.mkdir(parents=True, exist_ok=True)
                tar.extractall(dest, filter="data")
        studies[acc] = dest
    return sorted(studies.items())


def list_candidates(max_candidates=50):
    candidates = []
    for acc, folder in find_studies(EXPORT_DIR):
        with open(folder / f"{acc}_info.json") as f:
            info = json.load(f)
            task = pick_task(info.get("detected_tasks", []))
        if task is None:
            continue  # no usable target

        # metadata-level hard rules (A1/A2/A4/A5)
        results = hard_rules.run_metadata_checks(
            n_samples=info.get("n_samples"),
            n_features=info.get("n_genes"),
            task_type=task["type"],
            licence="public-domain",
            source="geo_rnaseq",
            name=acc,
        )
        stats.record(acc, "geo_rnaseq", acc, results)
        if not hard_rules.all_passed(results):
            continue

        candidates.append(CandidateInfo(
        id=acc,
        source="geo_rnaseq",
        name=acc,
        n_samples=info.get("n_samples"),
        n_features=info.get("n_genes"),
        task_type=task["type"],
        licence="public-domain",
        url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
        metadata={
            "organism": info.get("organism", "unknown"),
            "target_column": task["column"],
            "folder": str(folder),
        },
        domain="biological",
        ))
        if len(candidates) >= max_candidates:
            break

    return candidates



def fetch(candidate):
    id = candidate.id
    folder = Path(candidate.metadata["folder"])
    target = candidate.metadata["target_column"]

    X = pd.read_parquet(folder / f"{id}_X.parquet")
    meta = pd.read_parquet(folder / f"{id}_metadata.parquet")

    # target lives in the metadata table; keep only labelled samples
    y = meta[target].dropna()
    shared = X.index.intersection(y.index)
    if len(shared) > 2:
        return None
    X = X.loc[shared].copy()
    y = y.loc[shared]
    y.name = "target"

    # raw salmon counts -> log1p so soft rules see a sane value range (TEST)
    X = np.log1p(X.clip(lower=0))

    results = hard_rules.run_data_checks(
        X=X,
        y=y,
        task_type=candidate.task_type,
    )
    stats.record(id, "geo_rnaseq", id, results)
    if not hard_rules.all_passed(results):
        return None
    
    return Dataset(
        id=f"GEO-{id}",
        source="geo_rnaseq",
        name=candidate.name,
        X=X,
        y=y,
        task_type=candidate.task_type,
        metadata={**candidate.metadata, "licence": "public-domain", "transform": "log1p"},
        domain="biological",
    )


