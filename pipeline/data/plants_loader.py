# Plant RNA-seq conglomerate (Ficklin lab, Zenodo 13328785): 12 plant species -> per species
# a gene-expression matrix (rows = samples, cols = genes, read counts), plus Tissue and Age
# annotation files. Auto-downloads the record on first use and caches it. One dataset per
# species x target: tissue (classification) and age (regression), predicted from expression.

import os
from pathlib import Path

import requests

from pipeline.data.base import CandidateInfo

API = "https://zenodo.org/api/records/13328785"
CACHE_DIR = Path(os.environ.get("PLANTS_DIR", "/tmp/plants_cache"))
TARGETS = {"tissue": "classification", "age": "regression"}


def download():
    # fetch the record's file list via the Zenodo API
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for f in requests.get(API, timeout=60).json()["files"]:
        dest = CACHE_DIR / f["key"]
        if dest.exists():
            continue
        with requests.get(f["links"]["self"], stream=True, timeout=180) as r: 
            r.raise_for_status()
            with open(dest, "wb") as out:
                for chunk in r.iter_content(1 << 20): # only keep 1MB in memory at a time 
                    out.write(chunk)
    return CACHE_DIR


def list_candidates(max_candidates=50):
    # one candidate per (species x target): tissue -> classification, age -> regression.
    # species matrices are the "combined_output.tsv" 
    files = requests.get(API, timeout=60).json()["files"]
    species = [f["key"] for f in files if "combined_output" in f["key"]][:max_candidates] #get species files only, limit to max_candidates
    candidates = []
    for key in species:
        name = key.split("-", 1)[1].replace("_combined_output.tsv", "")  # need to remove the prefix and suffix to get the species name
        for target, task in TARGETS.items():
            candidates.append(CandidateInfo(
                id=f"plants-{name}-{target}",
                source="plants",
                name=f"{name.replace('_', ' ')} ({target})",
                n_samples=None,
                n_features=None,
                task_type=task,
                licence="cc-by-4.0",
                url=API,
                metadata={"file": key, "target": target, "url": API},
                domain="biological"))
    return candidates


def fetch(candidate):
    #TODO implement
    return None
