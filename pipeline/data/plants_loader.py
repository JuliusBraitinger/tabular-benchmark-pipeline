# Plant RNA-seq conglomerate (Ficklin lab, Zenodo 13328785): 12 plant species -> per species
# a gene-expression matrix (rows = samples, cols = genes, read counts), plus Tissue and Age
# annotation files. Auto-downloads the record on first use and caches it. One dataset per
# species x target: tissue (classification) and age (regression),  definded in AGE_SPECIES, which species will be regression. Other 6 will be classification tasks


import os
from pathlib import Path

import requests

from pipeline.data.base import CandidateInfo

API = "https://zenodo.org/api/records/13328785"
CACHE_DIR = Path(os.environ.get("PLANTS_DIR", "/tmp/plants_cache"))
TARGETS = {"tissue": "classification", "age": "regression"}
AGE_SPECIES = {"Arabidopsis_thaliana", "Zea_mays", "Oryza_sativa",
               "Triticum_aestivum", "Solanum_tuberosum", "Hordeum_vulgare"}



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
    candidates = []
    for f in files: #loop over the files in the Zenodo record because each species has two targets
        key = f["key"]
        if "combined_output" not in key:
            continue
        name = key.split("-", 1)[1].replace("_combined_output.tsv", "") #split off the prefix and suffix for species name
        target = "age" if name in AGE_SPECIES else "tissue"
        candidates.append(CandidateInfo(
                id=f"plants-{name}-{target}",
                source="plants",
                name=f"{name.replace('_', ' ')} ({target})",
                n_samples=None,
                n_features=None,
                task_type=TARGETS[target],
                licence="cc-by-4.0",
                url=API,
                metadata={"file": key, "target": target, "url": API},
                domain="biological"))
    return candidates


def fetch(candidate):
    #TODO implement
    return None
