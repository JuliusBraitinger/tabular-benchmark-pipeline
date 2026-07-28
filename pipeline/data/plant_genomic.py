# Plant genomic-prediction panels (Azodi et al., Zenodo 4980429): 6 crops. Per crop a SNP genotype
# matrix (rows = lines, cols = markers {-1,0,1}) + a phenotype table -> one dataset per crop, SNP ->
# one quantitative trait (regression). CROP_TRAIT picks ONE trait per crop so A6 doesn't drop the
# crop's other traits (they share the same genotype X) as duplicates. Same shape as plants_loader.

import requests
import os
import pandas as pd
from pathlib import Path
from pipeline.hard_rules import runner as hard_rules
from pipeline.data.base import CandidateInfo, Dataset
API = "https://zenodo.org/api/records/4980429"
# one trait per crop (picked for strongest signal), keyed by the "<crop>" in "<crop>_geno.csv"
CROP_TRAIT = {"rice": "FT", "maize": "FT", "soy": "YLD", "sorghum": "YLD",
              "spruce": "HT", "switchgrass": "HT"}

FILE = API + "/files/{}/content"                            # per-file download URL
CACHE_DIR = Path(os.environ.get("GP_DIR", "/tmp/gp_cache"))


def download(fname):
    # download one file from the record (cached), streamed so big geno files stay off-RAM
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / fname
    if not dest.exists():
        with requests.get(FILE.format(fname), stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as out:
                for chunk in r.iter_content(1 << 20):       # 1 MB at a time
                    out.write(chunk)
    return dest




def list_candidates(max_candidates=50):
    # one candidate per crop: the crop's SNP matrix -> its chosen trait (regression)
    files = requests.get(API, timeout=60).json()["files"]
    candidates = []
    for f in files:                           # each crop has a "<crop>_geno.csv" in the record
        key = f["key"]
        if "_geno" not in key:
            continue
        crop = key.split("_geno")[0]          # "rice_geno.csv" -> "rice"
        trait = CROP_TRAIT[crop]
        candidates.append(CandidateInfo(
            id=f"gp-{crop}-{trait}",
            source="gp",
            name=f"{crop} genomic prediction ({trait})",
            n_samples=None,
            n_features=None,
            task_type="regression",
            licence="cc0",
            url=API,
            metadata={"crop": crop, "trait": trait, "url": API},
            domain="biological"))
    return candidates


def fetch(candidate):
    crop = candidate.metadata["crop"]
    trait = candidate.metadata["trait"]
    # X = SNP genotype matrix (rows = lines, cols = markers); y = the chosen trait
    X = pd.read_csv(download(f"{crop}_geno.csv"), index_col=0)
    y = pd.read_csv(download(f"{crop}_pheno.csv"), index_col=0)[trait]

    # keep only lines present in both, with a non-missing trait value
    rows = X.index.intersection(y.dropna().index)
    X = X.loc[rows].reset_index(drop=True)
    y = y.loc[rows].reset_index(drop=True).rename("target")

    results = hard_rules.run_data_checks(X=X, y=y, task_type=candidate.task_type)
    if not hard_rules.all_passed(results):
        return None, results

    return Dataset(
        id=candidate.id,
        source="gp",
        name=candidate.name,
        X=X,
        y=y,
        task_type=candidate.task_type,
        metadata={**candidate.metadata, "licence": candidate.licence, "url": candidate.url},
        domain="biological"), results
