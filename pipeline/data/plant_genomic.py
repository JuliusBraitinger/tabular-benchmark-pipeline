# Plant genomic-prediction panels (Azodi et al., Zenodo 4980429): 6 crops. Per crop a SNP genotype
# matrix (rows = lines, cols = markers {-1,0,1}) + a phenotype table -> one dataset per crop, SNP ->
# one quantitative trait (regression). CROP_TRAIT picks ONE trait per crop so A6 doesn't drop the
# crop's other traits (they share the same genotype X) as duplicates. Same shape as plants_loader.

import requests
from pipeline.data.base import CandidateInfo

API = "https://zenodo.org/api/records/4980429"
# one trait per crop (picked for strongest signal), keyed by the "<crop>" in "<crop>_geno.csv"
CROP_TRAIT = {"rice": "FT", "maize": "FT", "soy": "YLD", "sorghum": "YLD",
              "spruce": "HT", "switchgrass": "HT"}


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
    #TODO implemt
    return 0