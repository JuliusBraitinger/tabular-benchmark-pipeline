# MGnify loader
#Biomes: host-associated + aquatic. Human-gut studies are skipped (already covered by cmd).

import requests

from pipeline.data.base import CandidateInfo
from pipeline import stats
from pipeline.hard_rules import runner as hard_rules

API = "https://www.ebi.ac.uk/metagenomics/api/v1"
H = {"Accept": "application/json"}

BIOMES = ["root:Host-associated", "root:Environmental:Aquatic"]
MIN_SAMPLES = 150
SKIP_BIOME = "Human:Digestive"   # human gut -> covered by cmd


def studies(biome):
    #TODO implement 
    return []

def has_taxonomy(acc):
    #TODO implement
    return False


def list_candidates(max_candidates=50):
    candidates = []
    for biome in BIOMES:
        for acc, proj, n, name in studies(biome):
            if not n or n < MIN_SAMPLES or not has_taxonomy(acc):
                continue
            results = hard_rules.run_metadata_checks(
                n_samples=n, n_features=None, task_type="classification",
                licence="public-domain", source="mgnify", name=name, skip=("A4",))
            if not hard_rules.all_passed(results):
                stats.record(acc, "mgnify", name, results, "biological")
                continue
            candidates.append(CandidateInfo(
                id=acc, source="mgnify",
                name=name, n_samples=n,
                n_features=None,
                task_type="classification",
                licence="public-domain",
                url=f"https://www.ebi.ac.uk/metagenomics/studies/{acc}",
                metadata={"project": proj, "biome": biome}, domain="biological"))
            


            if len(candidates) >= max_candidates:
                return candidates
    return candidates
