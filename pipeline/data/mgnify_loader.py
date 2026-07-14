# MGnify loader
#Biomes: host-associated + aquatic. Human-gut studies are skipped (already covered by cmd).

import io
from collections import Counter

import numpy as np
import pandas as pd
import requests

from pipeline.data.base import CandidateInfo, Dataset
from pipeline import stats
from pipeline.hard_rules import runner as hard_rules

API = "https://www.ebi.ac.uk/metagenomics/api/v1"
H = {"Accept": "application/json"}

MISSING = (None, "")


BIOMES = ["root:Host-associated", "root:Environmental:Aquatic"]
MIN_SAMPLES = 150
MIN_CLASS = 30                   # a target class needs at least this many samples
MAX_CLASSES = 10                 # skip near-unique fields (coordinates, ids)


def studies(biome):
    # fetch all studies for a given biome, yield (accession, project, n_samples, name)
    url = f"{API}/biomes/{biome}/studies"
    while url:
        resp = requests.get(url, headers=H, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for study in data.get("data", []):
            a = study["attributes"]
            acc = a["accession"]
            proj = a.get("secondary-accession")
            n = a.get("samples-count")
            name = a["study-name"]
            yield acc, proj, n, name
        url = data.get("links", {}).get("next")

def has_functional(acc):
    # a study qualifies only if it has an aggregated InterPro (IPR) functional
    # abundance table -> that is the wide (10k+ feature) matrix use as X
    url = f"{API}/studies/{acc}/downloads"
    resp = requests.get(url, headers=H, timeout=30)
    resp.raise_for_status()
    for d in resp.json().get("data", []):
        if "IPR_abundances" in d["attributes"]["alias"]:
            return True
    return False

def list_candidates(max_candidates=50):
    candidates = []
    for biome in BIOMES:
        for acc, proj, n, name in studies(biome):
            if not n or n < MIN_SAMPLES or not has_functional(acc):
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




def fetch(candidate):
   
    # 1. find the InterPro (IPR) functional table, download it, and transpose to samples x features
    downloads = requests.get(f"{API}/studies/{candidate.id}/downloads", headers=H, timeout=30).json()
    table_url = None
    for i in downloads["data"]:
        if "IPR_abundances" in i["attributes"]["alias"]:
            table_url = i["links"]["self"]
            break
    if table_url is None:
        return None, []

    text = requests.get(table_url, timeout=90).text
    table = pd.read_csv(io.StringIO(text), sep="\t")
    table = table.set_index(table.columns[0])                    # move sample id out and make it row labels 
    table = table.drop(columns="description", errors="ignore")   # 2nd column is a text description
    table = table.transpose()                                            # transpose to samples x features

    # target data lives in sample metadata -> two differenet id systems
    #
    sample_of = {}
    url = f"{API}/studies/{candidate.id}/analyses?page_size=100"
    while url:
        page = requests.get(url, headers=H, timeout=60).json() #look at trhough /analyses endpoint. Each analysis links sample to run/assembly it was derived from
        for analysis in page["data"]: 
            rel = analysis["relationships"]
            sample = rel["sample"]["data"]
            if sample is None:
                continue
            for kind in ("run", "assembly"):
                ref = rel[kind]["data"]
                if ref is not None:
                    sample_of[ref["id"]] = sample["id"]
        url = page["links"]["next"]

    # 3. read the metadata of every sample (field name -> value)
    meta_of = {}
    url = f"{API}/studies/{candidate.id}/samples?page_size=100" 
    while url:
        page = requests.get(url, headers=H, timeout=60).json()
        for s in page["data"]: #for each sample, get the metadata fields and values
            attrs = s["attributes"]
            fields = {}
            for m in attrs.get("sample-metadata", []): #all metadata key/value pairs
                fields[m["key"]] = m["value"]
            meta_of[attrs["accession"]] = fields
        url = page["links"]["next"]

    all_fields = set() #union of all metadata fields across all samples
    for fields in meta_of.values():
        all_fields.update(fields)

    # 4. MGnify has no ready-made label. Pick the first metadata field that splits the
    #    samples into 2..MAX_CLASSES groups that are each big enough.
    best_field = None
    best_labels = None
    for field in sorted(all_fields):
        # give each sample its value for this field
        labels = {}
        for sample_id in table.index:
            sample = sample_of.get(sample_id)
            value = meta_of.get(sample, {}).get(field)
            if value not in MISSING:
                labels[sample_id] = str(value).strip().lower()

        # keep only the groups that have enough samples
        big_groups = [g for g, n in Counter(labels.values()).items() if n >= MIN_CLASS]
        if 2 <= len(big_groups) <= MAX_CLASSES:
            best_field = field
            best_labels = {s: v for s, v in labels.items() if v in big_groups}
            break

    if best_field is None:
        return None, []

    # 5. build X and y, then run the data hard rules
    samples = list(best_labels.keys())
    X = np.log1p(table.loc[samples]).reset_index(drop=True)
    y = pd.Series([best_labels[s] for s in samples], name="target")

    results = hard_rules.run_data_checks(X=X, y=y, task_type="classification", skip=("A4",))
    if not hard_rules.all_passed(results):
        return None, results

    return Dataset(
                id=candidate.id,
                source="mgnify", 
                name=candidate.name,
                X=X,
                y=y,
                task_type="classification",
                metadata={**candidate.metadata,
                "target_field": best_field},
                domain="biological"), results

