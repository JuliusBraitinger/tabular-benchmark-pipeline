# ChEMBL loader: builds drug-target datasets for ML.
# idea from claude code for these datasets. currently in working mode. not sure yet if it works
#
# The idea: pick a human protein (a "target") and look at all the molecules
# that have been tested against it. Each molecule has a chemical structure (a
# SMILES string) and a pChEMBL value saying how strongly it binds the protein
# (its potency). The structure becomes a 2048-bit fingerprint that acts as the
# features, and the potency is the value to predict.
#   X = one row per molecule, 2048 fingerprint columns
#   y = how strongly it binds (a number, so it's regression)
# 2048 columns are used because the pipeline only keeps datasets with >= 1000
# features (the old version used ~10 properties and always got rejected).

import logging
import time

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from pipeline import config
from pipeline.config import CHEMBL_BASE_URL, MIN_ROWS
from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules

logger = logging.getLogger(__name__)

# each molecule becomes a 2048-bit ECFP4 fingerprint (Morgan, radius 2)
FP_N_BITS = 2048
fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_N_BITS)

# only bother with a target that has at least this many measurements
MIN_BIOACTIVITIES = 2000


def list_candidates(max_candidates=50):
    candidates = []
    offset = 0

    while len(candidates) < max_candidates:
        # grab a page of human single-protein targets
        url = (
            f"{CHEMBL_BASE_URL}/target.json"
            f"?target_type=SINGLE+PROTEIN&organism=Homo+sapiens"
            f"&limit=50&offset={offset}"
        )
        targets = requests.get(url, timeout=30).json().get("targets", [])
        if not targets:
            break  # ran out of targets

        for target in targets:
            target_id = target.get("target_chembl_id")
            target_name = target.get("pref_name") or "Unknown"
            if not target_id:
                continue

            # how many measurements does this target have? just read the count
            count_url = (
                f"{CHEMBL_BASE_URL}/activity.json"
                f"?target_chembl_id={target_id}&pchembl_value__isnull=false&limit=1"
            )
            count = requests.get(count_url, timeout=30).json()["page_meta"]["total_count"]
            if count < MIN_BIOACTIVITIES:
                continue

            logger.info("%s (%s): %d measurements", target_name, target_id, count)
            candidates.append(CandidateInfo(
                id=target_id,
                source="chembl",
                name=target_name,
                n_samples=count,
                n_features=FP_N_BITS,
                task_type="regression",
                licence="CC BY-SA 3.0",
                url=f"https://www.ebi.ac.uk/chembl/target_report_card/{target_id}/",
                metadata={},
                domain="chemical",
            ))
            time.sleep(config.REQUEST_DELAY)

            if len(candidates) >= max_candidates:
                break

        offset += 50

    return candidates


def fetch(candidate):
    target_id = candidate.id

    # download every measurement for this target, one page at a time.
    #    each record already has the molecule's SMILES and its potency,
    #    so there's no need for a separate request per molecule.
    records = []
    url = (
        f"{CHEMBL_BASE_URL}/activity.json"
        f"?target_chembl_id={target_id}&pchembl_value__isnull=false&limit=1000"
    )
    while url:
        data = requests.get(url, timeout=30).json()
        for activity in data["activities"]:
            smiles = activity.get("canonical_smiles")
            value = activity.get("pchembl_value")
            if smiles and value is not None:
                records.append({"smiles": smiles, "pchembl": float(value)})

        next_page = data["page_meta"]["next"]  # ChEMBL gives the next page, or null when done
        url = f"https://www.ebi.ac.uk{next_page}"  if next_page else None
        time.sleep(config.REQUEST_DELAY)

    if not records:
        logger.warning("no usable data")
        return None, []

    #  a molecule can be measured several times, so keep one row per molecule
    #    and use the median potency.
    table = pd.DataFrame(records).groupby("smiles", as_index=False)["pchembl"].median()

    # turn each molecule's SMILES into a 2048-bit fingerprint (the features)
    rows, potency = [], []
    for smiles, value in zip(table["smiles"], table["pchembl"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue  # couldn't read this molecule's structure
        fingerprint = fp_generator.GetFingerprint(mol)
        bits = np.zeros(FP_N_BITS, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, bits)
        rows.append(bits)
        potency.append(value)

    X = pd.DataFrame(np.array(rows), columns=[f"fp_{i}" for i in range(FP_N_BITS)])
    y = pd.Series(potency, name="pchembl_value")
    logger.info("%s: %d molecules x %d features", target_id, len(X), X.shape[1])

    if len(X) < MIN_ROWS:
        return None, []

    result, data_results = hard_rules.run_hard_rules(X, y, candidate)
    if result is None:
        return None, data_results

    _, task_type = result
    return Dataset(
        X=X,
        y=y,
        task_type=task_type,
        id=candidate.id,
        source="chembl",
        name=candidate.name,
        metadata=candidate.metadata,
        domain="chemical",
    ), data_results
