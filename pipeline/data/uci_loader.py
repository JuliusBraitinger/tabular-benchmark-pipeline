
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from ucimlrepo import fetch_ucirepo

from pipeline import stats
from pipeline.config import MAX_FEATURES, MIN_FEATURES, REQUEST_DELAY, MIN_ROWS
from pipeline.data.base import CandidateInfo, Dataset, infer_domain
from pipeline.hard_rules import runner as hard_rules
from pipeline.hard_rules.base import RuleResult


logger = logging.getLogger(__name__)

UCI_LIST_URL = "https://archive.ics.uci.edu/api/datasets/list"
UCI_META_URL = "https://archive.ics.uci.edu/api/dataset"
HTTP_TIMEOUT = 30

# parquet cache for downloaded X/y, matching the other loaders
CACHE_DIR = Path(os.environ.get("PIPELINE_DATA", "data")) / "cache" / "uci"


def fetch_metadata(uci_id: int) -> dict | None:
#get metadata for given UCI dataset
    try:
        resp = requests.get(UCI_META_URL, params={"id": uci_id}, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("data") or None
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.debug("UCI metadata fetch failed for id=%d: %s", uci_id, e)
        return None


def fetch(candidate:CandidateInfo):
    # download the dataset for the given candidate, return a Dataset object
    id = candidate.id
    folder = CACHE_DIR / str(id)
    x_file = folder / "X.parquet"
    y_file = folder / "y.parquet"

    if x_file.exists() and y_file.exists():
        X = pd.read_parquet(x_file)
        y = pd.read_parquet(y_file).iloc[:, 0] #parquet doesn't support Series, so read as DataFrame and take the first column
    else:  
        # option A: change format
        logger.info("Downloading UCI dataset id=%s from ucimlrepo...", id)
        try:
            results = fetch_ucirepo(id=int(id))
        except Exception as e:
            logger.info("UCI dataset fetch failed for id=%s: %s", id, e)
            return None, []
        
        X = results.data.features
        y = results.data.targets

        if X is None or y is None:
            logger.info("UCI dataset id=%s has no data, skipping", id)
            return None, []
        
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]

        folder.mkdir(parents=True, exist_ok=True)
        X.to_parquet(x_file)
        y.to_frame().to_parquet(y_file)
        logger.info("UCI dataset id=%s downloaded and cached", id)

    result, data_results = hard_rules.run_hard_rules(X, y, candidate)
    if result is None:
        return None, []

    _, task_type = result

    return Dataset(
        id=candidate.id,
        source = "uci",
        name=candidate.name,
        X=X,
        y=y,
        task_type=task_type,
        metadata = {
            **candidate.metadata,
            "license" : candidate.licence,
            "url" : candidate.url,
        },
        domain=candidate.domain,
    ), data_results


def list_candidates(max_candidates: int = 100):
    logger.info("Listing UCI datasets...")

    # step 1: get the full list of (id, name) pairs
    resp = requests.get(UCI_LIST_URL, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    listing = resp.json()["data"]
    logger.info("UCI: %d datasets in catalog", len(listing))

    candidates = []
    for item in listing:
        if len(candidates) >= max_candidates:
            break

        uci_id = item["id"]
        meta = fetch_metadata(uci_id)
        time.sleep(REQUEST_DELAY)
        if meta is None:
            continue
        
        if not meta.get("data_url"): #if no data url, can't fetch the dataset, so skip
            continue

        # step 2: pull the fields
        n = meta.get("num_instances") or 0 #if unknown or missing set to 0 so condition holds 
        p = meta.get("num_features") or 0
        name = meta.get("name", "")
        tasks = meta.get("tasks") or []
        task_type = tasks[0].lower() if tasks else "unknown"
        licence = meta.get("license") or "CC By 4.0"

        if n < MIN_ROWS or p < MIN_FEATURES or p > MAX_FEATURES:
            continue

        # step 4: metadata hard rules (A1, A2, A4, A5)
        results = hard_rules.run_metadata_checks(
            n_samples=n,
            n_features=p,
            task_type=task_type,
            licence=licence,
            source="uci",
            name=name,
            metadata={},
        )

        if hard_rules.failed_rules(results):
            continue

        # step 5: passed,
        candidates.append(CandidateInfo(
            id=str(uci_id),
            source="uci",
            name=name,
            n_samples=n,
            n_features=p,
            task_type=task_type,
            licence=licence,
            url=f"https://archive.ics.uci.edu/dataset/{uci_id}",
            metadata={},
            domain=infer_domain(name, meta.get("characteristics") or []),
        ))

    logger.info("UCI: %d candidates", len(candidates))
    return candidates
