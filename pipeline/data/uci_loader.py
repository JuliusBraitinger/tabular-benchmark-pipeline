
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from ucimlrepo import fetch_ucirepo

from pipeline import stats
from pipeline.config import MAX_FEATURES, MIN_FEATURES, REQUEST_DELAY
from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules
from pipeline.hard_rules.base import RuleResult

logger = logging.getLogger(__name__)

UCI_LIST_URL = "https://archive.ics.uci.edu/api/datasets/list"
UCI_META_URL = "https://archive.ics.uci.edu/api/dataset"
HTTP_TIMEOUT = 30

# parquet cache for downloaded X/y, matching the other loaders
CACHE_DIR = Path("data/cache/uci")


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
        logger.info("Downloading UCI dataset id=%d from ucimlrepo...", id)
        try:
            results = fetch_ucirepo(id)
        except Exception as e:
            logger.info("UCI dataset fetch failed for id=%d: %s", id, e)
            return None
        
        X = results.data.features
        y = results.data.targets

        if X is None or y is None:
            logger.info("UCI dataset id=%d has no data, skipping", id)
            return None
        
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]

        folder.mkdir(parents=True, exist_ok=True)
        X.to_parquet(x_file)
        y.to_frame().to_parquet(y_file)
        logger.info("UCI dataset id=%d downloaded and cached", id)

    data_results = hard_rules.run_data_checks(X, y, candidate.task_type)
    stats.record(candidate.id, "uci", candidate.name, data_results)
    failed = hard_rules.failed_rules(data_results)
    if failed:
        logger.info("UCI dataset id=%d failed hard rules: %s", id, ", ".join(f"{r.rule}: {r.reason}" for r in failed))
        return None
    task_type = hard_rules.inferred_task_type(data_results)
    logger.info("UCI dataset id=%d inferred task type: %s", id, task_type)

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
        }
    )


def list_candidates(max_per_source: int = 50):
    return None