
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
