"""Kaggle Datasets loader.

Flow:
  list_candidates() — paginate the Kaggle API, fetch the Croissant export
    per dataset to learn the column count + names, run metadata hard rules.
  fetch() — download the zip, unzip, load the largest CSV, run data hard rules.

Target column detection is heuristic: Kaggle doesn't flag targets in any of
its metadata formats, so we match common conventions (target / label / etc.)
and fall back to the last column.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from kaggle.api.kaggle_api_extended import KaggleApi
from pipeline import stats
from pipeline.config import (
    KAGGLE_MAX_BYTES,
    KAGGLE_MIN_BYTES,
    KAGGLE_TARGET_NAMES,
    KAGGLE_TARGET_SUFFIXES,
    MAX_FEATURES,
    MIN_FEATURES,
    REQUEST_DELAY,
)
from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules
from pipeline.hard_rules.base import RuleResult

logger = logging.getLogger(__name__)

CACHE_DIR = Path("/tmp/kaggle_cache")
CROISSANT_URL = "https://www.kaggle.com/datasets/{ref}/croissant/download"
HTTP_TIMEOUT = 30

def fetch(candidate: CandidateInfo):
    logger.info("Fetching dataset %s", candidate.id)
    extract_path = CACHE_DIR / candidate.id

    if not extract_path.exists():
        api = KaggleApi()
        try:
            api.authenticate()
            extract_path.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(candidate.id, path=str(extract_path), unzip=True)
        except Exception as e:
            logger.error("Failed to download dataset %s: %s", candidate.id, e)
            raise 