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
import os
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
    id = candidate.id
    logger.info("Downloading Kaggle %s...", id)

    extract_dir = CACHE_DIR / id

    # Step 1: download + unzip (skip if cached)
    if not extract_dir.exists():
        api = KaggleApi()
        try:
            api.authenticate()
            extract_dir.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(id, path=str(extract_dir), unzip=True)
        except Exception as e:
            logger.warning("Failed to download Kaggle %s: %s", id, e)
            return None
    csv_files = list(extract_dir.rglob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in Kaggle %s", id)
        return None
    csv_path = max(csv_files, key=lambda p: p.stat().st_size) # pick the largest CSV file, assuming it's the main one
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        logger.warning("Failed to read CSV from Kaggle %s: %s", id, e)
        return None

    # Step 2: detect target column
    target_col = candidate.metadata.get("target_col")
    lowered = {c.lower(): c for c in df.columns} # map lowercase column names to original names for case-insensitive matching
    resolved = lowered.get((target_col or "").lower(), df.columns[-1]) # match target_col case-insensitively, fall back to last column if not found.
    y = df[resolved]
    X = df.drop(columns=[resolved])

    # Step 3: run hard rules
    data_result = hard_rules.run_data_checks(X, y, task_type=candidate.task_type)
    stats.record(id, "kaggle", candidate.name, data_result)
    failed = hard_rules.failed_rules(data_result)
    if failed:
        logger.info("Kaggle %s failed hard rules: %s", id, failed)
        return None

    # A1 can refine task_type from the actual target (binary -> classification, etc.)
    task_type = hard_rules.inferred_task_type(data_result) or candidate.task_type

    logger.info("Kaggle %s passed hard rules, loading dataset...", id)

    return Dataset(
        id = "kaggle:" + id,
        source = "kaggle",
        name = candidate.name,
        X=X,
        y=y,
        task_type=task_type,
        metadata={
            **candidate.metadata,
            "licence": candidate.licence,
            "url": f"https://www.kaggle.com/datasets/{id}",
        },
    )



def fetch_croissant_fields(ref: str, auth: tuple [str,str]) -> list[str]:
    url = CROISSANT_URL.format(ref=ref)
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning("Failed to fetch Croissant for %s: %s", ref, e)
        return []
    
    record_sets = data.get("recordSet") or []
    if not record_sets:
        return []
    
    best = max(record_sets, key=lambda rs: len(rs.get("field") or [])) #given recodSet, return number of fields it hat and pick the one with the most fields as the best guess for the main dataset
    fields = best.get("field", [])
    names = []
    for f in fields:
        name = f.get("name")
        if name:
            names.appends(name)
    return names


def list_candidates(max_candidates: int = 50):
    logger.info("Listing Kaggle datasets...")

    try:
        username = os.environ["KAGGLE_USERNAME"]
        key = os.environ["KAGGLE_KEY"]
    except KeyError:
        logger.warning("Kaggle API credentials not found in environment variables, skipping Kaggle datasets.")
        return []
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        logger.warning("Failed to authenticate with Kaggle API: %s", e)
        return []
    
    candidates: list[CandidateInfo] = []
    page = 1

    while len(candidates) < max_candidates:
        try:
            results = api.dataset_list(
                file_type="csv",
                min_size=KAGGLE_MIN_BYTES,
                max_size=KAGGLE_MAX_BYTES,
                sort_by="hottest",
                page=page,
            )
        except Exception as e:
            logger.warning("Kaggle dataset_list page %d failed: %s", page, e)
            break

        for result in results:
            if len(candidates) >= max_candidates:
                break
            
            id = str(getattr(result, "ref", ""))
            if not id:
                continue

            title = getattr(result, "title", "")
            license = getattr(result, "license_name", "")
            total_bytes = getattr(result, "total_bytes", 0)

            # Fetch Croissant export to get column names for hard rules (skip if cached)
            field_names = fetch_croissant_fields(id, auth=(username, key))
            if not field_names:
                fail = RuleResult(rule="pre-filter", passed=False, reason="no croissant schema")
                stats.record(id, "kaggle", title, [fail])
                continue

            n_features = len(field_names)
            if n_features < MIN_FEATURES:
                failed = RuleResult(rule="pre-filter", passed=False,
                                  reason=f"P={n_features} < {MIN_FEATURES}")
                stats.record(id, "kaggle", title, [fail])
                continue
            if n_features > MAX_FEATURES:
                failed = RuleResult(rule="pre-filter", passed=False,
                                  reason=f"P={n_features} > {MAX_FEATURES} (RAM cap)")
                stats.record(id, "kaggle", title, [fail])
                continue
            results = hard_rules.run_metadata_checks(
                n_samples=None,
                n_features=n_features,
                task_type="unknown",
                licence=license,
                source="kaggle",
                name=title,
            )
            stats.record(id, "kaggle", title, results)
            if not hard_rules.all_passed(results):
                continue

            candidates.append(CandidateInfo(
                id=id,
                source="kaggle",
                name=title[:80],
                n_samples=None,
                n_features=n_features,
                task_type="unknown",
                licence=licence,
                url=f"https://www.kaggle.com/datasets/{id}",
            ))
            time.sleep(REQUEST_DELAY)

        page += 1 #paginate to the next page of results s

    logger.info("Kaggle: %d candidates after metadata filters", len(candidates))
    return candidates
            