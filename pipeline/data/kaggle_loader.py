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
import re
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# IMPORTANT: pipeline.config must be imported BEFORE kaggle, for whater reason, otherwise the
# Kaggle API client will ignore the timeout config and hang indefinitely on slow downloads.
from pipeline import stats
from pipeline.config import (
    KAGGLE_MAX_BYTES,
    KAGGLE_MIN_BYTES,
    MAX_FEATURES,
    MIN_FEATURES,
    REQUEST_DELAY,
)
from pipeline.data.base import CandidateInfo, Dataset, infer_domain
from pipeline.hard_rules import runner as hard_rules
from pipeline.hard_rules.base import RuleResult

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("PIPELINE_CACHE", "/tmp")) / "kaggle_cache"
CROISSANT_URL = "https://www.kaggle.com/datasets/{ref}/croissant/download"
HTTP_TIMEOUT = 30
NOT_TABULAR_KEYWORDS = ["image", "audio", "pictures","video", "text", "nlp", "language", "time series", "timeseries"]

def looks_tabular(field_names: list[str], title: str) -> tuple[bool, str]:
    title_lc = title.lower()
    for kw in NOT_TABULAR_KEYWORDS:
        if kw in title_lc:
            return False, f"title contains non-tabular keyword: {kw!r}"

    if not field_names:
        return True, ""

    # Pure-numeric column names (0, 1, 2, ...) or pixel-style indicate flattened data
    pixel_like = sum(1 for n in field_names if re.match(r"^(pixel[_-]?\d+|\d+)$", n))
    if pixel_like / len(field_names) > 0.5:
        return False, f"{pixel_like}/{len(field_names)} cols look like pixel/numeric indices"

    return True, ""


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
            return None, []
    # Look for csv OR parquet; pick the largest file (assumed main table).
    data_files = (
        list(extract_dir.rglob("*.csv"))
        + list(extract_dir.rglob("*.csv.gz"))
        + list(extract_dir.rglob("*.parquet"))
    )
    if not data_files:
        logger.warning("No CSV/Parquet files found in Kaggle %s", id)
        return None, []
    data_path = max(data_files, key=lambda p: p.stat().st_size)

    try:
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, low_memory=False)
    except Exception as e:
        logger.warning("Failed to read %s for Kaggle %s: %s", data_path.name, id, e)
        return None, []

    # Step 2: detect target column
    target_col = candidate.metadata.get("target_col")
    lowered = {c.lower(): c for c in df.columns} # map lowercase column names to original names for case-insensitive matching
    resolved = lowered.get((target_col or "").lower(), df.columns[-1]) # match target_col case-insensitively, fall back to last column if not found.
    y = df[resolved]
    X = df.drop(columns=[resolved])

    # Step 3: run hard rules
    result, data_results = hard_rules.run_hard_rules(X, y, candidate)
    if result is None:
        return None, data_results

    _, task_type = result

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
        domain=candidate.domain,
    ), data_results



def fetch_croissant_fields(ref: str, auth: tuple [str,str]) -> list[str]:
    url = CROISSANT_URL.format(ref=ref)
    try:
        response = requests.get(url, auth=auth, timeout=HTTP_TIMEOUT)
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
            names.append(name)
    return names


def list_candidates(max_candidates: int = 100):
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
                # no file_type filter -- wide datasets often ship as parquet,
                # not csv, so restricting to csv excludes them at listing time.
                min_size=str(KAGGLE_MIN_BYTES),
                max_size=str(KAGGLE_MAX_BYTES),
                sort_by="hottest",
                page=page,
            )
        except Exception as e:
            logger.warning("Kaggle dataset_list page %d failed: %s", page, e)
            break

        if not results:
            logger.info("  end of results at page %d", page)
            break

        logger.info("  page %d: %d datasets returned", page, len(results))

        for result in results:
            if len(candidates) >= max_candidates:
                break

            id = str(getattr(result, "ref", ""))
            if not id:
                continue

            title = getattr(result, "title", "")
            license = getattr(result, "license_name", "")
            total_bytes = getattr(result, "total_bytes", 0)

            logger.info("  checking %s ...", id)

            field_names = fetch_croissant_fields(id, auth=(username, key))
            if not field_names:
                logger.info("    rejected: no croissant schema")
                stats.record(id, "kaggle", title, [
                    RuleResult(rule="a4", passed=False, reason="no croissant schema")
                ])
                continue

            n_features = len(field_names)
            if n_features < MIN_FEATURES:
                logger.info("    rejected: P=%d < %d", n_features, MIN_FEATURES)
                stats.record(id, "kaggle", title, [
                    RuleResult(rule="a4", passed=False, reason=f"P={n_features} < {MIN_FEATURES}")
                ])
                continue
            if n_features > MAX_FEATURES:
                logger.info("    rejected: P=%d > %d (RAM cap)", n_features, MAX_FEATURES)
                stats.record(id, "kaggle", title, [
                    RuleResult(rule="a4", passed=False, reason=f"P={n_features} > {MAX_FEATURES} (RAM cap)")
                ])
                continue

            is_tabular, reason = looks_tabular(field_names, title)
            if not is_tabular:
                logger.info("    rejected: not tabular - %s", reason)
                stats.record(id, "kaggle", title, [
                    RuleResult(rule="a2", passed=False, reason=f"non-tabular: {reason}")
                ])
                continue

            meta_results = hard_rules.run_metadata_checks(
                n_samples=None,
                n_features=n_features,
                task_type="unknown",
                licence=license,
                source="kaggle",
                name=title,
            )
            if not hard_rules.all_passed(meta_results):
                logger.info("    rejected by hard rules")
                stats.record(id, "kaggle", title, meta_results)
                continue

            logger.info("    ACCEPTED %s (P=%d, licence=%s)", id, n_features, license)
            candidates.append(CandidateInfo(
                id=id,
                source="kaggle",
                name=title[:80],
                n_samples=None,
                n_features=n_features,
                task_type="unknown",
                licence=license,
                url=f"https://www.kaggle.com/datasets/{id}",
                metadata={"total_bytes": total_bytes},
                domain=infer_domain(title),
            ))
            time.sleep(REQUEST_DELAY)

        page += 1

    logger.info("Kaggle: %d candidates after metadata filters", len(candidates))
    return candidates
            