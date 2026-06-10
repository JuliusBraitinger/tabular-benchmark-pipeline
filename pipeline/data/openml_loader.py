"""OpenML data loader.

Implements the dataset loader for OpenML, a public repository of ML datasets.
Unlike GEO/TCGA, OpenML datasets already come with a predefined target, so
we don't have to build the prediction task ourselves.

Flow: query API, filter by size, run hard rules, download.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import openml
import pandas as pd

from pipeline.config import ACCEPTED_TASKS, MIN_FEATURES, MIN_ROWS  # imports thresholds from config.py
from pipeline.data.base import CandidateInfo, Dataset, infer_domain
from pipeline.hard_rules import runner as hard_rules  # hard rule checks
from pipeline import stats  # for recording stats about the datasets we process

logger = logging.getLogger(__name__)

# put openml's own download cache on scratch too when PIPELINE_DATA is set (cluster);
# otherwise leave openml's default (~/.openml). Native OPENML_CACHE_DIR still wins if set.
_data_root = os.environ.get("PIPELINE_DATA")
if _data_root and not os.environ.get("OPENML_CACHE_DIR"):
    openml.config.set_root_cache_directory(str(Path(_data_root) / "openml_cache"))


def list_candidates(max_candidates: int = 100) -> list[CandidateInfo]:
    """Orchestrator: searches OpenML, filters by size + hard rules, returns candidates.
    This is the equivalent of list_candidates in geo_array_loader.py.
    """
    logger.info("Fetching OpenML dataset listing...")

    # Step 1: get ALL datasets from OpenML as a dataframe
    all_ds = openml.datasets.list_datasets(output_format="dataframe")

    # Step 2: quick size filter - only keep datasets with enough rows and features
    filtered = all_ds[
        (all_ds["NumberOfFeatures"] >= MIN_FEATURES)
        & (all_ds["NumberOfInstances"] >= MIN_ROWS)
        & (all_ds["format"].str.lower() != "sparse_arff")  # sparse ARFF can't be parsed (openml+pandas2 bug)
    ].copy()

    logger.info("OpenML: %d total, %d after size filter", len(all_ds), len(filtered))

    candidates: list[CandidateInfo] = []

    # Step 3: loop through filtered datasets and check hard rules
    for _, row in filtered.head(max_candidates).iterrows():
        did = int(row["did"])  # dataset ID on OpenML
        name = str(row.get("name", ""))
        n_samples = int(row.get("NumberOfInstances", 0))
        n_features = int(row.get("NumberOfFeatures", 0))
        licence = str(row.get("licence", "") or "")

        # ask OpenML what kind of task this dataset is for (classification/regression)
        task_type = _fetch_task_type(did)

        # Step 4: run metadata-level hard rules (A1 task type, A2 synthetic, A4 dimensions, A5 licence)
        results = hard_rules.run_metadata_checks(
            n_samples=n_samples,
            n_features=n_features,
            task_type=task_type,
            licence=licence,
            source="openml",
            name=name,
            metadata={"tags": []},
        )

        stats.record(str(did), "openml", name, results)

        # if any hard rule failed, skip this dataset
        failed = hard_rules.failed_rules(results)
        if failed:
            reasons = ", ".join(f"{r.rule}: {r.reason}" for r in failed)
            logger.debug("OpenML %d (%s) discarded: %s", did, name, reasons)
            continue

        # Step 5: passed all metadata checks, save as a candidate
        candidates.append(CandidateInfo(
            id=str(did),
            source="openml",
            name=name,
            n_samples=n_samples,
            n_features=n_features,
            task_type=task_type,
            licence=licence,
            url=f"https://www.openml.org/d/{did}",
            metadata={},
            domain=infer_domain(name),
        ))

    logger.info("OpenML: %d candidates after metadata filters", len(candidates))
    return candidates


def fetch(candidate: CandidateInfo) -> Dataset | None: #aktuell werden hier auch noch die harten regeln überprüft auslagern?
    """Downloads the actual data for one candidate and runs data-level hard rules.
    Returns a Dataset object if everything passes, None if it fails.
    """
    did = int(candidate.id)
    logger.info("Downloading OpenML dataset %d (%s)...", did, candidate.name)

    # Step 1: download the actual data from OpenML
    # Unlike GEO, OpenML gives us X (features) and y (target) directly
    try:
        ds = openml.datasets.get_dataset(did, download_data=True)
        X, y, _, attribute_names = ds.get_data(
            target=ds.default_target_attribute,  # OpenML already knows the target column
            dataset_format="dataframe",
        )
    except Exception as e:
        logger.warning("Failed to download OpenML %d: %s", did, e)
        return None

    if X is None or y is None:
        logger.warning("OpenML %d: no data returned", did)
        return None

    # run data-level hard rules on the actual downloaded data.
    # A1 = if metadata task was 'unknown', infer from target (sets details['inferred_task'])
    # A3 = does the data have predictive signal (RandomForest cross-val)
    # A4 = do the real dimensions still meet thresholds
    data_results = hard_rules.run_data_checks(X, y, task_type=candidate.task_type)
    stats.record(str(did), "openml", candidate.name, data_results)

    failed = hard_rules.failed_rules(data_results)
    if failed:
        reasons = ", ".join(f"{r.rule}: {r.reason}" for r in failed)
        logger.info("OpenML %d discarded (data check): %s", did, reasons)
        return None

    task_type = hard_rules.inferred_task_type(data_results) or candidate.task_type

    logger.info(
        "OpenML %d: loaded %d samples x %d features, task=%s",
        did, X.shape[0], X.shape[1], task_type,
    )

    

    # everything passed, build the final Dataset and return it
    return Dataset(
        id=f"OpenML-{did}",
        source="openml",
        name=candidate.name,
        X=X,
        y=y,
        task_type=task_type,
        metadata={
            "openml_did": did,
            "licence": candidate.licence,
            "url": candidate.url,
        },
        domain=candidate.domain,
    )


def _fetch_task_type(did: int) -> str:
    """Helper: asks OpenML what task type a dataset has (classification, regression, etc.).
    OpenML stores tasks separately from datasets, so we need an extra API call.
    Returns 'unknown' if we can't figure it out; fetch() will infer it later from the target.
    """
    try:
        tasks = openml.tasks.list_tasks(data_id=did, output_format="dataframe")
        if not tasks.empty:
            # normalize: "Supervised Classification" -> "supervised_classification"
            tt = tasks["task_type"].iloc[0].lower().replace(" ", "_")
            if any(acc in tt for acc in ACCEPTED_TASKS):
                return tt
        time.sleep(0.3)  # rate limit so we don't hammer the API
    except Exception:
        pass
    return "unknown"


