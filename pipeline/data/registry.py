#TODO parallelize fetch_all. For each loader one thread -> no need to wait for the slowest source to finish (GEO)

"""Dataset registry.

Central dispatcher for all data sources (OpenML, TCGA, GEO). Instead of
calling each loader directly, the rest of the pipeline talks to this registry.
It picks the loader based on the 'source' string.

Adding a new data source = one new entry in the _LOADERS dict below.
"""
from __future__ import annotations

import logging
from typing import Callable

from pipeline.data import base  # Dataset + CandidateInfo dataclasses
from pipeline.data import geo_array_loader, kaggle_loader, openml_loader, tcga_loader  # one loader per source

logger = logging.getLogger(__name__)

# Lookup table: source name -> which functions to call for that source
# Each loader exposes two functions: list_candidates() and fetch()
_LOADERS: dict[str, dict[str, Callable]] = {
    "openml": {
        "list": openml_loader.list_candidates,
        "fetch": openml_loader.fetch,
    },
    "tcga": {
        "list": tcga_loader.list_candidates,
        "fetch": tcga_loader.fetch,
    },
    "geo_array": {
        "list": geo_array_loader.list_candidates,
        "fetch": geo_array_loader.fetch,
    },
    "kaggle": {
        "list": kaggle_loader.list_candidates,
        "fetch": kaggle_loader.fetch,
    },
}


def list_candidates(
    sources: list[str] | None = None,
    max_per_source: int = 50,
) -> list[base.CandidateInfo]:
    """Collects candidate datasets from all (or a subset of) sources.
    If sources=None, scrapes every source in _LOADERS.
    Returns a combined list of CandidateInfo from all scraped sources.
    """
    # if caller didn't specify sources, use all of them
    sources = sources or list(_LOADERS.keys())
    all_candidates: list[base.CandidateInfo] = []

    # loop over each source and call its list_candidates function
    for source in sources:
        if source not in _LOADERS:
            logger.warning("Unknown source: %s", source)
            continue
        logger.info("Scraping %s...", source)
        # dispatch: call the right loader's list function
        candidates = _LOADERS[source]["list"](max_candidates=max_per_source)
        all_candidates.extend(candidates)
        logger.info("%s: %d candidates", source, len(candidates))

    logger.info("Total candidates across all sources: %d", len(all_candidates))
    return all_candidates


def fetch(candidate: base.CandidateInfo) -> base.Dataset | None:
    """Downloads the actual data for ONE candidate.
    Looks at candidate.source ('openml', 'tcga', 'geo_array') and dispatches
    to the right loader's fetch function.
    basically dispatcher for each source to download which can be extended easily 
    """
    if candidate.source not in _LOADERS:
        logger.warning("Unknown source: %s", candidate.source)
        return None
    # dispatch: forward the candidate to the right loader
    return _LOADERS[candidate.source]["fetch"](candidate)


def fetch_all(
    candidates: list[base.CandidateInfo],
    max_datasets: int | None = None,
) -> list[base.Dataset]:
    """Downloads data for a WHOLE list of candidates (calls fetch() in a loop).
    Returns only the datasets that passed all hard rules; the rest are dropped.
    max_datasets stops the loop early once enough datasets are collected.
    """
    datasets: list[base.Dataset] = []
    for candidate in candidates:
        # stop early if we've collected enough
        if max_datasets and len(datasets) >= max_datasets:
            break
        # one bad candidate shouldn't kill the rest of the run
        try:
            dataset = fetch(candidate)
        except Exception:
            logger.exception("fetch failed for %s, skipping", candidate.id)
            continue
        # fetch() returns None if hard rules failed, skip those
        if dataset is not None:
            datasets.append(dataset)
    logger.info("Loaded %d datasets out of %d candidates", len(datasets), len(candidates))
    return datasets
