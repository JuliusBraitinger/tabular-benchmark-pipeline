"""Dataset registry.

Central dispatcher for all data sources (OpenML, TCGA, GEO). Instead of
calling each loader directly, the rest of the pipeline talks to this registry.
It picks the loader based on the 'source' string.

Adding a new data source = one new entry in the _LOADER_MODULES dict below.
"""
from __future__ import annotations

import importlib
import logging
from functools import lru_cache

from pipeline.data import base  # Dataset + CandidateInfo dataclasses

logger = logging.getLogger(__name__)

# source name -> module path. Imported lazily so an unused loader never loads
# (e.g. kaggle authenticates on import and kills a headless run without creds).
_LOADER_MODULES: dict[str, str] = {
    "openml": "pipeline.data.openml_loader",
    "tcga": "pipeline.data.tcga_loader",
    "geo_array": "pipeline.data.geo_array_loader",
    "geo_rnaseq": "pipeline.data.geo_rnaseq_loader",
    "kaggle": "pipeline.data.kaggle_loader",
    "uci": "pipeline.data.uci_loader",
    "local": "pipeline.data.local_loader",
    "chembl": "pipeline.data.chembl_loader",
    "metagenomics": "pipeline.data.metagenomics_loader",
    "cmd": "pipeline.data.cmd_loader",
    "mgnify": "pipeline.data.mgnify_loader",
}


@lru_cache(maxsize=None)
def _loader(source: str):
    # import the loader module on first use only; each exposes list_candidates() + fetch()
    return importlib.import_module(_LOADER_MODULES[source])


def list_candidates(
    sources: list[str] | None = None,
    max_per_source: int = 50,
) -> list[base.CandidateInfo]:
    """Collects candidate datasets from all (or a subset of) sources.
    If sources=None, scrapes every source in _LOADER_MODULES.
    Returns a combined list of CandidateInfo from all scraped sources.
    """
    # if caller didn't specify sources, use all of them
    sources = sources or list(_LOADER_MODULES.keys())
    all_candidates: list[base.CandidateInfo] = []

    # loop over each source and call its list_candidates function
    for source in sources:
        if source not in _LOADER_MODULES:
            logger.warning("Unknown source: %s", source)
            continue
        logger.info("Scraping %s...", source)
        # dispatch: call the right loader's list function (loader imported lazily here).
        # one source failing (network/DNS/import) must not abort the whole run -> skip it.
        try:
            candidates = _loader(source).list_candidates(max_candidates=max_per_source)
        except Exception:
            logger.exception("scraping %s failed, skipping this source", source)
            continue
        all_candidates.extend(candidates)
        logger.info("%s: %d candidates", source, len(candidates))

    logger.info("Total candidates across all sources: %d", len(all_candidates))
    return all_candidates


def fetch(candidate: base.CandidateInfo):
    if candidate.source not in _LOADER_MODULES:
        logger.warning("Unknown source: %s", candidate.source)
        return None, []
    # dispatch: forward the candidate to the right loader (loader imported lazily here)
    return _loader(candidate.source).fetch(candidate)
