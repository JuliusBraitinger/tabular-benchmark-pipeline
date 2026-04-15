"""Entry point for the pipeline.
Run with:  python -m pipeline
"""
from __future__ import annotations

import logging

from pipeline.data import registry


def main() -> None:
    # logging setup with timestamps and log levels
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("pipeline")

    # Phase 1: scrape candidates from all sources (metadata + metadata hard rules)
    log.info("=== Phase 1: scraping candidates ===")
    candidates = registry.list_candidates(
        sources=["openml"],   
        max_per_source=5,   
    )
    log.info("Got %d candidates", len(candidates))

    # Phase 2: download the actual data + run data-level hard rules
    log.info("=== Phase 2: fetching datasets ===")
    datasets = registry.fetch_all(candidates, max_datasets=3)
    log.info("Got %d datasets", len(datasets))

    # show a quick summary of what made it through
    for ds in datasets:
        log.info("  %s: %s  X=%s  y=%s", ds.id, ds.name, ds.X.shape, ds.y.shape)


if __name__ == "__main__":
    main()
