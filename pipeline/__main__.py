"""Entry point for the pipeline.
Run with:  python -m pipeline
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

from pipeline.data import registry
from pipeline import stats

OUTPUT_DIR = Path("data/datasets")


def main() -> None:
    # logging setup with timestamps and log levels
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("pipeline")

    # Phase 1: scrape candidates from all sources (metadata + metadata hard rules)
    log.info("=== Phase 1: scraping candidates ===")
    # geo gets searched the most since most studies are too small and get filtered out
    # openml and tcga find candidates more easily
    geo_candidates = registry.list_candidates(sources=["geo_array"], max_per_source=50)
    openml_candidates = registry.list_candidates(sources=["openml"], max_per_source=20)
    tcga_candidates = registry.list_candidates(sources=["tcga"], max_per_source=20)
    candidates = geo_candidates + openml_candidates + tcga_candidates
    log.info("Got %d candidates", len(candidates))

    # Phase 2: download the actual data + run data-level hard rules
    log.info("=== Phase 2: fetching datasets ===")
    datasets = registry.fetch_all(candidates)
    log.info("Got %d datasets", len(datasets))

    # Phase 3: save datasets to disk for reusing later
    log.info("=== Phase 3: saving to disk ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        ds_dir = OUTPUT_DIR / ds.id
        ds_dir.mkdir(exist_ok=True)
        X = ds.X
        if hasattr(X, "sparse"): #convert back to dense 
            X = X.sparse.to_dense()
        X.to_parquet(ds_dir / "X.parquet")
        y = ds.y
        if hasattr(y, "sparse"): #also for y data for some datasets 
            y = y.sparse.to_dense()
        y.to_frame("target").to_parquet(ds_dir / "y.parquet")
        with open(ds_dir / "meta.pkl", "wb") as f:
            pickle.dump({
                "id": ds.id, "source": ds.source, "name": ds.name,
                "task_type": ds.task_type, "metadata": ds.metadata,
            }, f)
        log.info("  Saved %s: X=%s task=%s", ds.id, ds.X.shape, ds.task_type)

    # save the rule stats csv
    stats.save_csv("rule_stats.csv")
    log.info("Saved rule_stats.csv")

    log.info("Done. %d datasets saved to %s", len(datasets), OUTPUT_DIR)


if __name__ == "__main__":
    main()
