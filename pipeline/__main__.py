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
    candidates = registry.list_candidates(
        sources=["tcga"],
        max_per_source=1,
    )
    log.info("Got %d candidates", len(candidates))

    # Phase 2: download the actual data + run data-level hard rules
    log.info("=== Phase 2: fetching datasets ===")
    datasets = registry.fetch_all(candidates, max_datasets=5)
    log.info("Got %d datasets", len(datasets))

    # Phase 3: save datasets to disk for reusing later
    log.info("=== Phase 3: saving to disk ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        ds_dir = OUTPUT_DIR / ds.id
        ds_dir.mkdir(exist_ok=True)
        X = ds.X
        if hasattr(X, "sparse"):
            X = X.sparse.to_dense()
        X.to_parquet(ds_dir / "X.parquet")
        ds.y.to_frame("target").to_parquet(ds_dir / "y.parquet")
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
