"""Entry point for the pipeline.
Run with:  python -m pipeline
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

from pipeline.data import registry
from pipeline import stats
from pipeline import soft_stats
from pipeline.soft_rules import s3_data_quality as s3
from pipeline.soft_rules import s1_uniqueness as s1
from pipeline.soft_rules import s2_iid as s2
from pipeline.soft_rules import s4_leakage as s4
from pipeline.soft_rules import s5_batch_effects as s5
from pipeline.soft_rules import s6_class_balance as s6




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
    candidates = registry.list_candidates(sources=["tcga"], max_per_source=50)
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


    # Phase 4: run soft rules and collect stats for accepted datasets
    log.info("=== Phase 4: running soft rules ===")

    # S1 needs the pool of fingerprints to compute pairwise uniqueness, so
    # precompute them in a first pass before scoring any dataset
    #NOT A FIX SOLUTION -> JUST WORKAROUND FOR NOW 
    log.info("Pre-computing S1 fingerprints for %d datasets...", len(datasets))
    t_start = time.time()
    s1_pool = {ds.id: s1.compute_fingerprint(ds) for ds in datasets}
    log.info("S1 fingerprints ready (%.1fs)", time.time() - t_start)

    pool_free_rules = [("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5), ("S6", s6)]
    n_datasets = len(datasets)
    for i, ds in enumerate(datasets, 1):
        log.info("[%d/%d] %s (X=%s) -- running soft rules", i, n_datasets, ds.id, ds.X.shape)
        results = []

        t_start = time.time()
        s1_result = s1.score(ds, pool_fingerprints=s1_pool)
        log.info("  S1: score=%.3f (%.1fs)", s1_result.score, time.time() - t_start)
        results.append(s1_result)

        for name, rule in pool_free_rules:
            t_start = time.time()
            result = rule.score(ds)
            log.info("  %s: score=%.3f (%.1fs)", name, result.score, time.time() - t_start)
            results.append(result)

        soft_stats.record(ds, results)

    soft_stats.save_csv("soft_stats.csv")
    log.info("Saved soft_stats.csv")

if __name__ == "__main__":
    main()
