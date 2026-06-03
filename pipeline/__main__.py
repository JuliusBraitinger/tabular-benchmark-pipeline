"""Entry point for the pipeline.
Run with:  python -m pipeline
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import pandas as pd

from pipeline.data import registry
from pipeline.data.base import Dataset
from pipeline.hard_rules import a6_cross_duplicate
from pipeline import stats
from pipeline import soft_stats
from pipeline.soft_rules import s3_data_quality as s3
from pipeline.soft_rules import s1_uniqueness as s1
from pipeline.soft_rules import s2_iid as s2
from pipeline.soft_rules import s4_leakage as s4
from pipeline.soft_rules import s5_batch_effects as s5
from pipeline.soft_rules import s6_class_balance as s6


OUTPUT_DIR = Path("data/datasets")
MAX_SAVED_DATASETS = 100  # stop fetching once this many datasets pass ALL hard rules


def _save_dataset(ds: Dataset, output_dir: Path) -> Path:
    """Persist a Dataset to {output_dir}/{ds.id}/ as parquet + pickle."""
    ds_dir = output_dir / ds.id
    ds_dir.mkdir(parents=True, exist_ok=True)
    X = ds.X.sparse.to_dense() if hasattr(ds.X, "sparse") else ds.X
    y = ds.y.sparse.to_dense() if hasattr(ds.y, "sparse") else ds.y
    X.to_parquet(ds_dir / "X.parquet")
    y.to_frame("target").to_parquet(ds_dir / "y.parquet")
    with open(ds_dir / "meta.pkl", "wb") as fh:
        pickle.dump({
            "id": ds.id, "source": ds.source, "name": ds.name,
            "task_type": ds.task_type, "metadata": ds.metadata,
        }, fh)
    return ds_dir


def _load_dataset(ds_dir: Path) -> Dataset:
    """Reconstruct a Dataset from its on-disk directory."""
    X = pd.read_parquet(ds_dir / "X.parquet")
    y = pd.read_parquet(ds_dir / "y.parquet")["target"]
    with open(ds_dir / "meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
    return Dataset(
        id=meta["id"], source=meta["source"], name=meta["name"],
        X=X, y=y, task_type=meta["task_type"], metadata=meta["metadata"],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("pipeline")

    # Phase 1: scrape candidates (metadata + metadata hard rules)
    log.info("=== Phase 1: scraping candidates ===")
    candidates = registry.list_candidates(sources=["uci"], max_per_source=30)
    
    log.info("Got %d candidates", len(candidates))

    # Phase 2: fetch each dataset, save to disk, then drop from memory.
    # Streaming save keeps RAM bounded to one dataset at a time and means a
    # crash mid-loop never loses already-fetched datasets.
    log.info("=== Phase 2: fetching + saving datasets ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_dirs: list[Path] = []
    a6_pool: list[tuple[str, set]] = []  # (id, row-hash set) for cross-dataset duplicate check
    for i, candidate in enumerate(candidates, 1):
        log.info("[%d/%d] fetching %s", i, len(candidates), candidate.id)
        try:
            ds = registry.fetch(candidate)
        except Exception:
            log.exception("fetch failed for %s, skipping", candidate.id)
            continue
        if ds is None:
            continue

        # A6: reject if this dataset duplicates one already accepted
        a6_result = a6_cross_duplicate.check(ds, a6_pool)
        if not a6_result.passed:
            log.info("  A6 rejected %s: %s", ds.id, a6_result.reason)
            del ds
            continue

        ds_dir = _save_dataset(ds, OUTPUT_DIR)
        log.info("  saved %s: X=%s task=%s", ds.id, ds.X.shape, ds.task_type)
        saved_dirs.append(ds_dir)
        a6_pool.append((ds.id, a6_cross_duplicate.hash_sorted_rows(ds)))
        del ds  # release memory before fetching the next candidate

        if len(saved_dirs) >= MAX_SAVED_DATASETS:
            log.info("Reached MAX_SAVED_DATASETS=%d, stopping fetch loop", MAX_SAVED_DATASETS)
            break

    stats.save_csv("rule_stats.csv")
    log.info("%d datasets saved to %s. Saved rule_stats.csv.", len(saved_dirs), OUTPUT_DIR)

    if not saved_dirs:
        log.info("No datasets saved, skipping soft rules.")
        return

    # Phase 4: soft rules. Two passes over the saved datasets, each loading
    # one at a time from disk so memory stays bounded.
    log.info("=== Phase 4: running soft rules ===")
    log.info("Pre-computing S1 fingerprints for %d datasets...", len(saved_dirs))
    t_start = time.time()
    s1_pool = {}
    for ds_dir in saved_dirs:
        ds = _load_dataset(ds_dir)
        s1_pool[ds.id] = s1.compute_fingerprint(ds)
        del ds
    log.info("S1 fingerprints ready (%.1fs)", time.time() - t_start)

    pool_free_rules = [("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5), ("S6", s6)]
    for i, ds_dir in enumerate(saved_dirs, 1):
        ds = _load_dataset(ds_dir)
        log.info("[%d/%d] %s (X=%s) -- running soft rules", i, len(saved_dirs), ds.id, ds.X.shape)
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
        del ds  # release before loading the next

    soft_stats.save_csv("soft_stats.csv")
    log.info("Saved soft_stats.csv")


if __name__ == "__main__":
    main()
