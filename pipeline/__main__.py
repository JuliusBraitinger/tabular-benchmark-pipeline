"""Entry point for the pipeline.
Run with:  python -m pipeline
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from itertools import zip_longest
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


OUTPUT_DIR = Path(os.environ.get("PIPELINE_DATA", "data")) / "datasets"
# rule_stats/soft_stats/sankey land here: under PIPELINE_DATA on the cluster, cwd locally
RESULTS_DIR = Path(os.environ.get("PIPELINE_DATA", "."))
MAX_SAVED_DATASETS = 120
MAX_PER_SOURCE_SAVED = 20  # cap saved datasets per source for a balanced benchmark  
A6_EXEMPT_SOURCES = {"chembl", "cmd"}

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
    default_sources = ["cmd"]
    env_sources = os.environ.get("PIPELINE_SOURCES")  # e.g. PIPELINE_SOURCES=cmd to run one source
    sources = env_sources.split(",") if env_sources else default_sources
    candidates = registry.list_candidates(sources=sources, max_per_source=90)
    log.info("Got %d candidates", len(candidates))

    # interleave sources round-robin so the total cap spreads evenly instead of
    # the first sources in the list eating the whole budget.
    by_source: dict[str, list] = {}
    for c in candidates:
        by_source.setdefault(c.source, []).append(c)
    candidates = [c for group in zip_longest(*by_source.values()) for c in group if c]

    # Phase 2: fetch each dataset, save to disk, then drop from memory.
    # Streaming save keeps RAM bounded to one dataset at a time and means a
    # crash mid-loop never loses already-fetched datasets.
    log.info("=== Phase 2: fetching + saving datasets ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_dirs: list[Path] = []
    saved_per_source: dict[str, int] = {}
    a6_pool: list[tuple[str, set]] = []  # (id, row-hash set) for cross-dataset duplicate check
    for i, candidate in enumerate(candidates, 1):
        if saved_per_source.get(candidate.source, 0) >= MAX_PER_SOURCE_SAVED:
            continue  # per-source quota reached, skip without fetching
        log.info("[%d/%d] fetching %s", i, len(candidates), candidate.id)
        try:
            ds, data_results = registry.fetch(candidate)
        except Exception:
            log.exception("fetch failed for %s, skipping", candidate.id)
            continue
        if ds is None:
            stats.record(candidate.id, candidate.source, candidate.name, data_results, candidate.domain, candidate.url)
            continue

        if candidate.source in A6_EXEMPT_SOURCES:
            stats.record(candidate.id, candidate.source, candidate.name, data_results, candidate.domain, candidate.url)
        else:
            a6_result = a6_cross_duplicate.check(ds, a6_pool)
            stats.record(candidate.id, candidate.source, candidate.name, data_results + [a6_result], candidate.domain, candidate.url)
            if not a6_result.passed:
                log.info("  A6 rejected %s: %s", ds.id, a6_result.reason)
                del ds
                continue
            a6_pool.append((ds.id, a6_cross_duplicate.hash_sorted_rows(ds)))

        ds_dir = _save_dataset(ds, OUTPUT_DIR)
        log.info("  saved %s: X=%s task=%s", ds.id, ds.X.shape, ds.task_type)
        saved_dirs.append(ds_dir)
        saved_per_source[candidate.source] = saved_per_source.get(candidate.source, 0) + 1
        del ds  # release memory before fetching the next candidate

        if len(saved_dirs) >= MAX_SAVED_DATASETS:
            log.info("Reached MAX_SAVED_DATASETS=%d, stopping fetch loop", MAX_SAVED_DATASETS)
            break

    rule_csv = str(RESULTS_DIR / "rule_stats.csv")
    stats.save_csv(rule_csv)
    stats.build_sankey(rule_csv).write_html(str(RESULTS_DIR / "rule_stats_sankey.html"))
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

    soft_stats.save_csv(str(RESULTS_DIR / "soft_stats.csv"))
    log.info("Saved soft_stats.csv")

    # Phase 5: per-dataset evaluation reports (characterisation, not a gate).
    # Skippable via PIPELINE_SKIP_REPORTS since it retrains a model per dataset.
    if os.environ.get("PIPELINE_SKIP_REPORTS"):
        log.info("PIPELINE_SKIP_REPORTS set -- skipping Phase 5 reports")
        return
    log.info("=== Phase 5: per-dataset reports ===")
    from pipeline.viz.reports import generate_reports
    csv = generate_reports(saved_dirs, _load_dataset, RESULTS_DIR)
    log.info("Wrote per-dataset report.html files + %s", csv)


if __name__ == "__main__":
    main()
