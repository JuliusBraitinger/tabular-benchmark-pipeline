"""Entry point for the pipeline.
Run with:  python -m pipeline
"""
from __future__ import annotations

import logging
import os
import pickle
from collections import Counter
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
RESULTS_DIR = Path(os.environ.get("PIPELINE_DATA", "."))
MAX_SAVED_DATASETS = 200   # total cap; = MAX_PER_SOURCE_SAVED x sources, with headroom
MAX_PER_SOURCE_SAVED = 20  # cap saved datasets per source for a balanced benchmark
A6_EXEMPT_SOURCES = {"chembl", "cmd"}

log = logging.getLogger("pipeline")


def save_dataset(ds: Dataset, output_dir: Path) -> Path:
    ds_dir = output_dir / ds.id
    ds_dir.mkdir(parents=True, exist_ok=True)
    ds.X.to_parquet(ds_dir / "X.parquet")
    ds.y.to_frame("target").to_parquet(ds_dir / "y.parquet")
    with open(ds_dir / "meta.pkl", "wb") as fh:
        pickle.dump({"id": ds.id, "source": ds.source, "name": ds.name,
                     "task_type": ds.task_type, "metadata": ds.metadata}, fh)
    return ds_dir


def load_dataset(ds_dir: Path) -> Dataset:
    X = pd.read_parquet(ds_dir / "X.parquet")
    y = pd.read_parquet(ds_dir / "y.parquet")["target"]
    with open(ds_dir / "meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
    return Dataset(
        id=meta["id"], source=meta["source"], name=meta["name"],
        X=X, y=y, task_type=meta["task_type"], metadata=meta["metadata"],
    )


def record_stats(candidate, results) -> None:
    stats.record(candidate.id, candidate.source, candidate.name,
                 results, candidate.domain, candidate.url)


def scrape_candidates() -> list:
    log.info("=== Phase 1: scraping candidates ===")
    default_sources = ["openml", "tcga", "uci", "geo_rnaseq", "cmd",
                       "mgnify", "chembl", "metagenomics", "geo_array"]
    env_sources = os.environ.get("PIPELINE_SOURCES")  # e.g. PIPELINE_SOURCES=cmd for one source
    sources = env_sources.split(",") if env_sources else default_sources
    max_per_source = int(os.environ.get("PIPELINE_MAX_PER_SOURCE", "90"))

    candidates = registry.list_candidates(sources=sources, max_per_source=max_per_source)
    log.info("Got %d candidates", len(candidates))
    by_source: dict[str, list] = {}
    for c in candidates:
        by_source.setdefault(c.source, []).append(c)
    return [c for group in zip_longest(*by_source.values()) for c in group if c]

def fetch_and_save(candidates: list) -> list[Path]:
    # save one dataset at a time
    log.info("=== Phase 2: fetching + saving datasets ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_dirs: list[Path] = [] #list of dierctories/path of the datasets not object itself (memory save)
    saved_per_source: Counter = Counter()
    a6_pool: list[tuple[str, set]] = []  # (id, row-hash set) for cross-dataset duplicate check

    for i, candidate in enumerate(candidates, 1):
        if saved_per_source[candidate.source] >= MAX_PER_SOURCE_SAVED:
            continue  # per-source quota reached, skip without fetching

        log.info("[%d/%d] fetching %s", i, len(candidates), candidate.id)
        try:
            ds, data_results = registry.fetch(candidate)
        except Exception:
            log.exception("fetch failed for %s, skipping", candidate.id)
            continue
        if ds is None:
            record_stats(candidate, data_results)
            continue

        # A6 cross-duplicate check against pool
        if candidate.source not in A6_EXEMPT_SOURCES:
            a6_result = a6_cross_duplicate.check(ds, a6_pool)
            record_stats(candidate, data_results + [a6_result])
            if not a6_result.passed:
                log.info("  A6 rejected %s: %s", ds.id, a6_result.reason)
                del ds
                continue
            a6_pool.append((ds.id, a6_cross_duplicate.hash_sorted_rows(ds)))
        else:
            record_stats(candidate, data_results)

        saved_dirs.append(save_dataset(ds, OUTPUT_DIR))
        saved_per_source[candidate.source] += 1
        log.info("  saved %s: X=%s task=%s", ds.id, ds.X.shape, ds.task_type)
        del ds  # release memory before fetching the next candidate

        if len(saved_dirs) >= MAX_SAVED_DATASETS:
            log.info("Reached MAX_SAVED_DATASETS=%d, stopping fetch loop", MAX_SAVED_DATASETS)
            break

    return saved_dirs 


def run_soft_rules(saved_dirs: list[Path]) -> None:
    log.info("=== Phase 4: running soft rules ===")

    # S1 compares each dataset against all the others, so gather every fingerprint first.
    log.info("Pre-computing S1 fingerprints for %d datasets...", len(saved_dirs))
    s1_pool = {}
    for ds_dir in saved_dirs:
        ds = load_dataset(ds_dir)
        s1_pool[ds.id] = s1.compute_fingerprint(ds)

    names = ["S1", "S2", "S3", "S4", "S5", "S6"]
    pool_free = [s2, s3, s4, s5, s6]   # S2..S6 don't need the pool, only S1 does
    for i, ds_dir in enumerate(saved_dirs, 1):
        ds = load_dataset(ds_dir)
        log.info("[%d/%d] %s (X=%s) -- soft rules", i, len(saved_dirs), ds.id, ds.X.shape)
        results = [s1.score(ds, pool_fingerprints=s1_pool)]
        for rule in pool_free:
            results.append(rule.score(ds))
        log.info("  %s", {n: round(r.score, 3) for n, r in zip(names, results)})
        soft_stats.record(ds, results)

    soft_stats.save_csv(str(RESULTS_DIR / "soft_stats.csv"))
    log.info("Saved soft_stats.csv")


def run_reports(saved_dirs: list[Path]) -> None:
    # per-dataset reports (characterisation, not a gate); skippable since it retrains a model each
    if os.environ.get("PIPELINE_SKIP_REPORTS"):
        log.info("PIPELINE_SKIP_REPORTS set -- skipping Phase 5 reports")
        return
    log.info("=== Phase 5: per-dataset reports ===")
    from pipeline.viz.reports import generate_reports
    csv = generate_reports(saved_dirs, load_dataset, RESULTS_DIR)
    log.info("Wrote per-dataset report.html files + %s", csv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    candidates = scrape_candidates()
    saved_dirs = fetch_and_save(candidates)
    rule_csv = str(RESULTS_DIR / "rule_stats.csv")
    stats.save_csv(rule_csv)
    stats.build_sankey(rule_csv).write_html(str(RESULTS_DIR / "rule_stats_sankey.html"))
    log.info("%d datasets saved to %s. Saved rule_stats.csv.", len(saved_dirs), OUTPUT_DIR)
    if not saved_dirs:
        log.info("No datasets saved, skipping soft rules.")
        return

    run_soft_rules(saved_dirs)
    run_reports(saved_dirs)


if __name__ == "__main__":
    main()
