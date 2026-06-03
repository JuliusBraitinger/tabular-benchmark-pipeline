> **Status: work in progress.** This pipeline is under active development and
>  contains bugs. It is not yet set up to run on the university's SLURM
> cluster. Everything here assumes a local environment. If you're using this
> package, expect rough edges and interfaces that may still change.

# Pipeline Architecture

A visual overview of what each file does and how they work together.

## Entry Point

```bash
python -m pipeline      # runs pipeline/__main__.py
```

`__main__.py` orchestrates the whole run in two phases. Datasets are streamed
to disk between phases so RAM stays bounded to one dataset at a time.

1. **Scrape + fetch** — `registry.list_candidates()` collects candidates from every
   loader, then `registry.fetch()` downloads each survivor. Each fetched
   `Dataset` is persisted under `data/datasets/{id}/` as `X.parquet`,
   `y.parquet`, `meta.pkl` and immediately released from memory.
2. **Soft rules** — every saved dataset is reloaded one at a time and scored
   by S1–S6. Results land in `soft_stats.csv`. Hard-rule stats from phase 1
   land in `rule_stats.csv`.

## File Map & Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            pipeline/config.py                                │
│  Central configuration: thresholds (MIN_ROWS, MIN_FEATURES), AHP weights,    │
│  API keys (loaded from .env), licence allow-list, accepted task types        │
│  → imported everywhere                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │  (constants)
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         pipeline/data/registry.py                            │
│                       ═══ DISPATCHER / ENTRY POINT ═══                       │
│                                                                              │
│   list_candidates()  ──▶ calls each loader's list_candidates()               │
│   fetch(candidate)   ──▶ routes to the right loader's fetch()                │
│   _LOADERS = {openml, tcga, geo_array, kaggle, uci}                          │
└─────────────────────────────────────────────────────────────────────────────┘
        │              │              │              │              │
        ▼              ▼              ▼              ▼              ▼
 ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
 │ openml_    │ │  tcga_     │ │ geo_array_ │ │  kaggle_   │ │   uci_     │
 │ loader.py  │ │ loader.py  │ │ loader.py  │ │ loader.py  │ │ loader.py  │
 │            │ │            │ │            │ │            │ │            │
 │ OpenML API │ │  GDC API   │ │ Entrez +   │ │ Kaggle API │ │ ucimlrepo  │
 │ (target    │ │ (target =  │ │ GEOparse   │ │ +Croissant │ │ package    │
 │  already   │ │  sample    │ │ (builds    │ │ (heuristic │ │            │
 │  defined)  │ │  type /    │ │  target    │ │  target    │ │            │
 │            │ │  vital)    │ │  from text)│ │  detection)│ │            │
 └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘
        │              │              │              │              │
        │ each loader calls hard_rules.runner twice:                │
        │   1) run_metadata_checks() — cheap, before download       │
        │   2) run_data_checks()     — expensive, after download    │
        ▼                                                           ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                  pipeline/hard_rules/runner.py                       │
   │                     ═══ RULE ORCHESTRATOR ═══                        │
   │                                                                      │
   │   _METADATA_CHECKS = [A1, A2, A4, A5]  ← cheap, metadata-only        │
   │   _DATA_CHECKS     = [A3, A4]          ← needs downloaded data       │
   │                                                                      │
   │   run_metadata_checks(**kwargs) → loops and calls each rule          │
   │   run_data_checks(X, y, ...)    → loops and calls each rule          │
   │   all_passed() / failed_rules() → helpers for checking results       │
   └─────────────────────────────────────────────────────────────────────┘
        │         │          │          │          │
        ▼         ▼          ▼          ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │  a1_    │ │  a2_    │ │  a3_    │ │  a4_    │ │  a5_    │
   │ task_   │ │synthetic│ │ signal  │ │dimension│ │ licence │
   │ type.py │ │  .py    │ │  .py    │ │  .py    │ │  .py    │
   │         │ │         │ │         │ │         │ │         │
   │ classif │ │ regex + │ │permutat-│ │ N ≥1000 │ │ open    │
   │ /regr?  │ │ TCGA/   │ │ ion RF  │ │ P ≥1000 │ │ licence?│
   │         │ │ GEO     │ │ +TabPFN │ │ both!   │ │         │
   │ meta +  │ │ auto-   │ │  ↓      │ │         │ │         │
   │ data    │ │ pass    │ │data only│ │ meta+   │ │ meta    │
   └─────────┘ └─────────┘ └─────────┘ │ data    │ │ only    │
                                       └─────────┘ └─────────┘
                                                 │
                                                 │
                                                 │
                                                 │
                                                 ▼  
                                      ┌────────────────────────┐
                                      │ a6_cross_duplicates.py │
                                      │                        │
                                      │  checks for duplicates │   
                                      │   in pool of datasets  │                    
                                      │ • RuleResult           │
                                      │   (rule,passed,reason) │
                                      └────────────────────────┘
                                                 │
                                                 ▼  (returns RuleResult)
                                      ┌──────────────────────┐
                                      │ hard_rules/base.py   │
                                      │ • RuleResult         │
                                      │   (rule,passed,reason)│
                                      └──────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────┐
   │                      pipeline/data/base.py                           │
   │           ═══ SHARED DATA TYPES (used by everyone) ═══               │
   │                                                                      │
   │   • CandidateInfo  = scraped dataset metadata (before download)      │
   │   • Dataset        = full dataset (X, y, metadata) after fetch       │
   │   • DataSource     = Protocol interface that each loader follows     │
   └─────────────────────────────────────────────────────────────────────┘
```

After phase 1 every surviving dataset lives on disk. Phase 2 reloads them
one at a time and runs the soft rules:

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │                pipeline/soft_rules/ (called by __main__)             │
   │                                                                      │
   │   Each rule exposes: score(dataset, ...) → SoftRuleResult            │
   │   SoftRuleResult(rule, score in [0,1], details dict)                 │
   └─────────────────────────────────────────────────────────────────────┘
        │         │          │          │          │          │
        ▼         ▼          ▼          ▼          ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │   S1    │ │   S2    │ │   S3    │ │   S4    │ │   S5    │ │   S6    │
   │ unique  │ │  IID    │ │ quality │ │ leakage │ │ batch   │ │ class   │
   │ -ness   │ │         │ │         │ │         │ │ effects │ │ balance │
   │         │ │         │ │         │ │         │ │         │ │         │
   │ stat    │ │ exact   │ │ miss /  │ │ TODO    │ │   TODO  │ │shannon  │
   │ finger- │ │duplicate│ │ const / │ │         │ │         │ │entropy  │
   │ print + │ │ rows on │ │ outliers│ │         │ │         │ │         │
   │ cosine  │ │ (X | y) │ │ consist │ │         │ │         │ │         │
   │ vs pool │ │  hashing│ │         │ │         │ │         │ │         │
   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
                                                                    │
                                                                    ▼
                                                    ┌──────────────────────┐
                                                    │   soft_stats.py      │
                                                    │ • record(ds, results)│
                                                    │ • save_csv()         │
                                                    │  → soft_stats.csv    │
                                                    └──────────────────────┘
```

S1 is a special case — it needs the pool of all fingerprints, so `__main__`
pre-computes them once before the per-dataset loop. The other rules are
pool-free and run in the same loop.

## End-to-End Flow

```
   registry.list_candidates(sources=[…])
           │
           ├─▶ openml_loader.list_candidates() ─┐
           ├─▶ tcga_loader.list_candidates()   ─┤
           ├─▶ geo_loader.list_candidates()    ─┼─ each runs metadata
           ├─▶ kaggle_loader.list_candidates() ─┤    hard rules
           └─▶ uci_loader.list_candidates()    ─┘
                              │
                              ▼
                [list of CandidateInfo objects]
                              │
                              ▼
              for candidate in candidates:
                  ds = registry.fetch(candidate)   # runs data hard rules
                  if ds is None: continue
                  save(ds → data/datasets/{id}/)   # parquet + pickle
                  del ds                            # free RAM
                              │
                              ▼
              [data/datasets/ now contains every passing dataset]
                              │
                              ▼
              for each saved dataset:
                  ds = load_from_disk(ds_dir)
                  run S1, S2, S3, S4, S5, S6 → soft_stats.csv
```

## Key Patterns

- **Dispatcher pattern** — `registry.py` routes calls based on the `source` string. Adding a new data source = adding one entry to the `_LOADERS` dict.
- **Two-phase validation** — cheap metadata checks first, expensive data checks only on survivors.
- **Streaming save** — each fetched dataset is written to disk and dropped from memory before the next one is fetched, so RAM is bounded regardless of cohort size.
- **Registry pattern (for rules)** — `runner.py` keeps a list of rule functions and loops over them. Adding a new rule = adding one line.
- **Protocol-based loaders** — every loader must expose `list_candidates()` + `fetch()`. That's what makes the dispatcher work.
- **Continuous scores** — soft rules return `score ∈ [0, 1]` (1 = clean); tiers (0/10/20/…/100) are only for human-readable display.

## Soft Rule Status

| Rule | Points | Implementation |
|------|--------|----------------|
| S1 Uniqueness     | 10 | per-column moment fingerprint (mean/std/skew/kurt) → cosine vs pool |
| S2 IID            | 10 | strict exact-duplicate rows on `(X | y)` via row hashing |
| S3 Data Quality   | 15 | composite: completeness, consistency, outliers (IF), constant + quasi-constant features |
| S4 Data Leakage   | 20 | per-feature predictive stat (Mann-Whitney / Spearman); group k-fold + MI spike planned |
| S5 Batch Effects  |  – | **stub** — returns 1.0; needs batch labels we don't reliably have |
| S6 Class Balance  |  5 | normalized Shannon entropy of class distribution |
| S7 Domain-QC      |  – | **placeholder** — too domain-specific to automate generically |

S2's duplicate detection used to live inside S3 as a "uniqueness" sub-metric
but was moved out — duplicates are an IID-assumption violation, not a data
cleanliness signal, and keeping them in both rules would double-count.

## Hard Rule Status

| Rule | Status |
|------|--------|
| A1 Task type        | implemented (classification/regression, infers from target if unknown) |
| A2 Synthetic check  | implemented (regex on name/tags + `make_*` sklearn generators; TCGA + GEO auto-pass) |
| A3 Signal           | implemented (permutation RF on `balanced_accuracy`/R²; TabPFN second-opinion for trivial signals) |
| A4 Dimensions       | implemented (N ≥ MIN_ROWS, P ≥ MIN_FEATURES; both metadata and data checks) |
| A5 Licence          | implemented (normalises string, rejects NC/ND, allow-list) |
| A6 Cross-duplicates | implemented (row-sort + bytes-hash; rejects if overlap with already-accepted dataset > threshold) |

## Domain tagging

Each `CandidateInfo` and `Dataset` carries a `domain` field — one of:
- `"biomedical"` — clinical / cancer / patient (TCGA hardcodes this)
- `"biological"` — broader life sciences (GEO microarray + RNA-seq hardcode this)
- `"general"` — everything else; default for OpenML / Kaggle / UCI

For the heterogeneous sources, `data/base.py:infer_domain(name, tags, description)`
runs a keyword regex to upgrade `"general"` candidates to biomedical/biological
when titles mention cancer, gene expression, methylation, etc.

## Testing

Quick smoke test for any loader — runs `list_candidates` against the real API and stops after the first candidate that passes metadata hard rules. Swap `tcga_loader` for `openml_loader`, `geo_array_loader`, `kaggle_loader`, or `uci_loader` to test the others.

```bash
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
from pipeline.data import tcga_loader
cands = tcga_loader.list_candidates(max_candidates=1)
print('---'); print('Got', len(cands), 'candidate(s)')
for c in cands:
    print(c.id, c.name, 'N=', c.n_samples, 'P=', c.n_features, 'task=', c.task_type)
"
```

What it does:
1. Turns on `INFO` logging so the loader's own progress lines show up.
2. Imports the loader (run from the project root so `pipeline/` is on the import path).
3. Calls `list_candidates(max_candidates=1)` — stops at the first valid candidate to stay fast.
4. Prints `id`, `name`, N, P and task_type so you can confirm the shape of the result.

Notes:
- `max_candidates` is the **max kept after filtering**, not scanned. If the first N fail hard rules, the loader keeps going.
- GEO is noticeably slower (Entrez rate limits + heavier downloads). Expect a minute or two even for `max_candidates=1`.

### End-to-end test (includes `fetch`)

To also exercise the download + data-level hard rules, chain `fetch` after `list_candidates`:

```python
ds = tcga_loader.fetch(cands[0])
print(ds.X.shape, ds.y.shape, ds.task_type)
```

This downloads the full feature matrix, so for a large TCGA project it's hundreds of files — use sparingly.

### Running soft rules on already-saved datasets

If `data/datasets/` already contains datasets from a previous run, you can
re-score them without refetching. `__main__._load_dataset` already does the
parquet + pickle boilerplate, and `rglob("X.parquet")` walks any depth so
kaggle's two-level `{owner}/{slug}/` layout works the same as everything else:

```python
from pathlib import Path
from pipeline.__main__ import _load_dataset
from pipeline.soft_rules import s2_iid

for x_path in Path("data/datasets").rglob("X.parquet"):
    ds = _load_dataset(x_path.parent)
    print(ds.id, s2_iid.score(ds).score)
```
