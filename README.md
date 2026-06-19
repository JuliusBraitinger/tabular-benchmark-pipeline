> **Status: work in progress.** This pipeline is under active development and
> contains bugs. The main curation pipeline (`python -m pipeline`) assumes a
> local environment; the GEO RNA-seq export pipeline that feeds the
> `geo_rnaseq` loader runs on the university's SLURM cluster (see the `tcml`
> branch). Expect rough edges and interfaces that may still change.

# Pipeline Architecture

A visual overview of what each file does and how they work together.

## Entry Point

```bash
python -m pipeline      # runs pipeline/__main__.py
```

`__main__.py` orchestrates the run as logged stages (labelled Phase 1, 2, and 4
in the code). Datasets are streamed to disk between stages so RAM stays bounded
to one dataset at a time.

1. **Phase 1 — scrape** — `registry.list_candidates()` collects candidates from
   every loader, each running the cheap metadata hard rules (A1/A2/A4/A5).
2. **Phase 2 — fetch + save** — `registry.fetch()` downloads each survivor and
   runs the data hard rules (A1/A2/A3/A4); `__main__` then applies **A6**
   (cross-dataset duplicate check) before persisting the `Dataset` under
   `data/datasets/{id}/` as `X.parquet`, `y.parquet`, `meta.pkl` and releasing
   it from memory. Stops at `MAX_SAVED_DATASETS` (100). Writes `rule_stats.csv`.
3. **Phase 4 — soft rules** — every saved dataset is reloaded one at a time and
   scored by S1–S6 (S1 first needs the pre-computed fingerprint pool). Results
   land in `soft_stats.csv`.

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
│   _LOADERS = {openml, tcga, geo_array, geo_rnaseq, kaggle, uci}              │
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
   │   _DATA_CHECKS     = [A1, A2, A3, A4]  ← needs downloaded data       │
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
   │ stat    │ │ exact   │ │ miss /  │ │ pred-gap│ │   TODO  │ │shannon  │
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

## Data Sources

Six loaders feed the dispatcher, each owning its own scrape → fetch path:

| Source | Loader | Access | Target |
|--------|--------|--------|--------|
| `openml` | `openml_loader.py` | OpenML API | already defined (skips `sparse_arff`) |
| `tcga` | `tcga_loader.py` | GDC API | sample_type (tumor/normal), vital_status fallback |
| `geo_array` | `geo_array_loader.py` | Entrez + GEOparse | built from sample text — **slow scrape** |
| `geo_rnaseq` | `geo_rnaseq_loader.py` | local Parquet exports | classification col with most labels |
| `kaggle` | `kaggle_loader.py` | Kaggle API | heuristic detection |
| `uci` | `uci_loader.py` | ucimlrepo | already defined |

**`geo_rnaseq` is new and unlike the others — it has no live API.** It reads
Parquet exports (`<ACC>_X.parquet`, `<ACC>_metadata.parquet`, `<ACC>_info.json`)
produced offline by the Nextflow RNA-seq pipeline
(`rnaseq_pipeline/build_tabular.py`), scanned from `data/rnaseq/` (override with
the `GEO_RNASEQ_DIR` env var). That export pipeline runs on the SLURM cluster
(`tcml` branch): each GEO study is fetched from SRA, pseudo-aligned with Salmon,
and collapsed to a gene-count matrix (always `P = 29607` genes against GRCh38,
capped at `MAX_SAMPLES` rows). The loader still runs the full hard-rule chain
(A1/A2/A4/A5 then A3/A4), so small-N cohorts fail A4 under `MIN_ROWS = 1000`.

## End-to-End Flow

```
   registry.list_candidates(sources=[…])
           │
           ├─▶ openml_loader.list_candidates()     ─┐
           ├─▶ tcga_loader.list_candidates()       ─┤
           ├─▶ geo_array_loader.list_candidates()  ─┤
           ├─▶ geo_rnaseq_loader.list_candidates() ─┼─ each runs metadata
           ├─▶ kaggle_loader.list_candidates()     ─┤    hard rules
           └─▶ uci_loader.list_candidates()        ─┘
                              │
                              ▼
                [list of CandidateInfo objects]
                              │
                              ▼
              for candidate in candidates:
                  ds = registry.fetch(candidate)   # runs data hard rules
                  if ds is None: continue
                  if a6_cross_duplicate.check(ds, pool) fails: continue  # A6
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
| S3 Data Quality   | 15 | composite: completeness, consistency, outliers (IsolationForest), constant features |
| S4 Data Leakage   | 20 | implemented — per-feature predictive-gap detection: worst vs median feature stat (Mann-Whitney / Spearman); low score when one feature sticks out far above the bulk (leak), high when signal is spread (biology). Optional group k-fold leak test is a future add-on |
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
| A3 Signal           | implemented (permutation RF on **adjusted** balanced accuracy / R² — chance-corrected so the floor means the same for binary, multiclass, and regression; TabPFN second-opinion for trivial signals) |
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

## Visualization

`stats.build_sankey(csv_path="rule_stats.csv")` renders the hard-rule funnel as
a Sankey diagram: each source flows source → A1 → A2 → A4 → A5 → A3 → Accepted,
with every rule's rejections draining into a single `Rejected` node. Counts are
shown in the node labels, and it writes `<csv>.html`. plotly is imported lazily
so the pipeline still imports without it (`kaleido` is only needed for PDF
export). Note: A6 cross-duplicate rejections are not recorded to `stats`, so
they do not appear in the Sankey.

## CRITIC Weight Validation

`pipeline/critic/` derives objective soft-rule weights from the score matrix and
compares them to the AHP baseline, run separately from the main pipeline:

- `critic.py` — the CRITIC method (min-max normalize → contrast intensity →
  conflict → informativeness → weights). Constant columns (e.g. S5) carry no
  signal and fall back to AHP.
- `score.py` — per-dataset CRITIC scores from the weight vector.
- `calibrate.py` — weight calibration helpers.

## Testing

Quick smoke test for any loader — runs `list_candidates` against the real API and stops after the first candidate that passes metadata hard rules. Swap `tcga_loader` for `openml_loader`, `geo_array_loader`, `kaggle_loader`, or `uci_loader` to test the others. (`geo_rnaseq_loader` reads local Parquet from `data/rnaseq/` instead of an API — point `GEO_RNASEQ_DIR` at your exports first.)

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
