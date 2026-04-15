# Pipeline Architecture

A visual overview of what each file does and how they work together.

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
│   fetch_all(list)    ──▶ loops fetch() over many candidates                  │
└─────────────────────────────────────────────────────────────────────────────┘
             │                        │                         │
             ▼                        ▼                         ▼
   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
   │ openml_loader.py │    │  tcga_loader.py  │    │ geo_array_loader.py  │
   │                  │    │                  │    │                      │
   │ • list_          │    │ • list_          │    │ • list_              │
   │   candidates()   │    │   candidates()   │    │   candidates()       │
   │ • fetch()        │    │ • fetch()        │    │ • fetch()            │
   │                  │    │                  │    │ • _parse_geoparse_   │
   │ Uses OpenML API  │    │ Uses GDC API     │    │   metadata()         │
   │ (target already  │    │ (target = sample │    │                      │
   │  defined)        │    │  type/vital)     │    │ Uses Entrez + GEO    │
   │                  │    │                  │    │ parse (builds target │
   │                  │    │                  │    │ from metadata text)  │
   └──────────────────┘    └──────────────────┘    └──────────────────────┘
             │                        │                         │
             │ each loader calls hard_rules.runner twice:       │
             │   1) run_metadata_checks() — cheap, before download
             │   2) run_data_checks()     — expensive, after download
             ▼                        ▼                         ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                  pipeline/hard_rules/runner.py                       │
   │                     ═══ RULE ORCHESTRATOR ═══                        │
   │                                                                      │
   │   _METADATA_CHECKS = [A1, A2, A5, A6]  ← cheap, metadata-only        │
   │   _DATA_CHECKS     = [A3, A5]          ← needs downloaded data       │
   │                                                                      │
   │   run_metadata_checks(**kwargs) → loops and calls each rule          │
   │   run_data_checks(X, y, ...)    → loops and calls each rule          │
   │   all_passed() / failed_rules() → helpers for checking results       │
   └─────────────────────────────────────────────────────────────────────┘
            │         │          │          │          │
            ▼         ▼          ▼          ▼          ▼
      ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
      │  a1_    │ │  a2_    │ │  a3_    │ │  a5_    │ │  a6_    │
      │ task_   │ │synthetic│ │ signal  │ │dimension│ │ licence │
      │ type.py │ │  .py    │ │  .py    │ │  .py    │ │  .py    │
      │         │ │         │ │         │ │         │ │         │
      │ classif │ │ MOCKUP  │ │ MOCKUP  │ │ N ≥1000 │ │ open    │
      │ /regr?  │ │ (stub)  │ │ (stub)  │ │ P≥10000 │ │ licence?│
      │         │ │         │ │         │ │ both!   │ │         │
      │ meta +  │ │ meta    │ │ data    │ │ meta +  │ │ meta    │
      │ data    │ │ only    │ │ only    │ │ data    │ │ only    │
      └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
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

## End-to-End Flow

```
   registry.list_candidates()
           │
           ├─▶ openml_loader.list_candidates() ──┐
           ├─▶ tcga_loader.list_candidates()  ───┤── each runs metadata
           └─▶ geo_loader.list_candidates()   ───┘    hard rules
                              │
                              ▼
                [list of CandidateInfo objects]
                              │
                              ▼
              registry.fetch_all(candidates)
                              │
                              ├─▶ downloads actual data
                              ├─▶ runs DATA-level hard rules (A3, A5)
                              └─▶ drops failures
                              │
                              ▼
                    [list of Dataset objects]
                              │
                              ▼
              (next: soft rules + CRITIC scoring)
```

## Key Patterns

- **Dispatcher pattern** — `registry.py` routes calls based on the `source` string. Adding a new data source = adding one entry to the `_LOADERS` dict.
- **Two-phase validation** — cheap metadata checks first, expensive data checks only on survivors.
- **Registry pattern (for rules)** — `runner.py` keeps a list of rule functions and loops over them. Adding a new rule = adding one line.
- **Protocol-based loaders** — every loader must expose `list_candidates()` + `fetch()`. That's what makes the dispatcher work.

## Currently Mocked Rules

| Rule | Status | Purpose |
|------|--------|---------|
| A2   | Stub (always passes) | Detect synthetic/artificial datasets |
| A3   | Stub (always passes) | Verify predictive signal via RandomForest CV |

Real implementations are planned — the stubs keep the runner working without failing datasets.

## Testing

Quick smoke test for any loader — runs `list_candidates` against the real API and stops after the first candidate that passes metadata hard rules. Swap `tcga_loader` for `openml_loader` or `geo_array_loader` to test the others.

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
