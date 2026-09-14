"""GEO microarray loader.

Implements the dataset loader for NCBI Gene Expression Omnibus (GEO) datasets
based on microarray expression profiling. Returns the raw expression data
(probes x samples) along with tumor/normal labels extracted from metadata.

This is a POC: focuses on tumor vs normal classification studies, which are
common in GEO and useful for biomedical ML benchmarks.
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import GEOparse
import numpy as np
import pandas as pd
import requests
from Bio import Entrez
from pipeline import stats

from pipeline.config import (
    ENTREZ_EMAIL,
    MAX_FEATURES,
    NCBI_API_KEY,
    REQUEST_DELAY,
)
from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules

logger = logging.getLogger(__name__)

# tell Entrez who we are (NCBI requires this for all requests)
Entrez.email = ENTREZ_EMAIL
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY

# NOTE: this is a LOOSE pre-filter, NOT the real benchmark threshold!
# The actual hard rule is MIN_ROWS = 1000 (from config.py, enforced by A4).
# We use this loose 50 here just to skip obviously tiny GEO studies before
# doing the expensive GEOparse download. Studies with 50-999 samples will
# still be parsed here, and then rejected later by the A4 hard rule.
MIN_SAMPLES = 500  # GEO studies are typically smaller

# how many SOFT files to download at once. The scrape is network-bound, one study
# takes ~1 min, so this sets the speedup. Lower it if NCBI starts refusing connections.
DOWNLOAD_WORKERS = 8

# where we cache GEOparse downloads so we don't re-download the same GSE
CACHE_DIR = os.path.join(os.environ.get("PIPELINE_CACHE", "/tmp"), "geoparse_cache")

# regex patterns for spotting tumor vs normal labels in sample metadata.
# we look for common keywords. this is rough but works for most GEO studies.
TUMOR_PATTERN = re.compile(
    r"\b(tumor|tumour|cancer|carcinoma|malignant|primary|metastas"
    r"|adenocarcinoma|neoplasm|cancerous)\b",
    re.IGNORECASE,
)
NORMAL_PATTERN = re.compile(
    r"\b(normal|healthy|control|benign|adjacent|non.tumor|non.tumour"
    r"|non.cancerous|non.malignant)\b",
    re.IGNORECASE,
)

# Keywords for scoring characteristics as classification targets
PHENOTYPE_KEYWORDS = [
    "disease", "condition", "phenotype", "status", "type", "subtype",
    "stage", "grade", "class", "diagnosis", "outcome", "response",
    "treatment", "pathology", "clinical", "category", "group",
]
METADATA_KEYWORDS = [
    "age", "gender", "sex", "batch", "plate", "technician", "date",
    "passage", "lot", "replicate", "time.point", "timepoint", "time",
    "sample.id", "id", "name", "donor", "patient", "individual",
]

MIN_CLASSES, MAX_CLASSES = 2, 50


# GEO search queries (we loop over all of them and combine the results).
# Each query filters for human studies large enough to satisfy the A4 hard
# rule (N>=500). We try several topical contrast pairs to get enough variety;
# duplicates across queries are dropped inside search_studies().
def geo_query(data_type, topic):
    return ('"Homo sapiens"[ORGN] AND gse[ETYP]'
            f' AND {data_type}[DataSet Type]'
            ' AND 500:1000000[Number of Samples]'
            f' AND {topic}')


GEO_QUERIES = [geo_query('"Expression profiling by array"', topic) for topic in (
    '("tumor"[All Fields] AND "normal"[All Fields])',
    '("cancer"[All Fields] AND "control"[All Fields])',
    '("disease"[All Fields] AND "healthy"[All Fields])',
    '("survival"[All Fields])',
    '("prognosis"[All Fields])',
    '("subtype"[All Fields])',
    '("differentiation"[All Fields])',
)] + [
    # non-genetic GEO data: proteomics and metabolomics
    geo_query('("Protein expression profiling" OR "proteomics")',
              '("disease"[All Fields] OR "cancer"[All Fields])'),
    geo_query('("Metabolite profiling" OR "metabolomics")',
              '("disease"[All Fields] OR "biomarker"[All Fields])'),
]


# ====== sample metadata: characteristics -> a classification target ======


def collect_characteristics(gse):
    # walk every sample (GSM) and split its "key: value" characteristic lines.
    # returns the per-sample view (first value of each key wins) and, for scoring,
    # every value seen per key across the study.
    per_sample = {}
    values = {}
    for gsm_id, gsm in gse.gsms.items():
        per_sample[gsm_id] = {}
        for line in gsm.metadata.get("characteristics_ch1", []):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip().lower(), value.strip()
            per_sample[gsm_id].setdefault(key, value)
            values.setdefault(key, []).append(value)
    return per_sample, values


def score_characteristic(name, values):
    # Score how good a characteristic is as a classification target.
    # Good targets have 2-50 unique values, balanced distribution, high coverage,
    # and phenotype-like names (not metadata like age/batch).
    filled = [v for v in values if v and str(v).strip()]
    n_unique = len(set(filled))
    if n_unique < MIN_CLASSES or n_unique > MAX_CLASSES:
        return 0.0

    counts = pd.Series(filled).value_counts()
    balance = counts.min() / counts.max()          # penalize 95/5 splits
    coverage = len(filled) / len(values)           # share of samples that have it
    score = n_unique * balance * coverage

    name = name.lower()
    if any(kw in name for kw in PHENOTYPE_KEYWORDS):
        score *= 2.0
    if any(kw in name for kw in METADATA_KEYWORDS):
        score *= 0.1
    return score


def best_target(values):
    # highest-scoring characteristic, or None if nothing scores above zero.
    # ties keep the first one seen, which is the order GEO lists them in.
    if not values:
        return None
    best = max(values, key=lambda name: score_characteristic(name, values[name]))
    return best if score_characteristic(best, values[best]) > 0 else None


def has_tumor_and_normal(gse):
    # tumor/normal only works if BOTH show up in the study. an all-tumor cohort
    # otherwise gets "tumor" for every sample -> constant target/ no viable dataset
    text = " ".join(" ".join(gsm.metadata.get("characteristics_ch1", [])
                             + gsm.metadata.get("source_name_ch1", []))
                    for gsm in gse.gsms.values())
    return bool(TUMOR_PATTERN.search(text) and NORMAL_PATTERN.search(text))


def sample_labels(gse, per_sample, target):
    # one label per sample: tumor/normal regex first, else the best characteristic
    tumor_normal = has_tumor_and_normal(gse)
    labels = {}
    for gsm_id, gsm in gse.gsms.items():
        text = " ".join(gsm.metadata.get("characteristics_ch1", [])
                        + gsm.metadata.get("source_name_ch1", []))
        if tumor_normal and TUMOR_PATTERN.search(text):
            labels[gsm_id] = "tumor"
        elif tumor_normal and NORMAL_PATTERN.search(text):
            labels[gsm_id] = "normal"
        elif target and per_sample[gsm_id].get(target):
            labels[gsm_id] = per_sample[gsm_id][target]
    return labels


# ====== Entrez search + GEOparse metadata (the slow part) ======


def esearch(query, retmax=2000):
    # search GEO via Entrez, returns a list of UIDs (study IDs)
    handle = Entrez.esearch(db="gds", term=query, retmax=retmax, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def esummary(uids):
    # summary records for a batch of UIDs
    handle = Entrez.esummary(db="gds", id=",".join(uids), retmode="xml")
    results = Entrez.read(handle)
    handle.close()
    return results


def search_studies():
    # every unique GSE the queries return, as (accession, title, n_samples).
    # a generator, so the caller stops it early instead of running all queries.
    seen = set()
    for query in GEO_QUERIES:
        logger.info("  Query: %s...", query[:70])
        try:
            uids = esearch(query)
        except Exception as e:
            logger.warning("  Search failed: %s", e)
            continue
        logger.info("  %d hits", len(uids))

        # fetch summaries in batches of 100 to stay under Entrez rate limits
        for i in range(0, len(uids), 100):
            try:
                summaries = esummary(uids[i:i + 100])
            except Exception as e:
                logger.warning("  Summary batch failed: %s", e)
                continue
            for record in summaries:
                # GSE = GEO Series (one study). Skip anything else, and repeats.
                accession = str(record.get("Accession", ""))
                if not accession.startswith("GSE") or accession in seen:
                    continue
                seen.add(accession)

                # cheap pre-filter: throw out tiny studies right away so we
                # don't waste a GEOparse download on them. The REAL N>=1000
                # check happens later inside hard_rules.run_metadata_checks (A4).
                n_samples = int(record.get("n_samples", 0) or 0)
                if n_samples < MIN_SAMPLES:
                    continue
                yield accession, str(record.get("title", "")), n_samples
            time.sleep(REQUEST_DELAY)


def study_metadata(accession):
    # GEOparse actually downloads the SOFT file here, so this is the expensive call.
    # Returns None when the download fails or the study has no usable target.
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        gse = GEOparse.get_GEO(accession, destdir=CACHE_DIR, silent=True)
    except Exception as e:
        logger.debug("GEOparse failed for %s: %s", accession, e)
        return None

    # platform + feature count come from the GPL (GEO Platform) table.
    # a study can have multiple platforms; we just take the first one.
    platform, n_features = "", None
    if gse.gpls:
        platform = list(gse.gpls.keys())[0]
        table = gse.gpls[platform].table
        if table is not None and not table.empty:
            n_features = len(table)

    _, values = collect_characteristics(gse)
    if has_tumor_and_normal(gse):
        class_labels = ["normal", "tumor"]
    else:
        target = best_target(values)
        class_labels = sorted({v for v in values[target] if v}) if target else []
        if not MIN_CLASSES <= len(class_labels) <= MAX_CLASSES:
            class_labels = []

    if not class_labels:
        return None
    return {
        "platform": platform,
        "n_features": n_features,
        "n_classes": len(class_labels),
        "class_labels": ", ".join(class_labels[:20]),
    }


# ====== public entry points (list_candidates + fetch) ======


def check_study(accession, title, n_samples, meta):
    # metadata hard rules for one scraped study, returns a CandidateInfo or None
    if meta is None:
        logger.debug("  %s: no usable target, skipping", accession)
        return None

    n_features = meta["n_features"]
    if n_features is not None and n_features > MAX_FEATURES:
        logger.debug("  %s: P=%d > %d, skipping (won't fit in RAM)",
                     accession, n_features, MAX_FEATURES)
        return None

    results = hard_rules.run_metadata_checks(
        n_samples=n_samples,
        n_features=n_features,
        task_type="classification",
        licence="public-domain",
        source="geo_array",
        name=title,
    )
    if not hard_rules.all_passed(results):
        stats.record(f"GEO-{accession}", "geo_array", title, results, "biological")
        return None

    logger.info("  %s: PASS N=%d P=%s task=classification", accession, n_samples, n_features)
    return CandidateInfo(
        id=f"GEO-{accession}",
        source="geo_array",
        name=title[:80],
        n_samples=n_samples,
        n_features=n_features,
        task_type="classification",
        licence="public-domain",
        url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
        metadata={
            "platform": meta["platform"],
            "n_classes": meta["n_classes"],
            "class_labels": meta["class_labels"],
            "organism": "Homo sapiens",
        },
        domain="biological",
    )


def list_candidates(max_candidates=50):
    # search GEO, then run the cheap hard rules on whatever survives the scrape
    logger.info("Searching GEO for microarray expression datasets...")
    candidates = []
    studies = search_studies()

    # the GEOparse download is the bottleneck (~1 min per study), so a batch of
    # them runs at once and the rules are applied to the results in order.
    # A batch downloads a few studies more than needed, which is the trade.
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        while len(candidates) < max_candidates:
            batch = list(islice(studies, DOWNLOAD_WORKERS))
            if not batch:
                break  # no studies left to look at
            metas = pool.map(lambda study: study_metadata(study[0]), batch)
            candidates += [c for c in (check_study(*study, meta)
                                       for study, meta in zip(batch, metas)) if c]

    logger.info("GEO microarray: %d candidates after metadata filters", len(candidates))
    return candidates[:max_candidates]


def series_matrix_url(accession):
    # GEO groups studies into sub-folders by accession prefix:
    #     GSE12345  -> .../GSE12nnn/GSE12345/matrix/
    #     GSE5      -> .../GSEnnn/GSE5/matrix/
    num = accession[3:]  # strip "GSE"
    folder = f"GSE{num[:-3] if len(num) > 3 else ''}nnn"
    return (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{folder}/{accession}"
            f"/matrix/{accession}_series_matrix.txt.gz")


def expression_from_series_matrix(accession):
    # `pd.read_csv(comment="!")` skips every header line (all start with `!`)
    # and the begin/end markers, leaving just the column header + data rows.
    # Returns probes x samples, transposed below.
    cache_path = os.path.join(CACHE_DIR, f"{accession}_series_matrix.txt.gz")
    if not os.path.exists(cache_path):
        url = series_matrix_url(accession)
        logger.info("  Fetching %s", url)
        with requests.get(url, timeout=180, stream=True) as response:
            response.raise_for_status()
            with open(cache_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=65536):
                    fh.write(chunk)
    df = pd.read_csv(cache_path, sep="\t", comment="!", index_col=0,
                     compression="gzip", low_memory=False)
    # Cast to float32 column-by-column so we don't carry a float64 copy alongside the result.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    return df.T


def expression_from_soft(gse):
    # Build X as float32 row-by-row to keep peak memory ~half of what
    # pd.concat(...).apply(pd.to_numeric) would use. Returns None when the
    # SOFT file ships without per-sample values (common for large studies).
    ref_index = None
    arrays: list[np.ndarray] = []
    sample_ids: list[str] = []

    for gsm_id, gsm in gse.gsms.items():
        table = gsm.table
        if (table is None or table.empty
                or "VALUE" not in table.columns or "ID_REF" not in table.columns):
            continue
        series = pd.to_numeric(table.set_index("ID_REF")["VALUE"], errors="coerce")
        if ref_index is None:
            ref_index = series.index
        arrays.append(series.reindex(ref_index).to_numpy(dtype=np.float32, copy=False))
        sample_ids.append(gsm_id)

    if not arrays:
        return None
    return pd.DataFrame(np.vstack(arrays), index=sample_ids, columns=ref_index)


def fetch(candidate):
    # Download GEO expression data + labels and run data-level hard rules.
    accession = candidate.id.removeprefix("GEO-")
    logger.info("Downloading GEO %s...", accession)

    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        gse = GEOparse.get_GEO(accession, destdir=CACHE_DIR, silent=True)
    except Exception as e:
        logger.warning("Failed to download %s: %s", accession, e)
        return None, []

    if not gse.gpls:
        logger.warning("%s: no platform found", accession)
        return None, []

    # labels come from SOFT metadata, which is there even when the values are stripped
    per_sample, values = collect_characteristics(gse)
    y = pd.Series(sample_labels(gse, per_sample, best_target(values)), name="target")

    X = expression_from_soft(gse)
    if X is None:
        logger.info("%s: per-GSM tables empty, falling back to Series Matrix file", accession)
        try:
            X = expression_from_series_matrix(accession)
        except Exception as e:
            logger.warning("%s: Series Matrix fallback failed: %s", accession, e)
            return None, []
        if X is None or X.empty:
            logger.warning("%s: no expression data found", accession)
            return None, []

    shared = X.index.intersection(y.index)
    if len(shared) < 10:
        logger.warning("%s: only %d labeled samples", accession, len(shared))
        return None, []
    X, y = X.loc[shared], y.loc[shared]

    result, data_results = hard_rules.run_hard_rules(X, y, candidate)
    if result is None:
        return None, data_results
    _, task_type = result

    logger.info("GEO %s: loaded %d samples x %d features, task=%s",
                accession, X.shape[0], X.shape[1], task_type)

    return Dataset(
        id=candidate.id,
        source="geo_array",
        name=candidate.name,
        X=X,
        y=y,
        task_type=task_type,
        metadata={
            **candidate.metadata,
            "licence": "public-domain",
            "url": candidate.url,
        },
        domain="biological",
    ), data_results
