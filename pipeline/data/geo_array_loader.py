"""GEO microarray loader.

Implements the dataset loader for NCBI Gene Expression Omnibus (GEO) datasets
based on microarray expression profiling. Returns the raw expression data
(probes x samples) along with tumor/normal labels extracted from metadata.

This is a POC: focuses on tumor vs normal classification studies, which are
common in GEO and useful for biomedical ML benchmarks.
"""

#TODO implement parallelization for the GEO requeuts for faster candidate listing -> bottleneck
#right now code is waiting all the time for answer of GEO request while doing nothing
import logging
import os
import re
import time

import GEOparse
import numpy as np
import pandas as pd
import requests
from Bio import Entrez
from pipeline import stats

from pipeline.config import (
    ENTREZ_BASE_URL,
    ENTREZ_EMAIL,
    MAX_FEATURES,
    MIN_FEATURES,
    NCBI_API_KEY,
    REQUEST_DELAY,
)
from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules
from pipeline.hard_rules.base import RuleResult

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

# GEO search queries (we loop over all of them and combine the results).
# Each query filters for human microarray studies large enough to satisfy
# the A4 hard rule (N>=500). We try several topical contrast pairs to get
# enough variety; duplicates across queries are dropped via the `seen` set.
_QUERY_BASE = (
    '"Homo sapiens"[ORGN] AND gse[ETYP]'
    ' AND "Expression profiling by array"[DataSet Type]'
    ' AND 500:1000000[Number of Samples]'
)
GEO_QUERIES = [
    f'{_QUERY_BASE} AND ("tumor"[All Fields] AND "normal"[All Fields])',
    f'{_QUERY_BASE} AND ("cancer"[All Fields] AND "control"[All Fields])',
    f'{_QUERY_BASE} AND ("disease"[All Fields] AND "healthy"[All Fields])',
    f'{_QUERY_BASE} AND ("survival"[All Fields])',
    f'{_QUERY_BASE} AND ("prognosis"[All Fields])',
    f'{_QUERY_BASE} AND ("subtype"[All Fields])',
    f'{_QUERY_BASE} AND ("differentiation"[All Fields])',
]


# Entrez wrappers.
# Entrez is NCBI's search API. We use it to find GEO study IDs (UIDs) and
# then fetch their summary metadata before downloading the full thing.


def _esearch(query, retmax=2000):
    """Search GEO via Entrez. Returns a list of UIDs (study IDs)."""
    handle = Entrez.esearch(db="gds", term=query, retmax=retmax, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def _esummary_batch(uids):
    """Fetch summary records for a batch of UIDs. Returns a list of dicts."""
    handle = Entrez.esummary(db="gds", id=",".join(uids), retmode="xml")
    results = Entrez.read(handle)
    handle.close()
    return results


# helper to clean up sample labels before comparing them

def _normalize_label(raw_label):
    """takes a raw label string and normalizes it for comparison."""
    label = raw_label.strip()
    label = label.lower()
    label = label.replace(" ", "_")
    # remove duplicates from label list
    parts = label.split(",")
    unique = []
    for p in parts:
        if p not in unique:
            unique.append(p)
    label = ",".join(unique)
    # if label is empty after cleaning, use "unknown"
    if label == "":
        label = "unknown"
    return label


# metadata extraction via GEOparse (the slow part)


def _parse_geoparse_metadata(accession):
    """Use GEOparse to pull platform info, feature count, and tumor/normal labels.

    This is the expensive call (it actually downloads the metadata file).
    """
    # start with default values, fill in what we can find
    result = {
        "task_type": "unknown",
        "n_classes": None,
        "class_labels": "",
        "platform": "",
        "n_features": None,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        gse = GEOparse.get_GEO(accession, destdir=CACHE_DIR, silent=True)
    except Exception as e:
        logger.debug("GEOparse failed for %s: %s", accession, e)
        return result

    # platform + feature count come from the GPL (GEO Platform) table.
    # a study can have multiple platforms; we just take the first one.
    if gse.gpls:
        gpl_name = list(gse.gpls.keys())[0]
        result["platform"] = gpl_name
        gpl = gse.gpls[gpl_name]
        if gpl.table is not None and not gpl.table.empty:
            result["n_features"] = len(gpl.table)

    # walk through every sample (GSM) and count tumor vs normal matches
    tumor_count = 0
    normal_count = 0
    all_values = set()  # collect all unique characteristic values seen

    for gsm in gse.gsms.values():
        # each sample has free-form "characteristics" and a "source_name"
        chars = gsm.metadata.get("characteristics_ch1", [])
        source = " ".join(gsm.metadata.get("source_name_ch1", []))
        combined = " ".join(chars) + " " + source

        # regex hit for tumor or normal?
        if TUMOR_PATTERN.search(combined):
            tumor_count += 1
        if NORMAL_PATTERN.search(combined):
            normal_count += 1

        # characteristics are formatted like "key: value"
        # grab the value part so we can see how many unique labels exist
        for char in chars:
            if ":" in char:
                val = char.split(":", 1)[1].strip().lower()
                if val:
                    all_values.add(val)

    # POC scope: only tumor vs normal for now, but can be extended later
    if tumor_count > 0 and normal_count > 0:
        # both present -> 2-class classification
        result["task_type"] = "classification"
        result["n_classes"] = 2
        result["class_labels"] = "normal, tumor"
    elif 2 <= len(all_values) <= 50:
        # no clean tumor/normal signal but the characteristics have a reasonable
        # number of unique values -> could still be classification
        result["task_type"] = "classification"
        result["n_classes"] = len(all_values)
        result["class_labels"] = ", ".join(sorted(all_values)[:20])

    return result


# ====== public entry points (list_candidates + fetch) ======


def list_candidates(max_candidates=50):
    """Search GEO for microarray expression datasets with tumor/normal contrast.

    Flow:
      1. Run each Entrez search query -> get lots of UIDs
      2. Fetch summaries in batches (fast, metadata only)
      3. Pre-filter by sample count (cheap)
      4. For survivors: GEOparse download (expensive) to get detailed metadata
      5. Run hard rule checks; if they pass, build a CandidateInfo
    """
    logger.info("Searching GEO for microarray expression datasets...")
    seen = set()           # GSE accessions we've already processed (avoid duplicates across queries)
    candidates = []

    # loop over each search query
    for query in GEO_QUERIES:
        if len(candidates) >= max_candidates:
            break

        logger.info("  Query: %s...", query[:70])
        try:
            uids = _esearch(query, retmax=2000)
        except Exception as e:
            logger.warning("  Search failed: %s", e)
            continue

        logger.info("  %d hits", len(uids))

        # fetch summaries in batches of 100 to stay under Entrez rate limits
        for i in range(0, len(uids), 100):
            if len(candidates) >= max_candidates:
                break

            batch = uids[i:i + 100]
            try:
                summaries = _esummary_batch(batch)
            except Exception as e:
                logger.warning("  Summary batch failed: %s", e)
                continue

            # process each summary in this batch
            for record in summaries:
                if len(candidates) >= max_candidates:
                    break

                # GSE = GEO Series (one study). Skip anything that isn't a GSE or we've seen.
                accession = str(record.get("Accession", ""))
                if not accession.startswith("GSE") or accession in seen:
                    continue
                seen.add(accession)

                # how many samples does this study have?
                n_samples = int(record.get("n_samples", 0) or 0)

                # cheap pre-filter: throw out tiny studies right away so we
                # don't waste a GEOparse download on them. The REAL N>=1000
                # check happens later inside hard_rules.run_metadata_checks (A4).
                if n_samples < MIN_SAMPLES:
                    continue

                title = str(record.get("title", ""))

                # deep metadata via GEOparse (this is the slow call)
                logger.info("  Parsing %s (N=%d)...", accession, n_samples)
                meta = _parse_geoparse_metadata(accession)

                # if we couldn't figure out a task type, skip
                if meta["task_type"] == "unknown":
                    logger.debug("  %s: no tumor/normal contrast, skipping", accession)
                    fail = RuleResult(rule="pre-filter", passed=False, reason="no tumor/normal contrast")
                    stats.record(accession, "geo_array", title, [fail])
                    continue

                # feature count check (if we have it)
                n_features = meta["n_features"]
                if n_features is not None and n_features < MIN_FEATURES:
                    logger.debug("  %s: P=%d < %d, skipping", accession, n_features, MIN_FEATURES)
                    reason = "P=" + str(n_features) + " < " + str(MIN_FEATURES)
                    fail = RuleResult(rule="pre-filter", passed=False, reason=reason)
                    stats.record(accession, "geo_array", title, [fail])
                    continue
                if n_features is not None and n_features > MAX_FEATURES:
                    logger.debug("  %s: P=%d > %d, skipping (won't fit in RAM)",
                                 accession, n_features, MAX_FEATURES)
                    reason = "P=" + str(n_features) + " > " + str(MAX_FEATURES) + " (RAM cap)"
                    fail = RuleResult(rule="pre-filter", passed=False, reason=reason)
                    stats.record(accession, "geo_array", title, [fail])
                    continue

                # run the central metadata hard rules
                results = hard_rules.run_metadata_checks(
                    n_samples=n_samples,
                    n_features=n_features,
                    task_type=meta["task_type"],
                    licence="public-domain",
                    source="geo_array",
                    name=title,
                )
                stats.record(accession, "geo_array", title, results)
                if not hard_rules.all_passed(results):
                    continue

                # passed everything -> build a CandidateInfo and add it
                candidates.append(CandidateInfo(
                    id=accession,
                    source="geo_array",
                    name=title[:80],
                    n_samples=n_samples,
                    n_features=n_features,
                    task_type=meta["task_type"],
                    licence="public-domain",
                    url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
                    metadata={
                        "platform": meta["platform"],
                        "n_classes": meta["n_classes"],
                        "class_labels": meta["class_labels"],
                        "organism": "Homo sapiens",
                    },
                    domain="biological",
                ))

                logger.info(
                    "  %s: PASS N=%d P=%s task=%s",
                    accession, n_samples, n_features, meta["task_type"],
                )
                time.sleep(REQUEST_DELAY)

            time.sleep(REQUEST_DELAY)

    logger.info("GEO microarray: %d candidates after metadata filters", len(candidates))
    

    return candidates


def _series_matrix_url(accession):
    """Build the NCBI FTP URL for {accession}_series_matrix.txt.gz.

    GEO groups studies into sub-folders by accession prefix:
        GSE12345  -> .../GSE12nnn/GSE12345/matrix/
        GSE5      -> .../GSEnnn/GSE5/matrix/
    """
    num = accession[3:]  # strip "GSE"
    folder_prefix = num[:-3] if len(num) > 3 else ""
    folder = f"GSE{folder_prefix}nnn"
    filename = f"{accession}_series_matrix.txt.gz"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{folder}/{accession}/matrix/{filename}"


def _expression_from_series_matrix(accession):
    """Download the Series Matrix file and parse the expression block.

    `pd.read_csv(comment="!")` skips every header line (all start with `!`)
    and the begin/end markers, leaving just the column header + data rows.
    Returns probes x samples df, transposed below.
    """
    cache_path = os.path.join(CACHE_DIR, f"{accession}_series_matrix.txt.gz")
    if not os.path.exists(cache_path):
        url = _series_matrix_url(accession)
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


def fetch(candidate):
    """Download GEO expression data + labels and run data-level hard rules.

    Tries per-GSM tables (from the SOFT file) first. Some studies don't
    embed expression values there, so falls back to downloading the
    Series Matrix file directly from NCBI FTP.
    """
    accession = candidate.id
    logger.info("Downloading GEO %s...", accession)

    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        gse = GEOparse.get_GEO(accession, destdir=CACHE_DIR, silent=True)
    except Exception as e:
        logger.warning("Failed to download %s: %s", accession, e)
        return None

    if not gse.gpls:
        logger.warning("%s: no platform found", accession)
        return None

    # Always try to derive labels from SOFT metadata (it's there even when values are stripped)
    # Build X as float32 row-by-row to keep peak memory ~half of what
    # pd.concat(...).apply(pd.to_numeric) would use.
    ref_index = None
    sample_arrays: list[np.ndarray] = []
    sample_ids: list[str] = []
    labels: dict[str, str] = {}

    for gsm_id, gsm in gse.gsms.items():
        table = gsm.table
        if (table is not None and not table.empty
                and "VALUE" in table.columns and "ID_REF" in table.columns):
            series = pd.to_numeric(
                table.set_index("ID_REF")["VALUE"], errors="coerce",
            )
            if ref_index is None:
                ref_index = series.index
            sample_arrays.append(
                series.reindex(ref_index).to_numpy(dtype=np.float32, copy=False)
            )
            sample_ids.append(gsm_id)

        chars = gsm.metadata.get("characteristics_ch1", [])
        source = " ".join(gsm.metadata.get("source_name_ch1", []))
        combined = " ".join(chars) + " " + source
        if TUMOR_PATTERN.search(combined):
            labels[gsm_id] = "tumor"
        elif NORMAL_PATTERN.search(combined):
            labels[gsm_id] = "normal"
        else:
            for char in chars:
                if ":" in char:
                    labels[gsm_id] = char.split(":", 1)[1].strip()
                    break

    if sample_arrays:
        X = pd.DataFrame(
            np.vstack(sample_arrays), index=sample_ids, columns=ref_index,
        )
    else:
        # SOFT file has no per-GSM tables; download the Series Matrix file instead
        logger.info("%s: per-GSM tables empty, falling back to Series Matrix file", accession)
        try:
            X = _expression_from_series_matrix(accession)
        except Exception as e:
            logger.warning("%s: Series Matrix fallback failed: %s", accession, e)
            return None
        if X is None or X.empty:
            logger.warning("%s: no expression data found", accession)
            return None

    y = pd.Series(labels, name="target")

    shared = X.index.intersection(y.index)
    if len(shared) < 10:
        logger.warning("%s: only %d labeled samples", accession, len(shared))
        return None

    X = X.loc[shared]
    y = y.loc[shared]

    data_results = hard_rules.run_data_checks(X, y, task_type=candidate.task_type)
    failed = hard_rules.failed_rules(data_results)
    if failed:
        reasons = ", ".join(f"{r.rule}: {r.reason}" for r in failed)
        logger.info("GEO %s discarded (data check): %s", accession, reasons)
        return None

    task_type = hard_rules.inferred_task_type(data_results) or candidate.task_type
    stats.record(accession, "geo_array", candidate.name, data_results)

    logger.info(
        "GEO %s: loaded %d samples x %d features, task=%s",
        accession, X.shape[0], X.shape[1], task_type,
    )

    return Dataset(
        id=f"GEO-{accession}",
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
    )
