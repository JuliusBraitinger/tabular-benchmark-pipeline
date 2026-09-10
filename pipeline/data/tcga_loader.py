# TCGA loader - one dataset per (project, data type) from the GDC API.
# X = per-sample expression / miRNA files stacked into samples x features (+ clinical columns),
# y = sample_type of the case the file belongs to (Primary Tumor, Solid Tissue Normal, ...).

import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import GDC_BASE_URL, REQUEST_DELAY
from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules
from pipeline import stats

logger = logging.getLogger(__name__)

MIN_CASES = 500
MAX_FILES = 3000
CACHE_DIR = Path(os.environ.get("PIPELINE_DATA", "data")) / "cache" / "tcga"

# data_type -> (workflow, feature column, value column) of the per-sample TSV.
# Copy Number Segment is not here: segment files are variable-length intervals, not a fixed
# feature vector, so there is nothing to stack into a matrix.
WORKFLOWS = {
    "Gene Expression Quantification": ("STAR - Counts", "gene_name", "unstranded"),
    "miRNA Expression Quantification": ("BCGSC miRNA Profiling", "miRNA_ID", "read_count"),
}

CLINICAL_FIELDS = ("submitter_id,demographic.vital_status,demographic.gender,"
                   "demographic.age_at_index,diagnoses.ajcc_pathologic_stage")


def gdc(endpoint, payload):
    resp = requests.post(f"{GDC_BASE_URL}/{endpoint}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def query_files(data_type, project_id=None, size=1, fields="file_id"):
    # open-access files of one data type, optionally inside one project
    where = {"data_type": data_type,
             "analysis.workflow_type": WORKFLOWS[data_type][0],
             "access": "open"}
    if project_id:
        where["cases.project.project_id"] = project_id
    data = gdc("files", {
        "filters": {"op": "and", "content": [
            {"op": "=", "content": {"field": f, "value": v}} for f, v in where.items()]},
        "fields": fields,
        "size": size,
        "format": "json",
    })
    return data["pagination"]["total"], data["hits"]


def parse(text, data_type):
    # one per-sample TSV -> Series feature -> value. The STAR summary rows (N_unmapped etc)
    # have no gene_name, so they fall out with the NaN index.
    _, index_col, value_col = WORKFLOWS[data_type]
    frame = pd.read_csv(io.StringIO(text), sep="\t", comment="#",
                        usecols=[index_col, value_col], index_col=index_col)
    values = pd.to_numeric(frame.squeeze("columns"), errors="coerce")
    values = values[values.index.notna()].dropna()
    # gene_name is not unique in GENCODE (PAR_Y copies, same symbol twice), and a feature
    # matrix cannot have duplicate columns, so the last row of each symbol wins
    return values[~values.index.duplicated(keep="last")]


def download_one(file_id, data_type):
    resp = requests.get(f"{GDC_BASE_URL}/data/{file_id}", timeout=60)
    resp.raise_for_status()
    return parse(resp.text, data_type)


@lru_cache(maxsize=None)
def feature_count(data_type):
    # same pipeline everywhere, so the feature list is the same in every project: count once
    _, hits = query_files(data_type)
    return len(download_one(hits[0]["file_id"], data_type)) if hits else 0


def list_candidates(max_candidates=100):
    projects = gdc("projects", {
        "filters": {"op": "in", "content": {"field": "program.name", "value": ["TCGA"]}},
        "fields": "project_id,name,primary_site,summary.case_count",
        "size": 100,
        "format": "json",
    })["hits"]
    logger.info("Found %d TCGA projects", len(projects))

    candidates = []
    for proj in projects:
        if len(candidates) >= max_candidates:
            break
        project_id = proj["project_id"]
        n_cases = int((proj.get("summary") or {}).get("case_count", 0))
        if n_cases < MIN_CASES:
            logger.debug("%s: N=%d < %d, skipping", project_id, n_cases, MIN_CASES)
            continue

        for data_type in WORKFLOWS:
            n_files, _ = query_files(data_type, project_id)
            time.sleep(REQUEST_DELAY)
            if not n_files:
                continue

            n_features = feature_count(data_type)
            cand_id = f"{project_id}_{data_type.replace(' ', '-')}"
            name = f"{proj.get('name', '')} - {data_type}"

            # A4 skipped: TCGA is exempt from the size filter
            results = hard_rules.run_metadata_checks(
                n_samples=n_cases, n_features=n_features, task_type="classification",
                licence="public", source="tcga", name=name, skip=("A4",))
            if not hard_rules.all_passed(results):
                stats.record(cand_id, "tcga", name, results, "biomedical")
                continue

            candidates.append(CandidateInfo(
                id=cand_id,
                source="tcga",
                name=name,
                n_samples=n_cases,
                n_features=n_features,
                task_type="classification",
                licence="public",
                url=f"https://portal.gdc.cancer.gov/projects/{project_id}",
                metadata={"project_id": project_id,
                          "data_type": data_type,
                          "workflow_type": WORKFLOWS[data_type][0],
                          "primary_site": ", ".join(proj.get("primary_site", []))},
                domain="biomedical"))
            logger.info("%s %s: N=%d P=%d files=%d", project_id, data_type, n_cases,
                        n_features, n_files)

    logger.info("TCGA: %d candidates after metadata filters", len(candidates))
    return candidates[:max_candidates]


def download_matrix(hits, data_type):
    # download every sample file in parallel, indexed by file_id like the parquet cache expects
    def one(hit):
        samples = ((hit.get("cases") or [{}])[0].get("samples") or [{}])
        try:
            return hit["file_id"], download_one(hit["file_id"], data_type), samples[0].get("sample_type", "")
        except Exception as e:
            logger.debug("Failed to download file %s: %s", hit["file_id"], e)
            return hit["file_id"], None, ""

    with ThreadPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(one, hits[:MAX_FILES]))

    # a file without data or without a sample type cannot become a labelled row
    rows = [r for r in rows if r[1] is not None and r[2]]
    logger.info("  downloaded %d of %d files", len(rows), len(hits[:MAX_FILES]))
    if not rows:
        return None, None
    X = pd.DataFrame({fid: series for fid, series, _ in rows}).T
    y = pd.Series({fid: label for fid, _, label in rows}, name="target").loc[X.index]
    return X, y


def fetch_clinical(project_id):
    # vital status / gender / age / stage per case, keyed by case submitter_id
    try:
        hits = gdc("cases", {
            "filters": {"op": "=", "content": {"field": "project.project_id", "value": project_id}},
            "fields": CLINICAL_FIELDS,
            "size": 5000,
            "format": "json",
        })["hits"]
    except Exception as e:
        logger.warning("Failed to fetch clinical for %s: %s", project_id, e)
        return {}

    out = {}
    for case in hits:
        if not case.get("submitter_id"):
            continue
        demo = case.get("demographic") or {}
        diag = (case.get("diagnoses") or [{}])[0] or {}
        out[case["submitter_id"]] = {
            "clinical_vital_status": demo.get("vital_status"),
            "clinical_gender": demo.get("gender"),
            "clinical_age": demo.get("age_at_index"),
            "clinical_stage": diag.get("ajcc_pathologic_stage"),
        }
    return out


def fetch(candidate):
    project_id = candidate.metadata["project_id"]
    data_type = candidate.metadata["data_type"]

    # file list is needed either way: for downloading, and to map file_id -> case
    _, hits = query_files(data_type, project_id, size=MAX_FILES,
                          fields="file_id,cases.submitter_id,cases.samples.sample_type")
    if not hits:
        logger.warning("No files found for %s %s", project_id, data_type)
        return None, []

    # downloading a project takes minutes, so the raw matrix is cached on disk
    folder = CACHE_DIR / f"{project_id}_{data_type.replace(' ', '-')}"
    x_file, y_file = folder / "X_raw.parquet", folder / "y_raw.parquet"
    if x_file.exists() and y_file.exists():
        logger.info("found cache for %s, skipping download", project_id)
        # wide matrices need larger thrift limits
        X = pd.read_parquet(x_file, thrift_string_size_limit=2**31 - 1,
                            thrift_container_size_limit=2**31 - 1)
        y = pd.read_parquet(y_file).squeeze("columns")
    else:
        logger.info("Downloading TCGA %s %s...", project_id, data_type)
        X, y = download_matrix(hits, data_type)
        if X is None:
            logger.warning("Failed to build matrix for %s %s", project_id, data_type)
            return None, []
        folder.mkdir(parents=True, exist_ok=True)
        X.to_parquet(x_file)
        y.to_frame("target").to_parquet(y_file)
        logger.info("saved raw data to %s", folder)

    # attach clinical columns, matched through the case each file came from
    clinical = pd.DataFrame.from_dict(fetch_clinical(project_id), orient="index")
    if not clinical.empty:
        case_of = {h["file_id"]: (h.get("cases") or [{}])[0].get("submitter_id", "") for h in hits}
        X[clinical.columns] = clinical.reindex([case_of.get(f, "") for f in X.index]).to_numpy()

    # A4 skipped: TCGA is exempt from the size filter
    result, data_results = hard_rules.run_hard_rules(X, y, candidate, skip=("A4",))
    if result is None:
        return None, data_results
    _, task_type = result

    logger.info("TCGA %s %s: loaded %d samples x %d features, task=%s",
                project_id, data_type, X.shape[0], X.shape[1], task_type)

    return Dataset(
        id=candidate.id,
        source="tcga",
        name=candidate.name,
        X=X,
        y=y,
        task_type=task_type,
        metadata={**candidate.metadata, "licence": "public", "url": candidate.url},
        domain="biomedical"), data_results
