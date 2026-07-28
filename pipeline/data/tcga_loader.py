# TCGA data loader - queries GDC API for gene expression, miRNA, and CNV data
# downloads raw matrices, attaches clinical metadata, and runs hard rules validation

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import GDC_BASE_URL, REQUEST_DELAY
from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules
from pipeline import stats

logger = logging.getLogger(__name__)

MIN_CASES = 500

ACCEPTED_WORKFLOWS = {
    "Gene Expression Quantification": [
        "STAR - Counts", "HTSeq - Counts", "HTSeq - FPKM", "HTSeq - FPKM-UQ",
    ],
    "Copy Number Segment": ["DNAcopy"],
    "miRNA Expression Quantification": ["BCGSC miRNA Profiling"],
}

CLASSIFICATION_SAMPLE_TYPES = {
    "Primary Tumor",
    "Solid Tissue Normal",
    "Blood Derived Normal",
    "Metastatic",
    "Recurrent Tumor",
    "Primary Blood Derived Cancer - Peripheral Blood",
}

SKIP_LINES_PREFIX = {
    "Gene Expression Quantification": ("N_", "__"),
    "miRNA Expression Quantification": ("miRNA_ID",),
    "Copy Number Segment": ("GDC_Aliquot",),
}

p_cache = {}
CACHE_DIR = Path(os.environ.get("PIPELINE_DATA", "data")) / "cache" / "tcga"

CLINICAL_COLS = ["clinical_vital_status", "clinical_gender", "clinical_age", "clinical_stage"]


def gdc_get(endpoint, params):
    url = f"{GDC_BASE_URL}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def gdc_post(endpoint, payload):
    url = f"{GDC_BASE_URL}/{endpoint}"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_projects():
    # get all TCGA projects and their case counts
    filters = {"op": "in", "content": {"field": "program.name", "value": ["TCGA"]}}
    data = gdc_get("projects", {
        "filters": json.dumps(filters),
        "fields": "project_id,name,disease_type,primary_site,summary.case_count",
        "size": 100,
        "format": "json",
    })
    projects = {}
    for hit in data["data"]["hits"]:
        projects[hit["project_id"]] = hit
    return projects


def fetch_file_groups(project_id):
    # find all unique (data_type, workflow, access) groups for one project
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                {"op": "in", "content": {"field": "data_type", "value": list(ACCEPTED_WORKFLOWS.keys())}},
            ],
        },
        "fields": "data_type,analysis.workflow_type,access,cases.samples.sample_type",
        "size": 5000,
        "format": "json",
    }
    data = gdc_post("files", payload)

    groups = {}
    for hit in data["data"]["hits"]:
        dt = hit.get("data_type", "")
        analysis = hit.get("analysis") or {}
        wft = analysis.get("workflow_type", "")
        acc = hit.get("access", "")
        key = (dt, wft, acc)

        if key not in groups:
            groups[key] = {
                "data_type": dt,
                "workflow_type": wft,
                "access": acc,
                "n_files": 0,
                "sample_types": set(),
            }

        groups[key]["n_files"] += 1

        for case in hit.get("cases", []):
            for sample in case.get("samples", []):
                st = sample.get("sample_type")
                if st:
                    groups[key]["sample_types"].add(st)

    return list(groups.values())


def get_one_file_id(project_id, data_type, workflow_type):
    # return one open-access file_id for feature counting
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                {"op": "=", "content": {"field": "data_type", "value": data_type}},
                {"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow_type}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
            ],
        },
        "fields": "file_id",
        "size": 1,
        "format": "json",
    }
    try:
        data = gdc_post("files", payload)
        hits = data["data"]["hits"]
        if hits:
            return hits[0]["file_id"]
        return None
    except Exception:
        return None


def count_features(file_id, data_type):
    # download one file and count feature rows
    skip_pfx = SKIP_LINES_PREFIX.get(data_type, ())
    try:
        resp = requests.get(f"{GDC_BASE_URL}/data/{file_id}", stream=True, timeout=60)
        resp.raise_for_status()
        count = 0

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")

            is_header = False
            for p in skip_pfx:
                if line.startswith(p):
                    is_header = True
                    break
            if is_header:
                continue

            count += 1

        if count > 0:
            return count
        return None
    except Exception as e:
        logger.warning("Feature count failed for %s: %s", file_id, e)
        return None


def infer_task(sample_types):
    # guess task type from sample types
    cls_types = sample_types & CLASSIFICATION_SAMPLE_TYPES

    if len(cls_types) >= 2:
        return "classification"
    if len(sample_types) >= 2:
        return "classification"
    if len(sample_types) == 1:
        return "classification"
    return "unknown"


def check_group(project_id, cancer_type, n_cases, primary_site, grp):
    # validate group and return CandidateInfo if accepted
    dt = grp["data_type"]
    wft = grp["workflow_type"]
    acc = grp["access"]
    sample_types = grp["sample_types"]

    cand_id = f"{project_id}_{dt.replace(' ', '-')}"
    cand_name = f"{cancer_type} - {dt}"

    # A5: open access only (source pre-filter, not tracked)
    if acc != "open":
        return None

    # A1: only accepted workflows (source pre-filter, not tracked)
    accepted = ACCEPTED_WORKFLOWS.get(dt, [])
    if accepted and wft not in accepted:
        return None

    task_type = infer_task(sample_types)

    # get feature count (cached by workflow)
    if wft in p_cache:
        p_count, file_id = p_cache[wft]
    else:
        file_id = get_one_file_id(project_id, dt, wft)
        if file_id is None:
            return None
        p_count = count_features(file_id, dt)
        if p_count is None:
            return None
        p_cache[wft] = (p_count, file_id)

    results = hard_rules.run_metadata_checks(
        n_samples=n_cases,
        n_features=p_count,
        task_type=task_type,
        licence="public",
        source="tcga",
        name=f"{project_id} {cancer_type}",
        skip=("A4",),
    )

    if not hard_rules.all_passed(results):
        stats.record(cand_id, "tcga", cand_name, results, "biomedical")
        return None

    return CandidateInfo(
        id=cand_id,
        source="tcga",
        name=cand_name,
        n_samples=n_cases,
        n_features=p_count,
        task_type=task_type,
        licence="public",
        url=f"https://portal.gdc.cancer.gov/projects/{project_id}",
        metadata={
            "project_id": project_id,
            "data_type": dt,
            "workflow_type": wft,
            "primary_site": primary_site,
            "sample_types": sorted(sample_types),
            "sample_file_id": file_id,
        },
        domain="biomedical",
    )


def list_candidates(max_candidates=100):
    # scrape GDC metadata to find eligible TCGA (project, data_type) pairs
    logger.info("Fetching TCGA projects from GDC...")
    projects = fetch_projects()
    logger.info("Found %d TCGA projects", len(projects))

    candidates = []

    for project_id, proj in projects.items():
        if len(candidates) >= max_candidates:
            break

        summary = proj.get("summary") or {}
        n_cases = int(summary.get("case_count", 0))
        cancer_type = proj.get("name", "")
        primary_site = ", ".join(proj.get("primary_site", []))

        # skip small projects
        if n_cases < MIN_CASES:
            logger.debug("%s: N=%d < %d, skipping", project_id, n_cases, MIN_CASES)
            continue

        groups = fetch_file_groups(project_id)
        time.sleep(REQUEST_DELAY)

        for grp in groups:
            if len(candidates) >= max_candidates:
                break

            candidate = check_group(project_id, cancer_type, n_cases, primary_site, grp)
            if candidate:
                candidates.append(candidate)
                logger.info(
                    "%s %s: N=%d P=%d task=%s",
                    project_id, grp["data_type"], n_cases, candidate.n_features, candidate.task_type,
                )

    logger.info("TCGA: %d candidates after metadata filters", len(candidates))
    return candidates


def fetch(candidate):
    # download feature matrix + clinical target, run data-level hard rules
    project_id = candidate.metadata["project_id"]
    data_type = candidate.metadata["data_type"]
    workflow_type = candidate.metadata["workflow_type"]

    # cache raw matrices (downloading is slow)
    folder = CACHE_DIR / f"{project_id}_{data_type.replace(' ', '-')}"
    x_file = folder / "X_raw.parquet"
    y_file = folder / "y_raw.parquet"

    if x_file.exists() and y_file.exists():
        logger.info("found cache for %s, skipping download", project_id)
        # wide methylation matrices need larger thrift limits
        X = pd.read_parquet(
            x_file,
            thrift_string_size_limit=2**31 - 1,
            thrift_container_size_limit=2**31 - 1,
        )
        y = pd.read_parquet(y_file).squeeze()
        file_ids = get_file_ids(project_id, data_type, workflow_type)
    else:
        logger.info("Downloading TCGA %s %s...", project_id, data_type)

        file_ids = get_file_ids(project_id, data_type, workflow_type)
        if not file_ids:
            logger.warning("No files found for %s %s", project_id, data_type)
            return None, []

        # build the feature matrix + get sample_type for each file
        result = build_feature_matrix(file_ids, data_type)
        X = result[0]
        sample_types = result[1]

        if X is None or X.empty:
            logger.warning("Failed to build matrix for %s %s", project_id, data_type)
            return None, []

        # use sample_type from expression file as label
        y = pd.Series(sample_types, name="target")

        # only keep rows that have both expression data and a label
        shared = X.index.intersection(y.index)
        logger.info("%s: %d samples with expression data and labels", project_id, len(shared))
        X = X.loc[shared]
        y = y.loc[shared]

        # drop samples without a known sample type
        has_label = y != ""
        X = X.loc[has_label]
        y = y.loc[has_label]

        # save so next run doesn't have to download again
        folder.mkdir(parents=True, exist_ok=True)
        X.to_parquet(x_file)
        y.to_frame("target").to_parquet(y_file)
        logger.info("saved raw data to %s", folder)

    # attach clinical (age, gender, vital_status, stage) per case
    case_id_map = {rec["file_id"]: rec["case_id"] for rec in file_ids}
    clinical = fetch_clinical(project_id)
    for col in CLINICAL_COLS:
        X[col] = [clinical.get(case_id_map.get(fid, ""), {}).get(col, None) for fid in X.index]

    # expensive data-level hard rules (A4 skipped: TCGA exempt from size filter)
    result, data_results = hard_rules.run_hard_rules(X, y, candidate, skip=("A4",))
    if result is None:
        return None, data_results

    _, resolved_task = result

    logger.info(
        "TCGA %s %s: loaded %d samples x %d features, task=%s",
        project_id, data_type, X.shape[0], X.shape[1], resolved_task,
    )

    return Dataset(
        id=candidate.id,
        source="tcga",
        name=candidate.name,
        X=X,
        y=y,
        task_type=resolved_task,
        metadata={
            **candidate.metadata,
            "licence": "public",
            "url": candidate.url,
        },
        domain="biomedical",
    ), data_results


def get_file_ids(project_id, data_type, workflow_type, max_files=3000):
    # get file_id + case_id + sample_type for every open-access file
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                {"op": "=", "content": {"field": "data_type", "value": data_type}},
                {"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow_type}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
            ],
        },
        "fields": "file_id,cases.case_id,cases.submitter_id,cases.samples.sample_type",
        "size": max_files,
        "format": "json",
    }
    try:
        data = gdc_post("files", payload)
        results = []

        for hit in data["data"]["hits"]:
            file_id = hit["file_id"]
            case_id = ""
            sample_type = ""

            # each file belongs to a case which has samples
            for case in hit.get("cases", []):
                case_id = case.get("submitter_id", case.get("case_id", ""))
                # get the sample type from the first sample
                for sample in case.get("samples", []):
                    sample_type = sample.get("sample_type", "")
                    break
                break

            results.append({
                "file_id": file_id,
                "case_id": case_id,
                "sample_type": sample_type,
            })

        return results
    except Exception as e:
        logger.warning("Failed to get file IDs for %s: %s", project_id, e)
        return []


def parse_gene_expression(lines):
    # STAR/HTSeq counts file: Cols are gene_id, gene_name, gene_type, unstranded
    data = {}
    for line in lines:
        if line.startswith("#") or line.startswith("gene_id") or line.startswith("N_"):
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            try:
                data[parts[1]] = float(parts[3])  # gene_name -> unstranded counts
            except ValueError:
                continue
    return data


def parse_mirna(lines):
    # BCGSC miRNA file: Cols are miRNA_ID, read_count, reads_per_million_miRNA_mapped
    data = {}
    for line in lines:
        if line.startswith("miRNA_ID") or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                data[parts[0]] = float(parts[1])  # read_count
            except ValueError:
                continue
    return data


PARSERS = {
    "Gene Expression Quantification": parse_gene_expression,
    "miRNA Expression Quantification": parse_mirna,
}


def download_single_file(file_id, data_type):
    # download one GDC file and parse it according to data_type
    parser = PARSERS.get(data_type)
    if parser is None:
        logger.debug("No parser for data_type %s", data_type)
        return None
    try:
        resp = requests.get(f"{GDC_BASE_URL}/data/{file_id}", timeout=60)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        data = parser(lines)
        if data:
            return pd.Series(data)
        return None
    except Exception as e:
        logger.debug("Failed to download file %s: %s", file_id, e)
        return None


def build_feature_matrix(file_records, data_type, max_files=3000):
    # download per-sample files and stack them into a samples x features matrix
    records = file_records[:max_files]

    def download_one(rec):
        series = download_single_file(rec["file_id"], data_type)
        return rec, series

    # download 10 files at the same time
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(download_one, records))

    logger.info("  downloaded %d files", len(results))

    rows = {}
    sample_types = {}
    for rec, series in results:
        if series is not None:
            rows[rec["file_id"]] = series
            sample_types[rec["file_id"]] = rec.get("sample_type", "")

    if not rows:
        return None, {}

    X = pd.DataFrame(rows).T
    return X, sample_types


def fetch_clinical(project_id):
    # fetch clinical data for all cases in the project, keyed by case_id
    payload = {
        "filters": {"op": "=", "content": {"field": "project.project_id", "value": project_id}},
        "fields": "submitter_id,demographic.vital_status,demographic.gender,demographic.age_at_index,diagnoses.ajcc_pathologic_stage",
        "size": 5000,
        "format": "json",
    }
    try:
        data = gdc_post("cases", payload)
    except Exception as e:
        logger.warning("Failed to fetch clinical for %s: %s", project_id, e)
        return {}

    out = {}
    for case in data["data"]["hits"]:
        case_id = case.get("submitter_id")
        if not case_id:
            continue
        demo = case.get("demographic") or {}
        diag = (case.get("diagnoses") or [{}])[0] or {}
        out[case_id] = {
            "clinical_vital_status": demo.get("vital_status") or None,
            "clinical_gender": demo.get("gender") or None,
            "clinical_age": demo.get("age_at_index"),
            "clinical_stage": diag.get("ajcc_pathologic_stage") or None,
        }
    return out
