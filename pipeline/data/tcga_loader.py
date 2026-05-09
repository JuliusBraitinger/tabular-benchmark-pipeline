"""TCGA data loader - scrapes GDC metadata and downloads gene expression matrices."""
import json
import logging
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

# only keep projects with at least this many cases
MIN_CASES = 500

# for each data type, the list of workflow names we accept
ACCEPTED_WORKFLOWS = {
    "Gene Expression Quantification": [
        "STAR - Counts", "HTSeq - Counts", "HTSeq - FPKM", "HTSeq - FPKM-UQ",
    ],
    "Methylation Beta Value": ["SeSAMe Methylation Beta Estimation"],
    "Copy Number Segment": ["DNAcopy"],
    "miRNA Expression Quantification": ["BCGSC miRNA Profiling"],
}

# sample types that make sense as a classification target
CLASSIFICATION_SAMPLE_TYPES = {
    "Primary Tumor",
    "Solid Tissue Normal",
    "Blood Derived Normal",
    "Metastatic",
    "Recurrent Tumor",
    "Primary Blood Derived Cancer - Peripheral Blood",
}

# header rows that should NOT be counted when we count feature rows in a file
SKIP_LINES_PREFIX = {
    "Gene Expression Quantification": ("N_", "__"),
    "Methylation Beta Value": ("Composite",),
    "miRNA Expression Quantification": ("miRNA_ID",),
    "Copy Number Segment": ("GDC_Aliquot",),
}

# cache the feature count for each workflow so we don't count the same thing twice
_p_cache = {}

# where to save raw X/y so we can skip the download next time
CACHE_DIR = Path("data/cache/tcga")


# --- GDC API wrappers ---


def _gdc_get(endpoint, params):
    # simple GET request to the GDC API, returns parsed JSON
    url = f"{GDC_BASE_URL}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _gdc_post(endpoint, payload):
    # simple POST request when we need to send a JSON filter body
    url = f"{GDC_BASE_URL}/{endpoint}"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()



def _fetch_projects():
    """Get all TCGA projects and their case counts."""
    # filter: only programs named "TCGA"
    filters = {"op": "in", "content": {"field": "program.name", "value": ["TCGA"]}}
    data = _gdc_get("projects", {
        "filters": json.dumps(filters),
        "fields": "project_id,name,disease_type,primary_site,summary.case_count",
        "size": 100,
        "format": "json",
    })
    # turn the list of hits into a dict keyed by project_id
    projects = {}
    for hit in data["data"]["hits"]:
        projects[hit["project_id"]] = hit
    return projects


def _fetch_file_groups(project_id):
    """For one project, find all unique (data_type, workflow, access) groups.

    Grouping lets us count features once per group instead of once per file.
    """
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
    data = _gdc_post("files", payload)

    # go through every file and collect groups keyed by (data_type, workflow, access)
    groups = {}
    for hit in data["data"]["hits"]:
        dt = hit.get("data_type", "")
        analysis = hit.get("analysis") or {}
        wft = analysis.get("workflow_type", "")
        acc = hit.get("access", "")
        key = (dt, wft, acc)

        # first time we see this group -> initialize it
        if key not in groups:
            groups[key] = {
                "data_type": dt,
                "workflow_type": wft,
                "access": acc,
                "n_files": 0,
                "sample_types": set(),
            }

        groups[key]["n_files"] += 1

        # collect every sample type this group has seen
        for case in hit.get("cases", []):
            for sample in case.get("samples", []):
                st = sample.get("sample_type")
                if st:
                    groups[key]["sample_types"].add(st)

    return list(groups.values())


def _get_one_file_id(project_id, data_type, workflow_type):
    """Return one open-access file_id. Used to download a sample file for feature counting."""
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
        data = _gdc_post("files", payload)
        hits = data["data"]["hits"]
        if hits:
            return hits[0]["file_id"]
        return None
    except Exception:
        return None


def _count_features(file_id, data_type):
    """Download one file and count how many feature rows it has."""
    skip_pfx = SKIP_LINES_PREFIX.get(data_type, ())
    try:
        resp = requests.get(f"{GDC_BASE_URL}/data/{file_id}", stream=True, timeout=60)
        resp.raise_for_status()
        count = 0

        # read the file line by line, skip header rows, count everything else
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")

            # check if this line looks like a header we should skip
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


def _infer_task(sample_types):
    """Guess task type from the sample types we have.

    There is only classification here because TCGA targets we care about
    (tumor vs normal, alive vs dead, etc.) are categorical - FOR NOW!
    TODO: handle regression targets too (e.g. days_to_death, age_at_diagnosis)
    """
    # how many of the sample types are ones we know make sense for classification?
    cls_types = sample_types & CLASSIFICATION_SAMPLE_TYPES

    if len(cls_types) >= 2:
        return "classification"
    if len(sample_types) >= 2:
        return "classification"
    if len(sample_types) == 1:
        return "classification"  # pan-cancer possible
    return "unknown"


# =========================
# main entry points
# =========================


def list_candidates(max_candidates=100):
    """Scrape GDC metadata to find eligible TCGA (project, data_type) pairs."""
    logger.info("Fetching TCGA projects from GDC...")
    projects = _fetch_projects()
    logger.info("Found %d TCGA projects", len(projects))

    candidates = []

    # loop over every project
    for project_id, proj in projects.items():
        # stop as soon as we have enough candidates
        if len(candidates) >= max_candidates:
            break

        # basic project info
        summary = proj.get("summary") or {}
        n_cases = int(summary.get("case_count", 0))
        cancer_type = proj.get("name", "")
        primary_site = ", ".join(proj.get("primary_site", []))

        # skip projects that are too small
        if n_cases < MIN_CASES:
            logger.debug("%s: N=%d < %d, skipping", project_id, n_cases, MIN_CASES)
            continue

        # get all (data_type, workflow) groups in this project
        groups = _fetch_file_groups(project_id)
        time.sleep(REQUEST_DELAY)

        # check each group against the rules
        for grp in groups:
            if len(candidates) >= max_candidates:
                break

            dt = grp["data_type"]
            wft = grp["workflow_type"]
            acc = grp["access"]
            sample_types = grp["sample_types"]

            # A6: only open access allowed
            if acc != "open":
                continue

            # A1: only workflow types we explicitly accept
            accepted = ACCEPTED_WORKFLOWS.get(dt, [])
            if accepted and wft not in accepted:
                continue

            # guess the task type from the sample composition
            task_type = _infer_task(sample_types)

            # count features, reusing the cache so we don't download the same file twice
            if wft in _p_cache:
                p_count, file_id = _p_cache[wft]
            else:
                file_id = _get_one_file_id(project_id, dt, wft)
                if file_id is None:
                    continue
                p_count = _count_features(file_id, dt)
                if p_count is None:
                    continue
                _p_cache[wft] = (p_count, file_id)

            # run the cheap metadata hard rules
            results = hard_rules.run_metadata_checks(
                n_samples=n_cases,
                n_features=p_count,
                task_type=task_type,
                licence="public",
                source="tcga",
                name=f"{project_id} {cancer_type}",
            )
            stats.record(project_id, "tcga", cancer_type + " - " + dt, results)

            if not hard_rules.all_passed(results):
                continue

            # passed all metadata checks -> build a CandidateInfo and add it
            candidates.append(CandidateInfo(
                id=f"{project_id}_{dt.replace(' ', '-')}",
                source="tcga",
                name=f"{cancer_type} - {dt}",
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
            ))
            logger.info(
                "%s %s: N=%d P=%d task=%s",
                project_id, dt, n_cases, p_count, task_type,
            )

    logger.info("TCGA: %d candidates after metadata filters", len(candidates))
    return candidates


def fetch(candidate):
    """Download the feature matrix + clinical target, then run data-level hard rules.

    Steps:
      1. Get all open file IDs for this (project, data_type) pair
      2. Download each file and pivot into a samples x features matrix
      3. Download clinical data and pick a target variable y
      4. Run data-level hard rules; if they pass, return a Dataset
    """
    project_id = candidate.metadata["project_id"]
    data_type = candidate.metadata["data_type"]
    workflow_type = candidate.metadata["workflow_type"]

    # downloading takes forever -> save the raw data for now 
    # after the first download. next time we just load from disk for testint the rules 
    folder = CACHE_DIR / f"{project_id}_{data_type.replace(' ', '-')}"
    x_file = folder / "X_raw.parquet"
    y_file = folder / "y_raw.parquet"

    if x_file.exists() and y_file.exists():
        # already downloaded before, just load it
        logger.info("found cache for %s, skipping download", project_id)
        # raise pyarrow's thrift schema-size limit so wide methylation matrices
        # (LUAD has ~915k columns) don't trip the default ~100MB cap
        X = pd.read_parquet(
            x_file,
            thrift_string_size_limit=2**31 - 1,
            thrift_container_size_limit=2**31 - 1,
        )
        y = pd.read_parquet(y_file).squeeze()
        file_ids = _get_file_ids(project_id, data_type, workflow_type)
    else:
        # no cache yet, download everything
        logger.info("Downloading TCGA %s %s...", project_id, data_type)

        file_ids = _get_file_ids(project_id, data_type, workflow_type)
        if not file_ids:
            logger.warning("No files found for %s %s", project_id, data_type)
            return None

        # build the feature matrix + get sample_type for each file
        result = _build_feature_matrix(file_ids, data_type)
        X = result[0]
        sample_types = result[1]

        if X is None or X.empty:
            logger.warning("Failed to build matrix for %s %s", project_id, data_type)
            return None

        # use the sample_type from the actual expression file as the label
        # before this was using clinical data which didnt match the files
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

        # save so next run doesnt have to download again
        folder.mkdir(parents=True, exist_ok=True)
        X.to_parquet(x_file)
        y.to_frame("target").to_parquet(y_file)
        logger.info("saved raw data to %s", folder)

    # attach clinical (age, gender, vital_status, stage) per case
    case_id_map = {rec["file_id"]: rec["case_id"] for rec in file_ids}
    clinical = _fetch_clinical(project_id)
    for col in CLINICAL_COLS:
        X[col] = [clinical.get(case_id_map.get(fid, ""), {}).get(col, None) for fid in X.index]

    # Step 4: expensive data-level hard rules
    data_results = hard_rules.run_data_checks(X, y, task_type=candidate.task_type)
    stats.record(project_id, "tcga", candidate.name, data_results)
    failed = hard_rules.failed_rules(data_results)
    if failed:
        reasons = ", ".join(f"{r.rule}: {r.reason}" for r in failed)
        logger.info("TCGA %s discarded (data check): %s", project_id, reasons)
        return None

    # if A1 inferred a task from the target, use it instead of 'unknown'
    resolved_task = hard_rules.inferred_task_type(data_results) or candidate.task_type

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
    )


# -- download + parsing stuff used inside fetch() --


def _get_file_ids(project_id, data_type, workflow_type, max_files=3000):
    """Get file_id + case_id + sample_type for every open-access file."""
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
        # also grab sample_type so we know what tissue this file is from
        "fields": "file_id,cases.case_id,cases.submitter_id,cases.samples.sample_type",
        "size": max_files,
        "format": "json",
    }
    try:
        data = _gdc_post("files", payload)
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


def _parse_gene_expression(lines):
    """STAR/HTSeq counts file. Cols: gene_id, gene_name, gene_type, unstranded, ..."""
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


def _parse_methylation(lines):
    """SeSAMe Methylation Beta file. Cols: Composite Element REF, Beta_value."""
    data = {}
    for line in lines:
        if line.startswith("Composite") or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                value = float(parts[1])  # NaN beta values appear as "NA" -> ValueError, skip
                data[parts[0]] = value
            except ValueError:
                continue
    return data


def _parse_mirna(lines):
    """BCGSC miRNA file. Cols: miRNA_ID, read_count, reads_per_million_miRNA_mapped, cross-mapped."""
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


_PARSERS = {
    "Gene Expression Quantification": _parse_gene_expression,
    "Methylation Beta Value": _parse_methylation,
    "miRNA Expression Quantification": _parse_mirna,
}


def _download_single_file(file_id, data_type):
    """Download one GDC file and parse it according to data_type.

    Returns a pandas Series: feature_name -> value.
    """
    parser = _PARSERS.get(data_type)
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


def _build_feature_matrix(file_records, data_type, max_files=3000):
    """Download per-sample files and stack them into a samples x features matrix.

    Each file becomes its own row, indexed by file_id, so patients with both
    tumor and matched-normal samples contribute multiple rows. Returns the
    matrix and a dict of file_id -> sample_type for the target labels.
    """
    records = file_records[:max_files]

    # download one file and return the record + result
    def download_one(rec):
        series = _download_single_file(rec["file_id"], data_type)
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


CLINICAL_COLS = ["clinical_vital_status", "clinical_gender", "clinical_age", "clinical_stage"]


def _fetch_clinical(project_id): #fetch clinical data for all cases in the project, keyed by case_id
    payload = { # from api values we want in clinical data
        "filters": {"op": "=", "content": {"field": "project.project_id", "value": project_id}},
        "fields": "submitter_id,demographic.vital_status,demographic.gender,demographic.age_at_index,diagnoses.ajcc_pathologic_stage",
        "size": 5000,
        "format": "json",
    }
    try:
        data = _gdc_post("cases", payload)
    except Exception as e:
        logger.warning("Failed to fetch clinical for %s: %s", project_id, e)
        return { }

    out = {}
    for case in data[data][hits]:
        case_id = case.get("submitter_id")
        if not case_id:
            continue
        demo = case.get("demographic") or {}
        diag = (case.get("diagnoses") or [{}])[0] or {}
        out[case_id] = {
            "linical_vital_status": demo.get("vital_status") ,
            "clinical_gender": demo.get("gender") ,
            "clinical_age": demo.get("age_at_index"),
            "clinical_stage": diag.get("ajcc_pathologic_stage") ,
        }
    return out
