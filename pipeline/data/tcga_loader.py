"""TCGA data loader - scrapes GDC metadata and downloads gene expression matrices."""
# TODO:  save full clinical data not just target(vital_status, tumor_stage, age, etc.) 
import json
import logging
import time

import pandas as pd
import requests

from pipeline.config import GDC_BASE_URL, REQUEST_DELAY
from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules

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

    logger.info("Downloading TCGA %s %s...", project_id, data_type)

    # Step 1: list all file ids
    file_ids = _get_file_ids(project_id, data_type, workflow_type)
    if not file_ids:
        logger.warning("No files found for %s %s", project_id, data_type)
        return None

    # Step 2: build the feature matrix (samples x features)
    X = _build_feature_matrix(file_ids, data_type)
    if X is None or X.empty:
        logger.warning("Failed to build matrix for %s %s", project_id, data_type)
        return None

    # Step 3: download clinical data and pick a target
    y = _download_clinical_target(project_id, candidate.task_type)
    if y is None or y.empty:
        logger.warning("No clinical target for %s", project_id)
        return None

    # X and y must line up on the same sample IDs
    shared = X.index.intersection(y.index)
    if len(shared) < MIN_CASES // 2:
        logger.warning("%s: only %d shared samples", project_id, len(shared))
        return None

    X = X.loc[shared]
    y = y.loc[shared]

    # Step 4: expensive data-level hard rules
    data_results = hard_rules.run_data_checks(X, y, task_type=candidate.task_type)
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
        id=f"TCGA-{candidate.id}",
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


def _get_file_ids(project_id, data_type, workflow_type, max_files=600):
    """Get file_id + case_id for every open-access file of this type."""
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
        "fields": "file_id,cases.case_id,cases.submitter_id",
        "size": max_files,
        "format": "json",
    }
    try:
        data = _gdc_post("files", payload)
        results = []

        # each hit is one file, paired with a case id for later X/y alignment
        for hit in data["data"]["hits"]:
            file_id = hit["file_id"]
            case_id = ""
            # take the first case attached to this file
            for case in hit.get("cases", []):
                case_id = case.get("submitter_id", case.get("case_id", ""))
                break
            results.append({"file_id": file_id, "case_id": case_id})

        return results
    except Exception as e:
        logger.warning("Failed to get file IDs for %s: %s", project_id, e)
        return []


def _download_single_file(file_id, data_type):
    """Download one GDC file. TCGA stores one file per patient so we need many of these.

    Returns a pandas Series: gene_name -> value.

    """
    try:
        resp = requests.get(f"{GDC_BASE_URL}/data/{file_id}", timeout=60)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")

        data = {}
        for line in lines:
            # skip comment lines and header rows
            if line.startswith("#") or line.startswith("gene_id"):
                continue
            # skip the summary rows (N_unmapped, N_multimapping, etc.)
            if line.startswith("N_"):
                continue

            parts = line.split("\t")
            if len(parts) >= 4:
                gene_name = parts[1]  # gene_name column
                try:
                    value = float(parts[3])  # unstranded counts
                    data[gene_name] = value
                except ValueError:
                    continue

        if data:
            return pd.Series(data)
        return None
    except Exception as e:
        logger.debug("Failed to download file %s: %s", file_id, e)
        return None


def _build_feature_matrix(file_records, data_type, max_files=200):
    """Download per-sample files and stack them into a samples x features matrix."""
    # each case id -> its Series of features
    rows = {}
    for i, rec in enumerate(file_records[:max_files]):
        series = _download_single_file(rec["file_id"], data_type)
        if series is not None:
            rows[rec["case_id"]] = series

        # log progress every 50 files so we know it's alive
        if (i + 1) % 50 == 0:
            logger.info("  downloaded %d/%d files", i + 1, min(len(file_records), max_files))
        time.sleep(0.1)

    if not rows:
        return None

    # dict of Series -> DataFrame (features x samples), then transpose to (samples x features)
    return pd.DataFrame(rows).T


def _download_clinical_target(project_id, task_type):
    """Download clinical data and pick a target variable.

    First try sample_type (e.g. Primary Tumor vs Solid Tissue Normal).
    """
    payload = {
        "filters": {
            "op": "=",
            "content": {"field": "project.project_id", "value": project_id},
        },
        "fields": ",".join([
            "submitter_id",
            "demographic.vital_status",
            "diagnoses.tumor_stage",
            "samples.sample_type",
            "samples.submitter_id",
        ]),
        "size": 2000,
        "format": "json",
    }
    try:
        data = _gdc_post("cases", payload)
        hits = data["data"]["hits"]

        # for each case, pick a single target value
        targets = {}
        for case in hits:
            case_id = case.get("submitter_id", "")
            if not case_id:
                continue

            # keep ONE target per patient which is in this case tumor state

            # preferred: sample_type from the first sample
            samples = case.get("samples", [])
            if samples:
                sample_type = samples[0].get("sample_type", "")
                if sample_type:
                    targets[case_id] = sample_type
                    continue

            # fallback: vital_status from demographics
            demo = case.get("demographic") or {}
            vital = demo.get("vital_status", "")
            if vital:
                targets[case_id] = vital

        if not targets:
            return None
        return pd.Series(targets, name="target")

    except Exception as e:
        logger.warning("Failed to download clinical data for %s: %s", project_id, e)
        return None
