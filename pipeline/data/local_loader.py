# Local loader: load datasets that are stored in data/datasets/{id}/
#takes csv and parquet files for X and y, and optional meta.pkl for metadata
import logging
import pandas as pd
from pathlib import Path
import pickle

from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules
from pipeline.hard_rules.base import RuleResult
from pipeline import stats

logger = logging.getLogger(__name__)


DATASETS_DIR = Path("data/datasets")


def save_target(dataset_dir, y):
    # save target variable as y.csv
    y_file = dataset_dir / "y.csv"
    y.to_csv(y_file, index=False)
    logger.info("Saved target to %s", y_file)


def find_or_build_target(dataset_dir):
    # search for target variable in CSV files
    # currently only for testing with Mimmic datasets 
    targetNames = ["target", "hospital_expire_flag", "mortality", "outcome", "y","label"]

    # search CSV files for target columns
    for csv_file in dataset_dir.glob("*.csv"):
        if csv_file.stem == "X":
            continue
        try:
            df = pd.read_csv(csv_file)
            for col in targetNames:
                if col in df.columns:
                    logger.info("Found target '%s' in %s", col, csv_file.name)
                    return df[col].squeeze()
        except Exception as e:
            logger.debug("Error reading %s: %s", csv_file, e)

    logger.warning("No target variable found in %s", dataset_dir)

    for parquet_file in dataset_dir.glob("*.parquet"):
        if parquet_file.stem == "X":
            continue
        try:
            df = pd.read_parquet(parquet_file)
            for col in targetNames:
                if col in df.columns:
                    logger.info("Found target '%s' in %s", col, parquet_file.name)
                    return df[col].squeeze()
        except Exception as e:
            logger.debug("Error reading %s: %s", parquet_file, e)

    # detect possible targets if not found
    candidates = []
    for file in dataset_dir.glob("*.csv"):
        if file.stem == "X":
            continue
        try:
            df = pd.read_csv(file)
            for col in df.columns:
                unique = df[col].nunique()
                if unique > 10:
                    continue
                if df[col].isna().sum() / len(df) < 0.5:
                    candidates.append({"file": file.name, "column": col, "cardinality": unique})
        except:
            pass

    if candidates:
        best = min(candidates, key=lambda c: c["cardinality"])
        file = dataset_dir / best["file"]
        df = pd.read_csv(file)
        logger.info("Auto-selected target: '%s' from %s", best["column"], best["file"])
        return df[best["column"]].squeeze()

    return None


def datasets(dataset_dir):
    # load X and y from CSV or parquet
    X = None
    y = None

    X_csv = dataset_dir / "X.csv"
    X_parquet = dataset_dir / "X.parquet"
    y_csv = dataset_dir / "y.csv"
    y_parquet = dataset_dir / "y.parquet"

    if X_csv.exists():
        X = pd.read_csv(X_csv)
    elif X_parquet.exists():
        X = pd.read_parquet(X_parquet)

    if y_csv.exists():
        y = pd.read_csv(y_csv).squeeze()
    elif y_parquet.exists():
        y = pd.read_parquet(y_parquet)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
    else:
        # search for target in other CSV files
        y = find_or_build_target(dataset_dir)
        if y is not None:
            # save the found target to y.csv
            save_target(dataset_dir, y)

    return X, y


def list_candidates(max_candidates=50):
    candidates = []

    for dataset_dir in sorted(DATASETS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_id = dataset_dir.name

        X, y = datasets(dataset_dir)
        if X is None or y is None:
            stats.record(dataset_id, "local", dataset_id, [
                RuleResult(rule="pre-filter", passed=False, reason="missing X/y files")
            ])
            continue

        n_samples = len(X)
        n_features = len(X.columns)

        if y.dtype in ['float64', 'float32']:
            task_type = "regression"
        else:
            task_type = "classification"

        meta_path = dataset_dir / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
        else:
            metadata = {}

        name = metadata.get("name", dataset_id)
        domain = metadata.get("domain", "general")

        results = hard_rules.run_metadata_checks(
            n_samples=n_samples,
            n_features=n_features,
            task_type=task_type,
            licence="public-domain",
            source="local",
            name=name,
        )

        if not hard_rules.all_passed(results):
            stats.record(dataset_id, "local", name, results)
            continue

        candidates.append(CandidateInfo(
            id=dataset_id,
            source="local",
            name=name,
            n_samples=n_samples,
            n_features=n_features,
            task_type=task_type,
            licence="public-domain",
            url="",
            metadata=metadata,
            domain=domain,
        ))

        if len(candidates) >= max_candidates:
            break

    return candidates


def fetch(candidate):
    dataset_dir = DATASETS_DIR / candidate.id

    X, y = datasets(dataset_dir)

    result, data_results = hard_rules.run_hard_rules(X, y, candidate)
    if result is None:
        return None, []

    _, task_type = result

    return Dataset(
        X=X,
        y=y,
        task_type=task_type,
        id=candidate.id,
        source="local",
        name=candidate.name,
        metadata=candidate.metadata,
    ), data_results

