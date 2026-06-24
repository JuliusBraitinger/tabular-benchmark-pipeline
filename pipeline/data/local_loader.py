# Local loader: load datasets that are stored in data/datasets/{id}/
# for various data sources. Each dataset is expected to have X.parquet, y.parquet, and optionally meta.pkl.
import logging
import pandas as pd
from pathlib import Path
import pickle

from pipeline.data.base import CandidateInfo, Dataset
from pipeline.hard_rules import runner as hard_rules
from pipeline import stats

logger = logging.getLogger(__name__)


DATASETS_DIR = Path("data/datasets")


def list_candidates(max_candidates=50):
    candidates = []

    for dataset_dir in sorted(DATASETS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_id = dataset_dir.name

        X_path = dataset_dir / "X.parquet"
        y_path = dataset_dir / "y.parquet"

        if not X_path.exists() or not y_path.exists():
            continue

        X = pd.read_parquet(X_path)
        y = pd.read_parquet(y_path)

        n_samples = len(X)
        n_features = len(X.columns)

        # Infer task type from y
        if y.dtype in ['float64', 'float32']:
            task_type = "regression"
        else:
            task_type = "classification"

        # Load metadata if exists
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

        stats.record(dataset_id, "local", name, results)

        if not hard_rules.all_passed(results):
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

    X = pd.read_parquet(dataset_dir / "X.parquet")
    y = pd.read_parquet(dataset_dir / "y.parquet")
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    result, data_results = hard_rules.run_hard_rules(X, y, candidate)
    stats.record(candidate.id, candidate.source, candidate.name, data_results)
    if result is None:
        return None

    _, task_type = result

    return Dataset(
        X=X,
        y=y,
        task_type=task_type,
        id=candidate.id,
        source="local",
        name=candidate.name,
        metadata=candidate.metadata,
    )

