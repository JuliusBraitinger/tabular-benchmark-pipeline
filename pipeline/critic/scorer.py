"""Scorer — runs all soft rules on all datasets, produces score matrices."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from pipeline.data.base import Dataset

logger = logging.getLogger(__name__)

# Soft rule modules will be imported here once implemented
# from pipeline.soft_rules import s1_uniqueness, s2_iid, s3_data_quality, s4_leakage, s6_class_balance


def score_all(
    datasets: list[Dataset],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all soft rules on all datasets.

    Outputs:
        score_matrix_continuous.csv — raw scores [0.0, 1.0] for CRITIC
        score_matrix_tiers.csv     — discrete tiers (0-100) for display
        rule_details.json          — per-dataset diagnostics

    Returns:
        (continuous_matrix, tier_matrix)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: implement once soft rule modules are built
    # For each dataset, run S1-S4, S6 and collect scores + details

    logger.info("Scoring %d datasets... (not yet implemented)", len(datasets))

    continuous_rows: list[dict] = []
    tier_rows: list[dict] = []
    details: dict[str, dict] = {}

    for ds in datasets:
        c_row = {"dataset_id": ds.id}
        t_row = {"dataset_id": ds.id}
        # result = s1_uniqueness.score(ds, pool=datasets)
        # c_row["S1"] = result.score    # continuous for CRITIC
        # t_row["S1"] = result.tier     # discrete for display
        # details[ds.id] = {"S1": result.details, ...}
        continuous_rows.append(c_row)
        tier_rows.append(t_row)

    continuous_matrix = pd.DataFrame(continuous_rows).set_index("dataset_id")
    tier_matrix = pd.DataFrame(tier_rows).set_index("dataset_id")

    continuous_matrix.to_csv(output_dir / "score_matrix_continuous.csv")
    tier_matrix.to_csv(output_dir / "score_matrix_tiers.csv")

    with open(output_dir / "rule_details.json", "w") as f:
        json.dump(details, f, indent=2)

    logger.info("Score matrices saved to %s", output_dir)
    return continuous_matrix, tier_matrix
