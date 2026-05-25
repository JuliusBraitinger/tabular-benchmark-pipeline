# Final dataset-level scoring: combines per-rule scores from soft_stats.csv
# with per-rule weights from CRITIC into one composite score per dataset.
# Applies PASS_THRESHOLD to mark which datasets make the benchmark.

import pandas as pd
from pipeline.config import TOTAL_POINTS, PASS_THRESHOLD


def compute_critic_scores(score_matrix, weights):
    #combine per rule score with critic weights to get composite score per dataset
    #score_matrix is a dataframe indexed by dataset_id, with one column per rule
    composite = sum(score_matrix[rule] * w for rule, w in weights.items())

    result = pd.DataFrame({
        "composite_score": composite,
        "composite_fraction": composite / TOTAL_POINTS,
    })
    result["passed"] = result["composite_fraction"] >= PASS_THRESHOLD

    return result.sort_values("composite_score", ascending=False)