# Final dataset-level scoring: combines per-rule scores from soft_stats.csv
# with per-rule weights from CRITIC into one composite score per dataset.
# Applies PASS_THRESHOLD to mark which datasets make the benchmark.

import pandas as pd
from pipeline.config import TOTAL_POINTS, PASS_THRESHOLD


def compute_critic_scores(score_matrix, critic_results): #combines score from each rule with critic weights 
    final_weights = {r.rule: r.critic_score for r in critic_results}

    composite = sum(
        score_matrix[rule] * weight
        for rule, weight in final_weights.items()
    )

    result = pd.DataFrame({
        "composite_score": composite,
        "composite_fraction": composite / TOTAL_POINTS,
    })
    result["passed"] = result["composite_fraction"] > PASS_THRESHOLD

    return result.sort_values("composite_score")