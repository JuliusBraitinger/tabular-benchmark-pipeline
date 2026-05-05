# S3 Data Quality (15 points)
# missing fraction, constant features, outlier percentage

#TODO implement score()

from pipeline.soft_rules.base import SoftRuleResult


# S3 Data Quality (15 points)
#
# Composite score over  sub-metrics; each sub-metric is in [0, 1] (1 = clean).
# Budach et al. (2022), Eq. for Completeness:
#        c_miss = 1 - (1/p) * sum_j ( missing(c_j) / n )

from pipeline.soft_rules.base import SoftRuleResult


def completeness(X): #penalizes dataset with empty columns more than a flat total-cells ratio would
    n_rows, n_cols = X.shape
    if n_rows == 0 or n_cols == 0:
        return 1.0, {"n_rows": int(n_rows), "n_features": int(n_cols), "n_cols_with_missing": 0}

    #missing rate per column 
    per_col_missing_rate = X.isna().sum(axis=0) / n_rows
    score = 1.0 - per_col_missing_rate.mean()
    n_cols_with_missing = (per_col_missing_rate > 0).sum()

    return score, {
        "n_rows": n_rows,
        "n_features": n_cols,
        "n_cols_with_missing": n_cols_with_missing,
        "mean_per_col_missing_rate": per_col_missing_rate.mean(),
        "max_per_col_missing_rate": per_col_missing_rate.max(),
    }


def score(dataset, pool=None):
    return SoftRuleResult(rule="S3", score=1.0, details={})
