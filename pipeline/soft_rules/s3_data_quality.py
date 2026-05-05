# S3 Data Quality (15 points)
# missing fraction, constant features, outlier percentage
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

def non_constant(X):
    n_constant = (X.nunique() <= 1).sum()
    n_quasiconstant =  0 #column where (top-1 value frequency) > 0.95
    p = X.shape[1]
    c_const = 1 - (n_constant + n_quasiconstant) / p
    score = c_const
    return score, {
        "n_constant": int(n_constant),
        "n_quasiconstant": int(n_quasiconstant),
        "n_features": int(p)
    }
def outlier_percentage(X):
    #TODO implement
    return 1.0, {}

def consistency(X):
    #TODO implement
    return 1.0, {}

def score(dataset): # here other metrics will be added  s
    completeness_score, completeness_details = completeness(dataset.X)
    return SoftRuleResult(
        rule="S3",
        score=completeness_score,
        details=completeness_details
    )   
