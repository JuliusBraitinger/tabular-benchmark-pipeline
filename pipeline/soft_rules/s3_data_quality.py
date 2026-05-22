# S3 Data Quality (15 points)
# missing fraction, constant features, outlier percentage
#
# Composite score over  sub-metrics; each sub-metric is in [0, 1] (1 = clean).
# Budach et al. (2022), Eq. for Completeness:
#        c_miss = 1 - (1/p) * sum_j ( missing(c_j) / n )

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from pipeline.soft_rules.base import SoftRuleResult


# When a dataset has more features than this, S3 subsamples to the
# top-k most variable columns before doing the heavy operations
# (PCA, hashing). Same trick A3 uses.
MAX_FEATURES_FOR_HEAVY_OPS = 1000


def top_variable_columns(X, k=MAX_FEATURES_FOR_HEAVY_OPS):
    # drop non-numeric columns (e.g. datetime strings like "2016-01-01 00:00:00")
    # so to_numpy(dtype=float) doesn't crash on them
    X = X.select_dtypes(include="number")
    if X.shape[1] <= k:
        return X
    # variance per column; all-NaN columns become -1 so they never get picked
    column_variances = np.nanvar(X.to_numpy(dtype=float, copy=False), axis=0)
    column_variances = np.nan_to_num(column_variances, nan=-1.0)

    # indices of the k columns with the highest variance
    top_k_indices = np.argpartition(-column_variances, k)[:k]

    return X.iloc[:, top_k_indices]

# Sub-metric weights for the final S3 score.
#  JUST A EDUCATED GUESS NEEDS IMPROVEMENT
# c_uniq (duplicate detection) moved to S2 — it's an IID signal, not a quality one.
# the freed 0.15 was redistributed across the four remaining sub-metrics.
SUB_WEIGHTS = {
    "c_miss":    0.35,  # completeness
    "c_consist": 0.25,  # consistent representation
    "c_out":     0.25,  # outliers
    "c_const":   0.15,  # constant features
}

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

# inspired by the "useless feature" heuristic of TFDV
# (Breck, Polyzotis, Roy, Whang, Zinkevich, 2019, "Data Validation for Machine Learning",
# SysML/MLSys 2019). TFDV covers many anomaly types; we implement only the simplest:
# a column is "useless" if it has <=1 unique value.

def non_constant(X):
    n_constant = (X.nunique() <= 1).sum()
    p = X.shape[1]
    c_const = 1 - n_constant / p
    score = c_const
    return score, {
        "n_constant": int(n_constant),
        "n_features": int(p)
    }
# reference: Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008).
# "Isolation Forest." 8th IEEE Int. Conf. on Data Mining  2008).
def outlier_percentage(X):
    # how IF works in plain terms:
    # builds 100 random trees that keep splitting the data on random featurus.
    # weird rows get isolted fast (few splits) -> flagged as outlier.
    # normal row need many splits because theyre surrounded by similar rows.
    # contamination='auto' uses Liu et al.'s default threshold so the flagged fraction
    # varies per dataset (not pinned to 5%) -> real variance for CRITIC.
    # sklearn convention: -1 = outlier, +1 = normal (same for all anomaly detectors).
    n_rows = X.shape[0]
    # IsolationForest needs numeric input
    X_num = X.select_dtypes(include="number")
    if X_num.shape[1] == 0:
        return 1.0, {"skipped": "no numeric columns"}

    # subsample wide matrices before fillna+PCA -- on 422k-cols methylation
    # matrices PCA without this takes 10+ minutes per dataset
    X_subset = top_variable_columns(X_num)
    X_filled = X_subset.fillna(X_subset.mean())

    # PCA pre-reduction so IF doesn't choke on the matrix
    k = min(40, n_rows - 1, X_filled.shape[1]) # 40 is educated guess, rows-1 is max for pca,
    X_reduced = PCA(n_components=k, random_state=42).fit_transform(X_filled)
    forest = IsolationForest(contamination='auto', n_estimators=100, random_state=42) #unsupervised -> guess
    labels = forest.fit_predict(X_reduced)
    n_outliers = (labels == -1).sum()
    score = 1 - n_outliers / n_rows # higher outlier percentage -> lower score

    return score, {
        "n_outliers": int(n_outliers),
        "n_components": int(k),
        "n_features_used": int(X_subset.shape[1]),
        "contamination": "auto",
    }

# inspired by Budach et al. (2022), "The Effects of Data Quality on ML Performance
# on Tabular Data", arXiv:2207.14529, "Consistent Representation" dimension (Eq. 1).
# the paper measures minimal replacement operations to make a column consistent
# (e.g. "USA"/"U.S.A."/"United States" -> one canonical form). we use a much
# cheaper proxy: flag object columns where pd.to_numeric introduces new NaNs,
# i.e. columns that mix numeric and string values.
def consistency(X):
    n_features = X.shape[1]
    n_mixed = 0 #columns with >1 unique type (e.g. int and string)
    for col in X.select_dtypes(include="object").columns:
        nans = X[col].isna().sum()
        coerced = pd.to_numeric(X[col], errors="coerce")
        new_Nans = coerced.isna().sum() -nans
        if new_Nans > 0:
            n_mixed += 1
    score = 1 - n_mixed / n_features
    return score, {"n_mixed": int(n_mixed), "n_features": int(n_features)}


def score(dataset):
    X = dataset.X
    c_miss, miss_details = completeness(X)
    c_const, const_details = non_constant(X)
    c_consist, consist_details = consistency(X)
    c_out, out_details = outlier_percentage(X)
    final_score = (
        SUB_WEIGHTS["c_miss"]    * c_miss
        + SUB_WEIGHTS["c_consist"] * c_consist
        + SUB_WEIGHTS["c_out"]     * c_out
        + SUB_WEIGHTS["c_const"]   * c_const
    )

    return SoftRuleResult(
        rule="S3",
        score=final_score,
        details={
            "c_miss": c_miss,
            "c_const": c_const,
            "c_consist": c_consist,
            "c_out": c_out,
            **miss_details,
            **const_details,
            **consist_details,
            **out_details,
        }
    )
