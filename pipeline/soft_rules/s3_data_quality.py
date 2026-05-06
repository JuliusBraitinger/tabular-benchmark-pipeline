# S3 Data Quality (15 points)
# missing fraction, constant features, outlier percentage
#
# Composite score over  sub-metrics; each sub-metric is in [0, 1] (1 = clean).
# Budach et al. (2022), Eq. for Completeness:
#        c_miss = 1 - (1/p) * sum_j ( missing(c_j) / n )

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import pandas as pd

from pipeline.soft_rules.base import SoftRuleResult

# Sub-metric weights for the final S3 score.
#  JUST A EDUCATED GUESS NEEDS IMPROVEMENT 
SUB_WEIGHTS = {
    "c_miss":    0.30,  # completeness 
    "c_consist": 0.20,  # consistent representation
    "c_out":     0.20,  # outliers
    "c_const":   0.15,  # constant features 
    "c_uniq":    0.15,  # duplicates 
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

# reference: Breck, E., Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2019).
# "Data Validation for Machine Learning." SysML 2019. (TFDV "useless feature" check)
# checking if columns are constant -> not informative + checking for mix types

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
    X_filled = X_num.fillna(X_num.mean())

    # PCA pre-reduction so IF doesn't choke on 100k+ feature matrices
    k = min(40, n_rows - 1, X_filled.shape[1]) # 40 is educated guess, rows-1 is max for pca,
    X_reduced = PCA(n_components=k, random_state=42).fit_transform(X_filled)
    forest = IsolationForest(contamination='auto', n_estimators=100, random_state=42) #unsupervised -> guess
    labels = forest.fit_predict(X_reduced)
    n_outliers = (labels == -1).sum()
    score = 1 - n_outliers / n_rows # higher outlier percentage -> lower score

    return score, {
        "n_outliers": int(n_outliers),
        "n_components": int(k),
        "contamination": "auto",
    }

# reference: Budach, L. et al. (2022). "The Effects of Data Quality on Machine
# Learning Performance on Tabular Data." arXiv:2207.14529 (Consistent Representation dim.) 
#checking for columns with mixed types/values 
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


# reference: Budach, L. et al. (2022). "The Effects of Data Quality on Machine
# Learning Performance on Tabular Data." arXiv:2207.14529, Eq. (10).
def uniquness(X, Y): #checks for duplicate rows including target -> non-unique = lower quality
    dataframe = X.assign(target=Y)
    n_total = dataframe.shape[0]
    n_unique = pd.util.hash_pandas_object(dataframe).nunique()
    score = n_unique / n_total
    return score, {"n_unique": int(n_unique), "n_total": int(n_total)}


def score(dataset):
    X = dataset.X
    c_miss, miss_details = completeness(X)
    c_const, const_details = non_constant(X)
    c_consist, consist_details = consistency(X)
    c_out, out_details = outlier_percentage(X)
    c_uniqueness, uniqueness_details = uniquness(X, dataset.y)
    final_score = (
        SUB_WEIGHTS["c_miss"]    * c_miss
        + SUB_WEIGHTS["c_consist"] * c_consist
        + SUB_WEIGHTS["c_out"]     * c_out
        + SUB_WEIGHTS["c_const"]   * c_const
        + SUB_WEIGHTS["c_uniq"]    * c_uniqueness
    )

    return SoftRuleResult(
        rule="S3",
        score=final_score,
        details={
            "c_miss": c_miss,
            "c_const": c_const,
            "c_consist": c_consist,
            "c_out": c_out,
            "c_uniqueness": c_uniqueness,
            **miss_details,
            **const_details,
            **consist_details,
            **out_details,
            **uniqueness_details,
        }
    )
