# S4 Data Leakage (20 points)
# group k-fold test, MI spike, distribution shift

#TODO implement score()

from pipeline.soft_rules.base import SoftRuleResult
from scipy.stats import rankdata
import numpy as np
from pipeline.soft_rules.s3_data_quality import top_variable_columns, MAX_FEATURES_FOR_HEAVY_OPS

SUB_WEIGHTS = {"A": 0.5, "B": 0.5} #JUST A EDUCATED GUESS NEEDS IMPROVEMENT

def per_feature_stat(X, y, task_type):
    # per-feature predictive stat in [0, 1]. 1.0 = uninformative (clean),
    # 0.0 = perfectly predictive of the target (suspicious for leakage).
    array = X.to_numpy(dtype=float, copy=False)
    col_means = np.nanmean(array, axis=0)
    array = np.where(np.isnan(array), col_means, array)
    ranks_x = rankdata(array, axis=0)

    if task_type == "classification":
        y_array = np.asarray(y)
        classes, counts = np.unique(y_array, return_counts=True)
        # most frequent class vs rest (one-vs-rest with majority as positive)
        positive = classes[np.argmax(counts)]
        y_bin = (y_array == positive).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = len(y_bin) - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.ones(X.shape[1])  # only one class -> nothing to predict, treat as clean
        # Mann-Whitney AUC per feature, vectorised over all columns.
        sum_pos_ranks = np.sum(ranks_x[y_bin == 1], axis=0)
        auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        # both AUC=1 and AUC=0 mean the feature is fully predictive
        return 2 * np.minimum(auc, 1 - auc)

    # toDO regression
    return None


def check_target_leakage(X, y, task_type): #TODO implement
    return per_feature_stat(X, y, task_type)

def score(dataset, pool=None):
    return SoftRuleResult(rule="S4", score=1.0, details={})