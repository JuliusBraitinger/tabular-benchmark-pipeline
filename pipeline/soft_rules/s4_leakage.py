# S4 Data Leakage (20 points)
# group k-fold test, MI spike, distribution shift

#TODO implement score()

from pipeline.soft_rules.base import SoftRuleResult
from scipy.stats import rankdata
import numpy as np
from pipeline.soft_rules.s3_data_quality import top_variable_columns, MAX_FEATURES_FOR_HEAVY_OPS



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
    if task_type == "regression":
        ranks_y = rankdata(np.asarray(y, dtype=float))
        x_centered = ranks_x - ranks_x.mean(axis=0)
        y_centered = ranks_y - ranks_y.mean()
        num = (x_centered * y_centered[:, None]).sum(axis=0)
        denom = np.sqrt((x_centered ** 2).sum(axis=0) * (y_centered ** 2).sum())
        rho = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0) #spearman correlation per feature 
        return 1 - np.abs(rho) # both rho=1 and rho=-1 mean the feature is fully predictive, so take absolute value and flip to [0, 1] range


def check_target_leakage(dataset): #TODO implement
    X = dataset.X.select_dtypes(include="number")
    X = X.loc[:, X.unique() < 1] #drop constant features since they can't leak
    stats = per_feature_stat(X, dataset.y, dataset.task_type)
    top_stats = stats.max() #if any feature is highly predictive of the target, it's suspicious for leakage
    p99 = np.percentile(stats, 99) # look at the 99th percentile to catch cases where one feature is super predictive but others are not

    leak_strength = max(top_stats, p99) 
    sub_score = 1.0 - leak_strength # flip to [0, 1] range where 1.0 means no leakage and 0.0 means strong leakage

    top_idx = np.argsort(-stats)[:5] #indices of the top 5 most predictive features
    top_features = [(str(X.columns[i]), float(stats[i])) for i in top_idx] #get the column names and stats of the top features for the details
    return sub_score, {
        "top_stat": top_stats,
        "p99_stat": p99,
        "gap": top_stats - p99,
        "top_features": top_features,
        "n_features_scored": int(X.shape[1]),
    }

def score(dataset, pool=None):
    score, details = check_target_leakage(dataset)
    return SoftRuleResult(rule="S4", score = score, details={"score": score, "details": details})

    #TODO implement group k-fold test for group leakage 