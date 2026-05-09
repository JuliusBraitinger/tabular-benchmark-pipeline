# S4 Data Leakage (20 points)
# group k-fold test, MI spike, distribution shift

#TODO implement score()

from pipeline.soft_rules.base import SoftRuleResult
from scipy.stats import rankdata
import numpy as np
from sklearn.metrics import roc_auc_score
from pipeline.soft_rules.s3_data_quality import top_variable_columns, MAX_FEATURES_FOR_HEAVY_OPS

SUB_WEIGHTS = {"A": 0.5, "B": 0.5} #JUST A EDUCATED GUESS NEEDS IMPROVEMENT

def per_feature_stat(X,y,task_type):
    #one predictive stat per feature in  0,1 Higher is more predictive of the target, which is a sign of potential leakage.
    array = X.to_numpy(dtype=float, copy=False)
    col_means = np.nanmean(array, axis=0)
    array = np.where(np.isnan(array), col_means, array) #impute missing values with column means so they don't mess up the stats
    ranks_x = rankdata(array,axis=0) 

    if task_type == "classification":
        y_array = np.asarray(y)
        classes, counts =  np.unique(y_array, return_counts=True)
        #most frequent class vs rest, multiclass gets an AUC too (from paper)
        positive = classes[np.argmax(counts)]
        y_bin = (y_array == positive).astype(int)
        n_pos = y_bin.sum()
        n_neg = len(y_bin) - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.zeros(X.shape[1]) # only one class, so no feature can be predictive of the target
        sum_pos_ranks = np.sum(ranks_x[y_bin == 1], axis=0)
        auc = np.array([roc_auc_score(y_bin, X.iloc[:, j]) for j in range(X.shape[1])])
        return np.minimum(auc, 1-auc) #if a feature is perfectly predictive of the target, it could be a sign of leakage, but if it's perfectly anti-predictive that's also a sign of leakage, so we take the minimum of AUC and 1-AUC to capture both cases.

def score(dataset, pool=None):
    return SoftRuleResult(rule="S4", score=1.0, details={})
