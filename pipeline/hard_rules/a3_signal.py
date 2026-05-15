# A3: check if the dataset actually has a signal or if its just noise
#
# based on: Ojala & Garriga (2010) - "Permutation Tests for Studying Classifier Performance"
# published in JMLR, vol 11, pages 1833-1863
#
# 
# 1. train a random forest on the real labels with cross validation and get a score
# 2. then shuffle the labels randomly 100 times and retrain each time
# 3. if the real score is way better than the shuffled ones, theres actual signal
# 4.  measure this with a p-value. p < 0.05 means the signal is real
#
# for classification  use AUC as metric, for regression R2
# sklearn already permutation_test_score
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import LabelEncoder

from pipeline.hard_rules.base import RuleResult

P_VALUE_THRESHOLD = 0.05 #actually dont know the correct values just cheking like that 
N_PERMUTATIONS = 100
N_ESTIMATORS = 50
MAX_DEPTH = 5
CV_FOLDS = 3



def check_data(X, y, task_type="classification", **_kwargs):
    """Permutation test: if p < 0.05, the data has real signal."""

    # if task_type is unknown, infer it from y: non-numeric or few unique values
    # -> classification (string labels like 'SITTING', 'WALKING' fall here)
    if task_type == "unknown" or not task_type:
        if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20:
            task_type = "classification"
        else:
            task_type = "regression"

    # pick the right model and metric based on task type
    if "classification" in task_type:
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=-1
        )
        scoring = "balanced_accuracy"
    else:
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=-1
        )
        scoring = "r2"

    # subsampling
    if len(X) > 5000:
        sample_idx = X.sample(n=5000, random_state=42).index
        X = X.loc[sample_idx]
        y = y.loc[sample_idx]

    # some openml datasets come as sparse, convert to normal dense format
    if hasattr(X, "sparse"):
        X = X.sparse.to_dense()

    # drop non-numeric columns (some openml datasets have strings mixed in)
    X = X.select_dtypes(include="number")

    # sklearn rejects DataFrames whose column names mix int and str types
    X = X.rename(columns=str)

    # subsample top 1000 most variable features before fillna so the rest of
    # the pipeline only operates on a 1000-column matrix
    if X.shape[1] > 1000:
        var_arr = np.nan_to_num(np.nanvar(X.values, axis=0), nan=-1.0)
        top_idx = np.argpartition(-var_arr, 1000)[:1000]
        X = X.iloc[:, top_idx]

    # replace NaNs with column means
    X = X.fillna(X.mean())

    # sklearn needs numeric labels, some datasets have strings like "tumor"/"normal"
    if "classification" in task_type:
        y = LabelEncoder().fit_transform(y)

    # run the permutation test - trains model on real labels, then shuffles labels N_PERMUTATIONS times
    results = permutation_test_score(
        model, X, y, scoring=scoring, cv=CV_FOLDS,
        n_permutations=N_PERMUTATIONS, random_state=42, n_jobs=-1,
    )
    real_score = results[0]  # how well the model did on real labels
    p_value = results[2]     # fraction of shuffled runs that beat the real score

    passed = p_value < P_VALUE_THRESHOLD
    return RuleResult(
        rule="A3",
        passed=passed,
        reason="" if passed else f"no signal detected (p={p_value:.3f}, score={real_score:.3f})",
        details={"p_value": float(p_value), "real_score": float(real_score)},
    )
