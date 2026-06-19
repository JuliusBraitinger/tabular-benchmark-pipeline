# A3: does the dataset have real predictive signal, or is it just noise?
#
# Method (Ojala & Garriga 2010, "Permutation Tests for Studying Classifier
# Performance", JMLR 11:1833-1863): train a random forest with cross-validation,
# then re-score on 100 random label shuffles. p < 0.05 means the real score beats
# the shuffled ones, so the signal is real. sklearn provides permutation_test_score.
#
# Metric: classification uses adjusted balanced accuracy, regression uses R2
# (both have chance = 0; see the threshold notes below).
#
# TODO: run TabPFN-2 to check whether the signal is TOO good -> flag / drop.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier, TabPFNRegressor

from pipeline.hard_rules.base import RuleResult

P_VALUE_THRESHOLD = 0.05  # signal must be statistically distinguishable from random
# absolute-score floor (the "strong enough" gate, on top of the p-value).
# Classification uses ADJUSTED balanced accuracy: (BA - 1/K) / (1 - 1/K), so chance = 0
# and perfect = 1 regardless of the number of classes K. Raw balanced accuracy has chance
# at 1/K, so a fixed 0.55 floor  rejected multiclass datasets that were far above
# chance (e.g. 50-class amazon-commerce-reviews at BA=0.42 ≈ 21x chance).  adjustd
# scale makes one threshold mean the same across binary/multiclass and matches regression's
# R2 (chance = 0). See Brodersen et al. 2010; Ojala & Garriga 2010.
MIN_CLF_SCORE = 0.10  # adjusted balanced accuracy (0 = chance, 1 = perfect)
MIN_REG_SCORE = 0.05
# dataset that scores near-perfect score -> reject it (trivial signal).
MAX_CLF_SCORE = 0.98  # adjusted balanced accuracy
MAX_REG_SCORE = 0.98
N_PERMUTATIONS = 100
N_ESTIMATORS = 50
MAX_DEPTH = 5
CV_FOLDS = 3
# TabPFN-2 limits (used to confirm trivial-signal verdicts from the RF)
TABPFN_MAX_FEATURES = 500



def check_data(X, y, task_type="classification", **_kwargs):
    """Permutation test: if p < 0.05, the data has real signal."""

    # drop rows with missing target — LabelEncoder + permutation_test_score
    # both fail on NaN. Must happen before any inference / encoding.
    mask = pd.notna(y)
    if not mask.all():
        X = X.loc[mask]
        y = y.loc[mask]

    if len(y) == 0:
        return RuleResult(rule="A3", passed=False, reason="all target values were NaN")

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
        # adjusted=True -> chance-corrected (0 = random, 1 = perfect) regardless of #classes
        scoring = make_scorer(balanced_accuracy_score, adjusted=True)
        metric_name = "adj_balanced_accuracy"
    else:
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=-1
        )
        scoring = "r2"
        metric_name = "r2"

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

    # pick the thresholds: classification uses balanced_acc, regression uses R2
    if "classification" in task_type:
        min_score = MIN_CLF_SCORE
        max_score = MAX_CLF_SCORE
    else:
        min_score = MIN_REG_SCORE
        max_score = MAX_REG_SCORE

    # checks in order: signal must be real, strong enough, but not trivial
    tabpfn_score = None
    if p_value >= P_VALUE_THRESHOLD:
        passed = False
        reason = f"no signal (p={p_value:.3f}, score={real_score:.3f})"
    elif real_score < min_score:
        passed = False
        reason = f"signal too weak ({metric_name}={real_score:.3f} < {min_score})"
    elif real_score > max_score:
        # Tabpfn is exepensive -> only run it if RF has trivial signal
        tabpfn_score = tabpfn_scorer(X, y, task_type, scoring)
        if tabpfn_score > max_score:
            passed = False
            reason = f"trivial: RF={real_score:.3f}, TabPFN={tabpfn_score:.3f} both > {max_score}"
        else:
            # RF found it easy but TabPFN didn't -- probably RF-specific, keep dataset
            passed = True
            reason = "not trivial"
    else:
        passed = True
        reason = "not trivial and not too weak"

    return RuleResult(
        rule="A3",
        passed=passed,
        reason=reason,
        details={
            "p_value": p_value,
            "real_score": real_score,
            "min_score": min_score,
            "max_score": max_score,
            "tabpfn_score": tabpfn_score,
        },
    )


def tabpfn_scorer(X, y, task_type, scoring):

    #TODO maybe include tabpfnwide?
    if X.shape[1] > TABPFN_MAX_FEATURES:
        var_arr = np.nanvar(X.values, axis=0)
        top_idx = np.argpartition(-var_arr, TABPFN_MAX_FEATURES)[:TABPFN_MAX_FEATURES]
        X = X.iloc[:, top_idx]

    if "classification" in task_type:
        model = TabPFNClassifier()
    else:
        model = TabPFNRegressor()
    scores = cross_val_score(model, X, y, scoring=scoring, cv=CV_FOLDS)
    return float(scores.mean())
