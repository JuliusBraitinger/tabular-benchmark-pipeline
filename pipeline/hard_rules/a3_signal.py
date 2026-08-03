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
# TODO: precision/recall for imbalanced datasets . Achtung random baseline
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabpfn import TabPFNClassifier, TabPFNRegressor

from pipeline.hard_rules.base import RuleResult
from pipeline.viz.reports import make_cv, make_reference_model

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
# TabPFN-2 limits (used to confirm trivial-signal verdicts from the RF)
TABPFN_MAX_FEATURES = 500
TABPFN_MAX_ROWS = 1000  # TabPFN refuses more than this on CPU
DEVICE = os.getenv("TABPFN_DEVICE", "cpu")  # mps runs out of memory even on small data
# Permutation test: if p < 0.05, the data has real signal.
def check_data(X, y, task_type="classification", **_kwargs):
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
    model = make_reference_model(task_type)
    if "classification" in task_type:
        # adjusted=True -> chance-corrected (0 = random, 1 = perfect) regardless of #classes
        scoring = make_scorer(balanced_accuracy_score, adjusted=True)
        metric_name = "adj_balanced_accuracy"
    else:
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
        model, X, y, scoring=scoring, cv=make_cv(task_type),
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
    ceiling_score = None
    if np.isnan(real_score):  # fail closed: nan loses every comparison below
        passed = False
        reason = f"{metric_name} undefined (nan)"
    elif p_value >= P_VALUE_THRESHOLD:
        passed = False
        reason = f"no signal (p={p_value:.3f}, score={real_score:.3f})"
    elif real_score < min_score:
        passed = False
        reason = f"signal too weak ({metric_name}={real_score:.3f} < {min_score})"
    else:
        # TabPFN decides trivial, not the gate model: the gate is too weak to reach
        # the ceiling on multiclass (a label copy only scored 0.41 at K=50)
        ceiling_score = ceiling_scorer(X, y, task_type, scoring)
        if ceiling_score is not None and ceiling_score > max_score:
            passed = False
            reason = f"trivial: ceiling={ceiling_score:.3f} > {max_score}"
        else:
            passed = True  # None = both models failed, keep the dataset
            # say so out loud when nothing ran, else a dead ceiling looks like a pass
            shown = "none ran" if ceiling_score is None else f"{ceiling_score:.3f}"
            reason = f"not trivial (ceiling={shown}) and not too weak ({metric_name}={real_score:.3f})"

    return RuleResult(
        rule="A3",
        passed=passed,
        reason=reason,
        details={
            "p_value": p_value,
            "real_score": real_score,
            "min_score": min_score,
            "max_score": max_score,
            "ceiling_score": ceiling_score,
        },
    )

def ceiling_scorer(X, y, task_type, scoring):
    linear = make_pipeline(
        VarianceThreshold(1e-8), StandardScaler(),
        LogisticRegression(max_iter=1000) if "classification" in task_type else Ridge())
    scores = [cross_val_score(linear, X, y, scoring=scoring, cv=make_cv(task_type)).mean()]

    if X.shape[1] > TABPFN_MAX_FEATURES:
        var_arr = np.nanvar(X.values, axis=0)
        top_idx = np.argpartition(-var_arr, TABPFN_MAX_FEATURES)[:TABPFN_MAX_FEATURES]
        X = X.iloc[:, top_idx]

    if len(X) > TABPFN_MAX_ROWS:
        idx = np.random.default_rng(42).choice(len(X), TABPFN_MAX_ROWS, replace=False) #pick random rows to avoid biasing the score
        X, y = X.iloc[idx], np.asarray(y)[idx] #only work with these rows

    if "classification" in task_type:
        model = TabPFNClassifier(device=DEVICE)
    else:
        model = TabPFNRegressor(device=DEVICE)
    try: #try to score TabPFN, but it fails on some datasets (e.g. 0 variance)
        scores.append(cross_val_score(model, X, y, scoring=scoring, cv=make_cv(task_type)).mean())
    except Exception:
        pass

    scores = [s for s in scores if np.isfinite(s)] #filter out nan/inf scores, which happen on some datasets (e.g. 0 variance)
    return float(max(scores)) if scores else None
