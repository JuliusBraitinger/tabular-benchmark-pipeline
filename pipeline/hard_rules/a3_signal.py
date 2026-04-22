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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import permutation_test_score

from pipeline.hard_rules.base import RuleResult

P_VALUE_THRESHOLD = 0.05 #actually dont know the correct values just cheking like that 
N_PERMUTATIONS = 100
N_ESTIMATORS = 50
MAX_DEPTH = 5
CV_FOLDS = 3


def check_metadata(**_kwargs: object) -> None:
    # can't check signal from metadata alone, defer to data check
    return None


def check_data(X, y, task_type="classification", **_kwargs):
    """Permutation test: if p < 0.05, the data has real signal."""

    # pick the right model and metric based on task type
    if task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42
        )
        scoring = "roc_auc_ovr"
    else:
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42
        )
        scoring = "r2"

    # subsampling
    if len(X) > 5000:
         sample_idx = X.sample(n=5000, random_state=42).index
            X = X.loc[sample_idx]
            y = y.loc[sample_idx]

    # replace NaNs with median? -> from paper
    X = X.fillna(X.mean())

    # trains model on real labels, then shuffles labels N_PERMUTATIONS times. 
    real_score, perm_scores, p_value = permutation_test_score(
        model, X, y,
        scoring=scoring,
        cv=CV_FOLDS,
        n_permutations=N_PERMUTATIONS,
        random_state=42,
    )

    passed = p_value < P_VALUE_THRESHOLD
    return RuleResult(
        rule="A3",
        passed=passed,
        reason="" if passed else f"no signal detected (p={p_value:.3f}, score={real_score:.3f})",
        details={"p_value": float(p_value), "real_score": float(real_score)},
    )
