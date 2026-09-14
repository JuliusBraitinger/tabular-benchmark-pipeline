# S1 Uniqueness (10 points)
# statistical fingerprint: per-column mean+std+skew+kurtosis
# inspired by the statistical meta-feature group of pymfe
# (Alcobaça et al., 2020, "MFE: Towards reproducible meta-feature extraction", JMLR 21:111).
# we do not call pymfe directly; we re-use the concept of summarizing per-column
# moments into a fixed-length fingerprint, but the specific 28-dim vector
# and cosine-similarity scoring are our own.
#TODO vllt noch ein paar weitere stats hinzufügen ( missing values, wenn datei parquet gleich ist -> gleicher datensatz, checke auch zeilen etc )
#subset detecion mit sortierung von werten pro zeile -> kein effect von feature scrample -> schaue auf intersection von den datensätzen 
#datasets need to be thrown out und auch drauf achten wen subsets von datensätzen gleich sind -> mögen wir nicht 
import numpy as np
from scipy import stats as scipy_stats

from pipeline.soft_rules.base import SoftRuleResult


QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def summarize(values):  # collapse  per-column stats into 7 fixed numbers so fingerprints are comparable across datasets with different Features
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)] #filter out NAN
    if len(arr) == 0:
        return np.zeros(7)
    quants = np.quantile(arr, QUANTILES)
    return np.concatenate([quants, [arr.mean(), arr.std()]])


def compute_fingerprint(dataset):
    X = dataset.X.select_dtypes(include="number")
    arr = X.to_numpy(dtype=float, copy=False)

    # constant columns give nan skew/kurtosis and columns with <2 values give a
    # nan std. summarize() drops those nans, so the four moment blocks would each
    # describe a different set of columns. drop them once here instead.
    arr = arr[:, (~np.isnan(arr)).sum(axis=0) >= 2]
    arr = arr[:, np.nanstd(arr, axis=0) > 0]

    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0, ddof=1)
    skews = scipy_stats.skew(arr, axis=0, nan_policy="omit")
    kurts = scipy_stats.kurtosis(arr, axis=0, nan_policy="omit")

    return np.concatenate([
        summarize(means),
        summarize(stds),
        summarize(skews),
        summarize(kurts),
    ])


def score(dataset, pool_fingerprints=None):
    if pool_fingerprints is not None and dataset.id in pool_fingerprints:
        own_fingerprint = pool_fingerprints[dataset.id]
    else:
        own_fingerprint = compute_fingerprint(dataset)

    if pool_fingerprints is None or len(pool_fingerprints) < 2:
        return SoftRuleResult(rule="S1", score=1.0, details={
            "meta_fingerprint": own_fingerprint.tolist(),
            "max_similarity": 0.0,
        })
    # standardize each of the 28 dimensions across the pool so mean, std,
    # skew and kurtosis all contribute equally to the cosine similarity
    pool_matrix = np.array(list(pool_fingerprints.values()))
    pool_mean = pool_matrix.mean(axis=0)
    pool_std = pool_matrix.std(axis=0) + 1e-9
    own_normalized = (own_fingerprint - pool_mean) / pool_std

    max_similarity = 0.0
    for other_id, other_fingerprint in pool_fingerprints.items():
        if other_id == dataset.id:
            continue
        other_normalized = (other_fingerprint - pool_mean) / pool_std
        similarity = np.dot(own_normalized, other_normalized) / (
            np.linalg.norm(own_normalized) * np.linalg.norm(other_normalized)
        )
        if similarity > max_similarity:
            max_similarity = similarity

    score_value = max(0.0, min(1.0, 1.0 - max_similarity))

    return SoftRuleResult(rule="S1", score=score_value, details={
        "meta_fingerprint": own_fingerprint.tolist(),
        "max_similarity": float(max_similarity),
    })

