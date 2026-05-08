# S1 Uniqueness (10 points)
# statistical fingerprint: per-column mean+std+skew+kurtosis
# reference: pymfe paper (Alcobaca et al., 2020)

import numpy as np

from pipeline.soft_rules.base import SoftRuleResult


QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def summarize(values):
    quants = [values.quantile(q) for q in QUANTILES]
    return np.array(quants + [values.mean(), values.std()])


def compute_fingerprint(dataset):
    X = dataset.X.select_dtypes(include="number")

    means = X.mean()
    stds = X.std()
    skews = X.skew()
    kurts = X.kurtosis()

    return np.concatenate([
        summarize(means),
        summarize(stds),
        summarize(skews),
        summarize(kurts),
    ])


def score(dataset, pool_fingerprints=None):
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

