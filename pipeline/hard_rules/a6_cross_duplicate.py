# A6 Cross-Dataset Duplicate
# checks if a dataset shares too many rows with another dataset in the benchmark pool.
# subset detection: if A's rows are mostly contained in B, A is probably a subset of B.
# sorting values per row makes hashes invariant to column order (catches feature-shuffled copies).
#  limitation: sorting also drops column meaning, so two semantically-different rows with
# the same multiset of values will collide. VERY unlikley 

import numpy as np

from pipeline.hard_rules.base import RuleResult

# NaN values are problem -> set to a very small number so that they hash to the same value.
# stays well outside the range of any real measurement but avoids the float64-min overflow
# that the previous -1.7e308 sentinel triggered in pandas hash routines.
_NA_SENTINEL = -1e30

# if >30% of unique row patterns overlap with another accepted dataset -> reject
OVERLAP_THRESHOLD = 0.3


def hash_sorted_rows(dataset):
    # sort each row's values then hash it. column order then doesnt matter.
    # hash row-bytes directly instead of going through pd.util.hash_pandas_object
    # faster
    X = dataset.X.select_dtypes(include="number")
    arr = X.to_numpy(dtype=np.float64, na_value=_NA_SENTINEL, copy=True)
    arr.sort(axis=1)              # in-place row-wise sort
    return {hash(arr[i].tobytes()) for i in range(arr.shape[0])}


def row_overlap(hash_a, hash_b):
    # fraction of unique row patterns in the smaller dataset that also appear in the larger one
    if len(hash_a) == 0 or len(hash_b) == 0:
        return 0.0
    intersection = len(hash_a.intersection(hash_b))
    smaller_size = min(len(hash_a), len(hash_b))
    return intersection / smaller_size


def check(dataset, pool):
    own_hashes = hash_sorted_rows(dataset)

    # check against each pool member, fail on the first one that exceeds the threshold
    for other_id, other_hashes in pool:
        overlap = row_overlap(own_hashes, other_hashes)
        if overlap > OVERLAP_THRESHOLD:
            return RuleResult(
                rule="A6",
                passed=False,
                reason=f"duplicate of {other_id} (overlap={overlap:.2f})",
                details={"matched_id": other_id, "overlap": overlap},
            )

    return RuleResult(rule="A6", passed=True, details={"pool_size": len(pool)})

