# A7 Cross-Dataset Duplicate
# checks if a dataset shares too many rows with another dataset in the benchmark pool.
# subset detection: if A's rows are mostly contained in B, A is probably a subset of B.
# sorting values per row makes hashes invariant to column order (catches feature-shuffled copies).
#  limitation: sorting also drops column meaning, so two semantically-different rows with
# the same multiset of values will collide. VERY unlikley 

import pandas as pd

# NaN values are problem -> set to a very small number so that they hash to the same value
_NA_SENTINEL = -1.7e308


def hash_sorted_rows(dataset):
    # sort each row's values then hash it. column order then doesnt matter.
    X = dataset.X.select_dtypes(include="number").fillna(_NA_SENTINEL)

    arr = X.to_numpy(copy=True)   # copy so .sort() doesnt mutate the original df
    arr.sort(axis=1)              # in-place row-wise sort
 
    sorted_df = pd.DataFrame(arr)
    hashes = pd.util.hash_pandas_object(sorted_df, index=False)
    return set(hashes.tolist())


def row_overlap(hash_a, hash_b):
# fraction of unique row patterns in the smaller dataset that also appear in the larger one
    if len(hash_a) == 0 or len(hash_b) == 0:
        return 0.0
    intersection = len(hash_a.intersection(b))
    smaller_size = min(len(hash_a), len(hash_b))
    return intersection / smaller_size

