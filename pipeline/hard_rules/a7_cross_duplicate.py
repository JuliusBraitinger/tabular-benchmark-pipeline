# A7 Cross-Dataset Duplicate
# checks if a dataset shares too many rows with another dataset in the benchmark pool.
# subset detection: if A's rows are mostly contained in B, A is probably a subset of B.
# sorting values per row makes hashes invariant to column order (catches feature-shuffled copies).

import pandas as pd


def hash_sorted_rows(dataset):
    # sort each row's values then hash it. column order then doesnt matter.
    # nan -> 0 because pd.util.hash_pandas_object treats nans weirdly otherwise
    X = dataset.X.select_dtypes(include="number").fillna(0)

    arr = X.to_numpy(copy=True)   # copy so .sort() doesnt mutate the original df
    arr.sort(axis=1)              # in-place row-wise sort

    sorted_df = pd.DataFrame(arr)
    hashes = pd.util.hash_pandas_object(sorted_df, index=False)
    return set(hashes.tolist())



