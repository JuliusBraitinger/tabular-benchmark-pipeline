# S2 IID Assumption (10 points)
# duplicate rows, duplicate groups, duplicate fraction


from pipeline.soft_rules.base import SoftRuleResult
import pandas as pd


def hash_rows(dataset):
    # reset indices so concat aligns by position, not by index labels
    x = dataset.X.reset_index(drop=True)
    y = dataset.y.reset_index(drop=True)
    combined = pd.concat([x, y], axis=1)
    hashed = pd.util.hash_pandas_object(combined, index=False)
    return hashed

def find_duplicates(dataset):
    hashed = hash_rows(dataset)
    counter = hashed.value_counts()
    duplicates_groups = counter[counter > 1]
    n_rows = len(hashed)
    n_duplicates_rows = int(duplicates_groups.sum())  # every row that belongs to any duplicate group
    return {
        "n_rows": n_rows,
        "n_duplicate_rows": n_duplicates_rows,
        "n_duplicate_groups": int(len(duplicates_groups)),
        "duplicate_fraction": n_duplicates_rows / n_rows if n_rows else 0.0,
    }

 

def score(dataset, pool=None):
    duplicate_info =find_duplicates(dataset)
    return SoftRuleResult(
        rule = "S2",
        score= 1.0 - duplicate_info["duplicate_fraction"],
        details=duplicate_info,
    )
 