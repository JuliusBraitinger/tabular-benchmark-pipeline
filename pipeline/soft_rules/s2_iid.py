# S2 IID Assumption (10 points)
# autocorrelation, runs test, duplicate detection

#TODO implement score()

from pipeline.soft_rules.base import SoftRuleResult
import pandas as pd


def hash_rows(dataset):
    x= dataset.X
    y= dataset.y
    combined = pd.concat([x, y], axis=1)
    hashed = pd.util.hash_pandas_object(combined, index=False)
    return hashed 

def find_duplicates(dataset):
    hashed = hash_rows(dataset)
    counter = hashed.value_counts()
    duplicates_groups = counter[counter > 1]
    n_rows = len(hashed)
    n_duplidates_rows = duplicates_groups.sum() #every row that belongs to any duplicate group
    return {
        "n_rows": n_rows,
        "n_duplicate_rows": n_duplidates_rows,
        "n_duplicate_groups": int(len(n_duplidates_rows)),
        "duplicate_fraction": n_duplidates_rows / n_rows
    }
def score(dataset, pool=None):
    return SoftRuleResult(rule="S2", score=1.0, details={})
