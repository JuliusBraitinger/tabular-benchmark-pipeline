# S5 Batch Effects
# DROPPED: no reliable automated detection method

#TODO revisit if a good detection method is found
import pandas as pd
from pipeline.soft_rules.base import SoftRuleResult

from sklearn.metrics import normalized_mutual_info_score

def score(dataset, pool=None):
    batch = dataset.metadata.get("batch")        # per-sample batch label (saved by the loader)
    if batch is None:
        return SoftRuleResult("S5", 1.0, {"reason": "no batch info"})   # can't assess -> no penalty
    nmi = normalized_mutual_info_score(dataset.y, batch)   # 0 = independent, 1 = identical
    return SoftRuleResult("S5", 1.0 - nmi, {"nmi": round(nmi, 3)})
