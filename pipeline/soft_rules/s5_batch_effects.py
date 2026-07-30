# S5 Batch Effects — per-dataset diagnostic flag.
# A target is only trustworthy if it isn't just a technical batch variable relabelled.
# score = 1 - NMI(target, batch) 0= target IS the batch. 1= good
# Needs a per-sample "batch" label in metadata (e.g. mgnify collection date); sources without
# one (openml, uci, ...) return 1.0 -> so dont use it on them (they are not batch-effect-prone).
from sklearn.metrics import normalized_mutual_info_score
from pipeline.soft_rules.base import SoftRuleResult


def score(dataset, pool=None):
    batch = dataset.metadata.get("batch")        # per-sample batch label saved by the loader
    if batch is None:
        return SoftRuleResult("S5", float("nan"), {"reason": "no batch info"})
    nmi = float(normalized_mutual_info_score(dataset.y, batch))   
    return SoftRuleResult("S5", 1.0 - nmi, {"nmi": round(nmi, 3)})
