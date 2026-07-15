# S5 Batch Effects — per-dataset diagnostic flag.
# A target is only trustworthy if it isn't just a technical batch variable relabelled.
# score = 1 - NMI(target, batch):  1.0 = independent of batch (clean), 0.0 = target IS the batch.
# Needs a per-sample "batch" label in metadata (e.g. mgnify collection date); sources without
# one (openml, uci, ...) return 1.0 -> nothing to assess, no penalty.
from sklearn.metrics import normalized_mutual_info_score
from pipeline.soft_rules.base import SoftRuleResult


def score(dataset, pool=None):
    batch = dataset.metadata.get("batch")        # per-sample batch label saved by the loader
    if batch is None:
        return SoftRuleResult("S5", 1.0, {"reason": "no batch info"})
    nmi = float(normalized_mutual_info_score(dataset.y, batch))   # 0 = independent, 1 = identical
    return SoftRuleResult("S5", 1.0 - nmi, {"nmi": round(nmi, 3)})
