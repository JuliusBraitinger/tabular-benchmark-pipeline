# S4 Data Leakage (20 points)
# group k-fold test, MI spike, distribution shift

#TODO implement score()

from pipeline.soft_rules.base import SoftRuleResult

def score(dataset, pool=None):
    return SoftRuleResult(rule="S4", score=1.0, details={})
