# S6 Class Balance (5 points)
# imbalance ratio and entropy of class distribution

#TODO implement score()

from pipeline.soft_rules.base import SoftRuleResult

def score(dataset, pool=None):
    return SoftRuleResult(rule="S6", score=1.0, details={})
