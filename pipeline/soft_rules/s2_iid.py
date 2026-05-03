# S2 IID Assumption (10 points)
# autocorrelation, runs test, duplicate detection

#TODO implement score()

from pipeline.soft_rules.base import SoftRuleResult

def score(dataset, pool=None):
    return SoftRuleResult(rule="S2", score=1.0, details={})
