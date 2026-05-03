# S3 Data Quality (15 points)
# missing fraction, constant features, outlier percentage

#TODO implement score()

from pipeline.soft_rules.base import SoftRuleResult

def score(dataset, pool=None):
    return SoftRuleResult(rule="S3", score=1.0, details={})
