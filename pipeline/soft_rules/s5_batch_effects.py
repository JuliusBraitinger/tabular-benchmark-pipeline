# S5 Batch Effects
# DROPPED: no reliable automated detection method

#TODO revisit if a good detection method is found
import pandas as pd
from pipeline.soft_rules.base import SoftRuleResult

def score(dataset, pool=None):
    return SoftRuleResult(rule="S5", score=1.0, details={})