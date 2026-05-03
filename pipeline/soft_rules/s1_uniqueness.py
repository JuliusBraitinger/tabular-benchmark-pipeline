# S1 Uniqueness (10 points)
# statistical fingerprint: per-column mean+std+skew+kurtosis
# reference: pymfe paper (Alcobaca et al., 2020)

#TODO implement score()
import pandas as pd
from pipeline.soft_rules.base import SoftRuleResult

def score(dataset):
    stdv = dataset.std()
    mean = dataset.mean()
    skew = dataset.skew()
    kurt = dataset.kurtosis()
    fingerprint = pd.concat([mean, stdv, skew, kurt], axis=1)
    return SoftRuleResult(rule="S1", score=1.0, details={"fingerprint": fingerprint}) #score is placeholder for now
