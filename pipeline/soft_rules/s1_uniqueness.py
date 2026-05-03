# S1 Uniqueness (10 points)
# statistical fingerprint: per-column mean+std+skew+kurtosis
# reference: pymfe paper (Alcobaca et al., 2020)

#TODO implement score()
import pandas as pd
from pipeline.soft_rules.base import SoftRuleResult

def score(dataset):
    stdv = dataset.X.std()
    mean = dataset.X.mean()
    skew = dataset.X.skew()
    kurt = dataset.X.kurtosis()
    fingerprint = pd.concat([mean, stdv, skew, kurt], axis=0)
    return SoftRuleResult(rule="S1", score=1.0, details={"fingerprint": fingerprint}) #score is placeholder for now
