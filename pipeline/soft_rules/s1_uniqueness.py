# S1 Uniqueness 
#TODO implement

from __future__ import annotations

from pipeline.config import MIN_FEATURES, MIN_ROWS
from pipeline.hard_rules.base import RuleResult

def check_metadata(**_kwargs):
    return RuleResult(rule="S1", passed=True)

def check_data(**_kwargs): #not implemented
    return RuleResult(rule="S1", passed=True)