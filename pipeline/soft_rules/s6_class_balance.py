# S6 Class Balance 

#TODO implement


from pipeline.hard_rules.base import RuleResult

def check_metadata(**_kwargs):
    return RuleResult(rule="S1", passed=True)

def check_data(**_kwargs): #not implemented
    return RuleResult(rule="S1", passed=True)