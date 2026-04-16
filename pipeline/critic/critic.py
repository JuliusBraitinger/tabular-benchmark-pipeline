#Pyhton file that implements the CRITIC method for choosing the best datasets.
#  It uses the AHP weights as a starting point and then adjusts
#  them based on the variability and conflict of the criteria.

#TODO implement the CRITIC method 

import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pipeline.config import TOTAL_POINTS, AHP_WEIGHTS, DIVERGENCE_THRESHOLD


@dataclass(frozen=True)
class CriticResults: #this gives a structured way to store the results of the CRITIC method for each rule and candidate dataset.
    """Results of CRITIC method in a class """
    rule: str #rules that were evaluated, e.g. "S1", "S2", "S3", "S4", "S6" 
    stdv: float #standard deviation of the scores for this rule across all candidates
    ahp_score: float #score coming from AHP method 
    critic_score: float #score comign from CRITIC method 
    critic_weight: float #weight assigned to this rule by the CRITIC method
    final_score: float 
    delta: int #ahp score - critic score
    verdict: str        # "AGREE" if delta <= threshold, else "DIVERGE"
    informativeness: float #measure of how informative this rule is for the final decision, based on the variability and conflict of the scores across candidates.


def run_critic(score_matrix):
    #Step 1 - pick the rules to evaluate (S1, S2, S3, S4, S6)
    rules = ["S1", "S2", "S3", "S4", "S6"]
    scaler = MinMaxScaler()
    normalized_scores= pd.DataFrame(scaler.fit_transform(score_matrix[rules]), columns=rules, index=score_matrix.index)
    standad_deviation = normalized_scores.std()
    correlation_matrix = normalized_scores.corr() #pearson correlation between rules
    conflicts = pd.Series(dtype=float)
    for rule in rules:
    conflicts[rule] = sum(1- correlation_matrix.loc[rule]) 

    informativeness = standad_deviation * conflicts
    total_info= informativeness.sum()
    critic_weights = informativeness / total_info
    critic_scores = {}
    for rule in rules:
        critic_scores[rule] = int(round(critic_weights[rule] * TOTAL_POINTS))
    point_diff = TOTAL_POINTS - sum(critic_scores.values())
    if point_diff != 0:
    # find the rule with the biggest weight
        max_rule = rules[0]
        for rule in rules:
            if critic_weights[rule] > critic_weights[max_rule]:
             max_rule = rule
        critic_scores[max_rule] += point_diff
    #last step compaer to ahp
    results = []
    for rule in rules:
        ahp_points = AHP_WEIGHTS.get(rule, 0)
        crit_points = critic_scores[rule]
        delta = abs(ahp_points - crit_points)
        verdict = "DIVERGE" if delta > DIVERGENCE_THRESHOLD else "AGREE"
        final = ahp_points if verdict == "AGREE" else crit_points
        results.append(CriticResults(
            rule=rule,
            stdv=standad_deviation[rule],
            ahp_score=ahp_points,
            critic_score=crit_points,
            critic_weight=critic_weights[rule],
            final_score=final,
            delta=delta,
            verdict=verdict,
            informativeness=informativeness[rule]
        ))
    return results

