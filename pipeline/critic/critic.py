from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pipeline.config import TOTAL_POINTS, AHP_WEIGHTS, DIVERGENCE_THRESHOLD


@dataclass(frozen=True)
class CRITICResults: #this gives a structured way to store the results of the CRITIC method for each rule and candidate dataset.
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
    rules = ["S1", "S2", "S3", "S4", "S6"]

    # impute NaN cells with column medians instead of dropping the whole row.
    # avoids artificially inflating S6's variance (regression rows would otherwise be dropped,
    # leaving only classification rows where S6 happens to spread widely).
    sub = score_matrix[rules].fillna(score_matrix[rules].median())
    informative = [r for r in rules if sub[r].std() > 1e-9] # rules with variance; constants carry no CRITIC signal

    if len(informative) < 2: #if there are fewer than 2 informative rules, can't really apply the CRITIC method, so just return the AHP scores and mark everything as "AGREE"
        results = []
        for rule in rules:
            ahp = AHP_WEIGHTS.get(rule, 0)
            results.append(CRITICResults(
                rule=rule,
                stdv=0.0,
                ahp_score=ahp,
                critic_score=ahp,
                critic_weight=0.0,
                final_score=ahp,
                delta=0,
                verdict="AGREE",
                informativeness=0.0,
            ))
        return results

    scaler = MinMaxScaler()
    normalized_scores = pd.DataFrame(
        scaler.fit_transform(sub[informative]),
        columns=informative,
        index=sub.index,
    )

    standard_deviation = normalized_scores.std()
    correlation_matrix = normalized_scores.corr()
    conflicts = (1 - correlation_matrix).sum()
    informativeness = standard_deviation * conflicts

    total_info = informativeness.sum()
    critic_weights = pd.Series(0.0, index=rules) # start every rule at 0
    critic_weights.loc[informative] = informativeness / total_info #only informative rules get a real weight; degenerates stay at 0

    critic_scores = {rule: int(round(critic_weights[rule] * TOTAL_POINTS)) for rule in rules}
    point_diff = TOTAL_POINTS - sum(critic_scores.values())
    if point_diff != 0:
        max_rule = informative[0] # rounding leftover lands on the biggest informative rule, never on a degenerate one
        for rule in informative:
            if critic_weights[rule] > critic_weights[max_rule]:
                max_rule = rule
        critic_scores[max_rule] += point_diff

    results = []
    for rule in rules:
        ahp_points = AHP_WEIGHTS.get(rule, 0)
        crit_points = critic_scores[rule]
        delta = abs(ahp_points - crit_points)
        if rule in informative:
            verdict = "DIVERGE" if delta > DIVERGENCE_THRESHOLD else "AGREE"
            final = ahp_points if verdict == "AGREE" else crit_points
            stdv = float(standard_deviation[rule])
            info_val = float(informativeness[rule])
        else: # degenerate rule: no CRITIC signal, fall back to AHP
            verdict = "AGREE"
            final = ahp_points
            stdv = 0.0
            info_val = 0.0
        results.append(CRITICResults(
            rule=rule,
            stdv=stdv,
            ahp_score=ahp_points,
            critic_score=crit_points,
            critic_weight=float(critic_weights[rule]),
            final_score=final,
            delta=delta,
            verdict=verdict,
            informativeness=info_val,
        ))
    return results


def final_weights(critic_results):
    return {r.rule: r.final_score for r in critic_results}
