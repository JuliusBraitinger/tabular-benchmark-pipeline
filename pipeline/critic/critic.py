import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pipeline.config import TOTAL_POINTS, AHP_WEIGHTS, DIVERGENCE_THRESHOLD, PASS_THRESHOLD, WEIGHTS_PATH, AHP_PART


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


RULES = ["S1", "S2", "S3", "S4", "S5", "S6"] 


def run_critic(score_matrix):
    # weight each soft rule by how much its scores vary and how independent it is, then blend with AHP.
    # impute NaN cells with column medians instead of dropping the whole row.
    # avoids artificially inflating S6's variance (regression rows would otherwise be dropped,
    # leaving only classification rows where S6 happens to spread widely).
    sub = score_matrix[RULES].fillna(score_matrix[RULES].median())
    informative = [r for r in RULES if sub[r].std() > 1e-9] # rules with variance; constants carry no CRITIC signal

    if len(informative) < 2: #if there are fewer than 2 informative rules, can't really apply the CRITIC method, so just return the AHP scores and mark everything as "AGREE"
        results = []
        for rule in RULES:
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
    critic_weights = pd.Series(0.0, index=RULES) # start every rule at 0
    critic_weights.loc[informative] = informativeness / total_info #only informative rules get a real weight; degenerates stay at 0

    critic_scores = {rule: int(round(critic_weights[rule] * TOTAL_POINTS)) for rule in RULES}
    point_diff = TOTAL_POINTS - sum(critic_scores.values())
    if point_diff != 0:
        max_rule = informative[0] # rounding leftover lands on the biggest informative rule, never on a degenerate one
        for rule in informative:
            if critic_weights[rule] > critic_weights[max_rule]:
                max_rule = rule
        critic_scores[max_rule] += point_diff

    results = []
    for rule in RULES:
        ahp_points = AHP_WEIGHTS.get(rule, 0)
        crit_points = critic_scores[rule]
        delta = abs(ahp_points - crit_points)
        if rule in informative:
            verdict = "DIVERGE" if delta > DIVERGENCE_THRESHOLD else "AGREE"
            # see Tzeng et al. and other AHP-CRITIC integration variants in the MCDM literature.
            final = round(AHP_PART * ahp_points + (1 - AHP_PART) * crit_points)
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


def results_table(results):
    #per-rule CRITIC results for reporting (one row per soft rule)
    return pd.DataFrame([{
        "rule": r.rule,
        "ahp_pts": r.ahp_score,
        "critic_pts": r.critic_score,
        "final_pts": r.final_score,
        "std": round(r.stdv, 4),
        "informativeness": round(r.informativeness, 4),
        "delta": r.delta,
        "verdict": r.verdict,
    } for r in results])


def correlation_table(score_matrix): #helper that can be run without whole critic pipeline for debugging or reporting
    sub = score_matrix[RULES].fillna(score_matrix[RULES].median())
    informative = [r for r in RULES if sub[r].std() > 1e-9]
    normalized = pd.DataFrame(MinMaxScaler().fit_transform(sub[informative]),
                              columns=informative, index=sub.index)
    return normalized.corr()


def load_weights(path=WEIGHTS_PATH):
    # load the frozen soft-rule weights that calibrate() derived from one big CRITIC run.
    # falls back to the AHP weights if no calibration file exists yet.
    p = Path(path)
    if not p.exists():
        return dict(AHP_WEIGHTS)
    data = json.loads(p.read_text())
    return {r: data[r] for r in RULES if r in data}   # keep only the rule weights, ignore metadata


def compute_critic_scores(score_matrix, weights):
    # composite = weighted average of the soft-rule scores per dataset, using CRITIC weights.
    rules = [r for r in weights if r in score_matrix.columns]
    w = pd.Series({r: float(weights[r]) for r in rules})
    scores = score_matrix[rules]
    present = scores.notna()

    weighted = (scores.fillna(0.0) * w).sum(axis=1)         # Sigma score*weight over present rules
    applicable = present.mul(w, axis=1).sum(axis=1)          # total weight of the present rules
    fraction = weighted / applicable                         # weighted mean in [0, 1]

    result = pd.DataFrame({
        "composite_fraction": fraction,
        "composite_score": fraction * TOTAL_POINTS,
    })
    result["passed"] = result["composite_fraction"] >= PASS_THRESHOLD

    return result.sort_values("composite_score", ascending=False)


def calibrate(score_matrix_path="soft_stats.csv", out_path=WEIGHTS_PATH):
    # one-off: derive the soft-rule weights on soft_stats.csv and freeze them (for debugging and reproducibility)
    score_matrix = pd.read_csv(score_matrix_path, index_col=0)
    results = run_critic(score_matrix)
    weights = {r.rule: r.final_score for r in results}   # rule -> final (AHP+CRITIC) weight
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "derived_at": date.today().isoformat(),
        "n_datasets": int(score_matrix.shape[0]),
        "source": f"CRITIC calibration on {score_matrix_path}",
        **weights,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    results_table(results).to_csv(out_dir / "critic_results.csv", index=False)
    correlation_table(score_matrix).round(4).to_csv(out_dir / "critic_correlations.csv")
    return weights


if __name__ == "__main__":
    # run once after a big configuration run; the pipeline then loads WEIGHTS_PATH each run.
    weights = calibrate()
    print("CRITIC-calibrated soft-rule weights:", weights)
    print(f"written to {WEIGHTS_PATH} (+ critic_results.csv, critic_correlations.csv)")
