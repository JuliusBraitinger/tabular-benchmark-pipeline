"""CRITIC weight derivation and AHP comparison."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.config import AHP_WEIGHTS, DIVERGENCE_THRESHOLD, TOTAL_POINTS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRITICResult:
    """Result of CRITIC analysis for one criterion."""

    rule: str
    sigma: float  # contrast intensity
    conflict: float  # sum of (1 - r_jk)
    information: float  # C_j = sigma * conflict
    critic_weight: float  # w_j = C_j / sum(C_k)
    critic_points: int  # round(w_j * total_points)
    ahp_points: int
    delta: int  # |ahp - critic|
    verdict: str  # "AGREE" or "DIVERGE"


def run_critic(score_matrix: pd.DataFrame) -> list[CRITICResult]:
    """Compute CRITIC weights from a score matrix and compare with AHP.

    Args:
        score_matrix: DataFrame with columns S1, S2, S3, S4, S6
                      and rows = datasets, values = continuous scores [0.0, 1.0].

    Returns:
        List of CRITICResult, one per criterion.
    """
    rules = [col for col in score_matrix.columns if col.startswith("S")]

    # Step 1: Min-max normalize each column to [0, 1]
    normalized = score_matrix[rules].copy()
    for col in rules:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.0  # no variance

    # Step 2: Contrast intensity (std dev per column)
    sigmas = normalized.std()

    # Step 3: Pearson correlation matrix
    corr_matrix = normalized.corr()

    # Step 4: Conflict per criterion
    conflicts = pd.Series(dtype=float)
    for rule in rules:
        conflicts[rule] = sum(1 - corr_matrix.loc[rule, other] for other in rules)

    # Step 5: Information content
    information = sigmas * conflicts

    # Step 6: CRITIC weights
    total_info = information.sum()
    if total_info == 0:
        logger.warning("Total information is 0 — all criteria have no variance")
        critic_weights = pd.Series({r: 1.0 / len(rules) for r in rules})
    else:
        critic_weights = information / total_info

    # Step 7: Map to points
    critic_points = {rule: int(round(critic_weights[rule] * TOTAL_POINTS)) for rule in rules}

    # Ensure points sum to TOTAL_POINTS (rounding adjustment)
    point_diff = TOTAL_POINTS - sum(critic_points.values())
    if point_diff != 0:
        # Add/subtract from the criterion with largest weight
        max_rule = max(rules, key=lambda r: critic_weights[r])
        critic_points[max_rule] += point_diff

    # Step 8: Compare with AHP
    results: list[CRITICResult] = []
    for rule in rules:
        ahp_pts = AHP_WEIGHTS.get(rule, 0)
        crit_pts = critic_points[rule]
        delta = abs(ahp_pts - crit_pts)
        verdict = "DIVERGE" if delta > DIVERGENCE_THRESHOLD else "AGREE"

        results.append(CRITICResult(
            rule=rule,
            sigma=float(sigmas[rule]),
            conflict=float(conflicts[rule]),
            information=float(information[rule]),
            critic_weight=float(critic_weights[rule]),
            critic_points=crit_pts,
            ahp_points=ahp_pts,
            delta=delta,
            verdict=verdict,
        ))

    return results


def results_to_dataframe(results: list[CRITICResult]) -> pd.DataFrame:
    """Convert CRITIC results to a comparison table."""
    rows = []
    for r in results:
        rows.append({
            "Rule": r.rule,
            "AHP_weight": r.ahp_points / TOTAL_POINTS,
            "AHP_pts": r.ahp_points,
            "CRITIC_weight": r.critic_weight,
            "CRITIC_pts": r.critic_points,
            "sigma": r.sigma,
            "conflict": r.conflict,
            "delta": r.delta,
            "Verdict": r.verdict,
        })
    return pd.DataFrame(rows).set_index("Rule")


def final_weights(results: list[CRITICResult]) -> dict[str, int]:
    """Produce final point allocation: keep AHP where they agree, adopt CRITIC where they diverge."""
    final: dict[str, int] = {}
    for r in results:
        if r.verdict == "AGREE":
            final[r.rule] = r.ahp_points
        else:
            final[r.rule] = r.critic_points
            logger.info(
                "%s: DIVERGE — adopting CRITIC weight (%d pts, was %d pts AHP)",
                r.rule, r.critic_points, r.ahp_points,
            )
    return final
