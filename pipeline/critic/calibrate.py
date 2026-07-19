#calibration of the final weights via critic weights

import json
from datetime import date
from pathlib import Path
import pandas as pd
from pipeline.config import WEIGHTS_PATH
from pipeline.critic.critic import run_critic, final_weights, results_table, correlation_table

def calibrate(score_matrix_path="soft_stats.csv", out_path=WEIGHTS_PATH):
    score_matrix = pd.read_csv(score_matrix_path, index_col=0)
    results = run_critic(score_matrix)
    weights = final_weights(results)
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1. the final weights (what the scoring step consumes)
    payload = {
        "derived_at": date.today().isoformat(),
        "n_datasets": int(score_matrix.shape[0]),
        "source": f"CRITIC calibration on {score_matrix_path}",
        **weights,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    # 2. the per-rule CRITIC table + correlation matrix (the reporting deliverables)
    results_table(results).to_csv(out_dir / "critic_results.csv", index=False)
    correlation_table(score_matrix).round(4).to_csv(out_dir / "critic_correlations.csv")
    return weights


if __name__ == "__main__":
    # run once after a big configuration run; the pipeline then loads WEIGHTS_PATH each run.
    weights = calibrate()
    print("CRITIC-calibrated soft-rule weights:", weights)
    print(f"written to {WEIGHTS_PATH} (+ critic_results.csv, critic_correlations.csv)")