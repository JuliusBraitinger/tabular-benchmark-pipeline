#calibration of the final weights via critic weights

import json
from datetime import date
from pathlib import Path
import pandas as pd
from pipeline.config import WEIGHTS_PATH
from pipeline.critic.critic import run_critic, final_weights

def calibrate(score_matrix_path="soft_stats.csv", out_path=WEIGHTS_PATH):
    score_matrix = pd.read_csv(score_matrix_path, index_col=0)
    results = run_critic(score_matrix)
    weights = final_weights(results)
    payload = {
        "_derived_at": date.today().isoformat(),
        "_n_datasets": score_matrix.shape[0],
        "_source": f"CRITIC calibration on {score_matrix_path}",
        **weights,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))

    return weights


if __name__ == "__main__":
    print(calibrate())