

import pandas as pd
rows = []

def record (dataset, soft_results):
    row = {
        "dataset_id": dataset.id,
        "source": dataset.source,
        "name": dataset.name,
        "n_rows": dataset.n_rows,
        "n_features": dataset.n_features,
    }

    for r in soft_results:
        row[r.rule] = r.score
    rows.append(row)
        #TODO add details to row as well

def to_dataframe():
    return pd.DataFrame(rows)


def save_csv(path="soft_stats.csv"):
    df = to_dataframe()
    df.to_csv(path, index=False)
    return df