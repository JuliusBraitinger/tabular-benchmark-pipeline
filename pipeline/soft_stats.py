

import pandas as pd
rows = []

def record (dataset, soft_results):
    row = {
        "dataset_id": dataset.id,
        "source": dataset.source,
        "name": dataset.name,
        "url": dataset.metadata.get("url", ""),
        "n_rows": dataset.X.shape[0],
        "n_features": dataset.X.shape[1],
        "task_type": dataset.task_type,
        "n_classes": dataset.y.nunique() if "classification" in dataset.task_type else None,
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