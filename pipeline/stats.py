# collects rule results for every candidate so we can export a summary csv later
import pandas as pd

_rows = []


def record(dataset_id, source, name, rule_results):
    row = {
        "dataset_id": dataset_id,
        "source": source,
        "name": name,
    }
    # fill in pass/fail for each rule we got a result for
    for r in rule_results:
        row[r.rule] = "pass" if r.passed else "fail: " + r.reason

    # did everything pass?
    row["status"] = "accepted" if all(r.passed for r in rule_results) else "rejected"
    _rows.append(row)


def to_dataframe():
    return pd.DataFrame(_rows)


def save_csv(path="rule_stats.csv"):
    df = to_dataframe()
    df.to_csv(path, index=False)
    return df
