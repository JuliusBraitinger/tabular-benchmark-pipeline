# collects rule results for every candidate so we can export a summary csv later
import pandas as pd
import plotly.graph_objects as go

_rows = []

RULE_ORDER = ["A1", "A2", "A4", "A5", "A3", "A6"]
RULE_NAME = {"A1": "task type", "A2": "synthetic", "A3": "signal",
             "A4": "dimensions", "A5": "licence", "A6": "cross-dup"}




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



def build_sankey(out_html="hard_rules_sankey.html"): #creates a sankey plot of the hard rules 
    df = to_dataframe() #get the dataframe of the rule results in the same run
    rules = [r for r in RULE_ORDER if r in df.columns]
    stage = {r: f"{r} {RULE_NAME[r]}" for r in rules}
    flows = {} #count flow between named nodes
    targets = [stage[r] for r in rules] + ["Accepted"]
    flows = {}
    for _, grp in df.groupby("dataset_id"):
        flows[(grp["source"].iloc[0], targets[0])] = \
            flows.get((grp["source"].iloc[0], targets[0]), 0) + 1
        for i, r in enumerate(rules):
            failed = grp[r].dropna().astype(str).str.startswith("fail").any()
            b = f"Rejected: {r}" if failed else targets[i + 1]
            flows[(targets[i], b)] = flows.get((targets[i], b), 0) + 1
            if failed:
                break
    labels = list(dict.fromkeys(n for pair in flows for n in pair))
    idx = {name: i for i, name in enumerate(labels)}

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=18, thickness=18),
        link=dict(
            source=[idx[a] for a, b in flows],
            target=[idx[b] for a, b in flows],
            value=list(flows.values()),
        ),
    )).update_layout(title_text="Hard-rule funnel", font_size=12)
    fig.write_html(out_html)
    return fig
