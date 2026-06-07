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



def build_sankey(csv_path = "rule_stats.csv"): #creates a sankey plot of the hard rules 
    df = pd.read_csv(csv_path)
    rules = [r for r in RULE_ORDER if r in df.columns]
    stage = {r: f"{r} {RULE_NAME[r]}" for r in rules}
    flows = {} #count flow between named nodes
    targets = [stage[r] for r in rules] + ["Accepted"]
    for _, grp in df.groupby("dataset_id"):
        flows[(grp["source"].iloc[0], targets[0])] = \
            flows.get((grp["source"].iloc[0], targets[0]), 0) + 1
        for i, r in enumerate(rules):
            failed = grp[r].dropna().astype(str).str.startswith("fail").any()
            b = f"Rejected {r}" if failed else targets[i + 1]
            flows[(targets[i], b)] = flows.get((targets[i], b), 0) + 1
            if failed:
                break
    labels = list(dict.fromkeys(n for pair in flows for n in pair))
    idx = {name: i for i, name in enumerate(labels)}

    incoming = {n: 0 for n in labels}
    outgoing = {n: 0 for n in labels}
    for (a, b), v in flows.items():
        outgoing[a] += v
        incoming[b] += v
    node_labels = [f"{n}: {max(incoming[n], outgoing[n])}" for n in labels]

    # explicit column per node so reject branches don't overlap the funnel
    sources = set(df["source"])
    step = 1 / (len(rules) + 1)
    node_x = []
    for n in labels:
        if n in sources:
            x = 0.0
        elif n in ("Accepted", "Rejected"):
            x = 1.0
        else:
            x = (rules.index(n.split()[0]) + 1) * step
        node_x.append(min(max(x, 0.001), 0.999))

    # colors: passing funnel green, rejections red, distinct source/stage nodes
    palette = ["#4C78A8", "#9D755D", "#B279A2", "#F58518", "#EECA3B", "#72B7B2"]  # no green (reserved for Accepted)
    node_colors, ci = [], 0
    for n in labels:
        if n == "Accepted":
            node_colors.append("#2CA02C")
        elif n == "Rejected":
            node_colors.append("#D62728")
        else:
            node_colors.append(palette[ci % len(palette)])
            ci += 1
    link_colors = ["rgba(150,150,150,0.35)"
                   for a, b in flows]

    # push Rejected to the bottom; keep the passing funnel up top
    node_y = [0.92 if n == "Rejected" else 0.30 for n in labels]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=node_labels, x=node_x, y=node_y, pad=30, thickness=22,
                  color=node_colors, line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
        link=dict(
            source=[idx[a] for a, b in flows],
            target=[idx[b] for a, b in flows],
            value=list(flows.values()),
            color=link_colors,
        ),
    )).update_layout(
        title_text="Hard-rule funnel", font_size=13,
        width=1400, height=800, margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.write_html(csv_path.replace(".csv", ".html"))
    return fig

if __name__ == "__main__":
    build_sankey()
