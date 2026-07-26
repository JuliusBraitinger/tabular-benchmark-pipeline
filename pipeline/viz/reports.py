# Per-dataset evaluation report (Phase 5). Trains the shared RF on each accepted
# dataset with out-of-fold CV and writes a self-contained HTML per dataset plus
# an aggregate metrics.csv. clf: acc/balanced/precision/recall/f1/auc; reg:
# r2/rmse/mae. Characterisation only, not a gate.
# reports are created by claude code

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, label_binarize
from pipeline.config import TOTAL_POINTS

log = logging.getLogger(__name__)
CV_FOLDS = 3


def make_reference_model(task_type, max_features=None):
    # A3's gate model: a deliberately shallow forest to detect whether signal
    # exists at all. Kept weak on purpose. "sqrt" for wide matrices (P>1000).
    kw = dict(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    if max_features:
        kw["max_features"] = max_features
    cls = RandomForestClassifier if "classification" in task_type else RandomForestRegressor
    return cls(**kw)


def make_report_model(task_type, max_features=None):
    # the report's model: stronger than the gate so metrics reflect *achievable*
    # quality, not just baseline signal.
    kw = dict(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    if max_features:
        kw["max_features"] = max_features
    cls = RandomForestClassifier if "classification" in task_type else RandomForestRegressor
    return cls(**kw)


def make_cv(task_type):
    # shuffle the folds, else datasets sorted by group get rejected for no reason
    if "classification" in task_type:
        return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)


def evaluate_dataset(ds):
    clf = "classification" in ds.task_type
    y = ds.y[ds.y.notna()]
    X = ds.X.loc[y.index].select_dtypes("number").rename(columns=str).fillna(0.0)
    if clf:
        y = pd.Series(LabelEncoder().fit_transform(y), index=X.index)
        keep = y.isin(y.value_counts().loc[lambda c: c >= CV_FOLDS].index)  # drop classes too rare for a fold
        X, y = X.loc[keep], y.loc[keep]
        y = pd.Series(LabelEncoder().fit_transform(y), index=X.index)  # re-encode so labels stay 0..K-1 contiguous
    if len(X) < CV_FOLDS or (clf and y.nunique() < 2):
        raise ValueError(f"too little data to evaluate ({len(X)} rows)")
    if len(X) > 10_000:  # cap huge datasets so CV stays quick
        idx = X.sample(10_000, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]

    model = make_report_model(ds.task_type, "sqrt" if X.shape[1] > 1000 else None)
    yt = y.to_numpy()
    yp = cross_val_predict(model, X, yt, cv=make_cv(ds.task_type), n_jobs=-1)

    if clf:
        labels = sorted(set(yt.tolist()))
        proba = cross_val_predict(model, X, yt, cv=make_cv(ds.task_type), n_jobs=-1, method="predict_proba")
        m = {"accuracy": skm.accuracy_score(yt, yp),
             "balanced_accuracy": skm.balanced_accuracy_score(yt, yp),
             "precision": skm.precision_score(yt, yp, average="macro", zero_division=0),
             "recall": skm.recall_score(yt, yp, average="macro", zero_division=0),
             "f1": skm.f1_score(yt, yp, average="macro", zero_division=0),
             "mcc": skm.matthews_corrcoef(yt, yp),
             "roc_auc": auc_score(yt, proba, labels),
             "pr_auc": pr_auc_score(yt, proba, labels)}
    else:
        labels, proba = None, None
        m = {"r2": skm.r2_score(yt, yp),
             "rmse": skm.mean_squared_error(yt, yp) ** 0.5,
             "mae": skm.mean_absolute_error(yt, yp),
             "median_ae": skm.median_absolute_error(yt, yp),
             "mape": skm.mean_absolute_percentage_error(yt, yp),
             "max_error": skm.max_error(yt, yp)}

    return dict(id=ds.id, source=ds.source, name=ds.name, task=ds.task_type,
                url=ds.metadata.get("url", ""),
                n=len(X), p=X.shape[1], metrics=m, yt=yt, yp=yp, proba=proba, labels=labels)


def auc_score(yt, proba, labels):
    try:
        if len(labels) == 2:
            return skm.roc_auc_score(yt, proba[:, 1])
        return skm.roc_auc_score(yt, proba, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        return None  # e.g. a class missing from a fold


def pr_auc_score(yt, proba, labels):
    # average precision = area under the precision-recall curve (imbalance-friendly)
    try:
        if len(labels) == 2:
            return skm.average_precision_score(yt, proba[:, 1])
        return skm.average_precision_score(label_binarize(yt, classes=labels), proba, average="macro")
    except ValueError:
        return None


def point_density(x, y, bins=60):
    # per-point 2D-histogram density -> colour dense scatters so structure shows, not a blob
    x, y = np.asarray(x, float), np.asarray(y, float)
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.searchsorted(xe, x) - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, y) - 1, 0, bins - 1)
    return h[xi, yi]


def insights(ev):
    m = ev["metrics"]
    if "classification" in ev["task"]:
        ba = m["balanced_accuracy"]
        out = ["Balanced accuracy %.2f - %s" % (ba, "highly learnable" if ba > 0.9 else "weak/hard" if ba < 0.3 else "moderately learnable")]
        if m["roc_auc"] is not None:
            out.append("ROC-AUC %.2f class separability" % m["roc_auc"])
        return out
    r2 = m["r2"]
    return ["R^2 %.2f - %s" % (r2, "strong signal" if r2 > 0.75 else "weak/hard target" if r2 < 0.1 else "moderate signal")]


def figures(ev):
    import plotly.graph_objects as go
    from scipy import stats

    if "classification" in ev["task"]:
        labels, yt, proba = ev["labels"], ev["yt"], ev["proba"]
        cm = skm.confusion_matrix(yt, ev["yp"], labels=labels)
        figs = [go.Figure(go.Heatmap(z=cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                text=cm, texttemplate="%{text}", colorscale="Blues")).update_layout(
                title="Confusion matrix", xaxis_title="predicted", yaxis_title="true", yaxis_autorange="reversed")]

        if proba is not None:
            roc, pr = go.Figure(), go.Figure()
            if len(labels) == 2:
                fpr, tpr, _ = skm.roc_curve(yt, proba[:, 1], pos_label=labels[1])
                roc.add_trace(go.Scatter(x=fpr, y=tpr, name="AUC=%.3f" % skm.auc(fpr, tpr)))
                prec, rec, _ = skm.precision_recall_curve(yt, proba[:, 1], pos_label=labels[1])
                pr.add_trace(go.Scatter(x=rec, y=prec, name="AP=%.3f" % skm.average_precision_score(yt, proba[:, 1])))
            else:
                for i, l in enumerate(labels):
                    fpr, tpr, _ = skm.roc_curve((yt == l).astype(int), proba[:, i])
                    roc.add_trace(go.Scatter(x=fpr, y=tpr, name="%s AUC=%.3f" % (l, skm.auc(fpr, tpr))))
                    prec, rec, _ = skm.precision_recall_curve((yt == l).astype(int), proba[:, i])
                    pr.add_trace(go.Scatter(x=rec, y=prec, name=str(l)))
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="chance"))
            figs.append(roc.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR"))
            figs.append(pr.update_layout(title="Precision-Recall curve", xaxis_title="recall", yaxis_title="precision"))

        prec, rec, f1, _ = skm.precision_recall_fscore_support(yt, ev["yp"], labels=labels, zero_division=0)
        figs.append(go.Figure(go.Heatmap(
            z=[prec, rec, f1], x=[str(l) for l in labels], y=["precision", "recall", "f1"],
            text=[["%.2f" % v for v in row] for row in (prec, rec, f1)], texttemplate="%{text}",
            colorscale="Blues", zmin=0, zmax=1)).update_layout(title="Per-class metrics"))
        return figs

    # regression: predicted-vs-actual (density) + residuals-vs-fitted (density) + Q-Q of residuals
    yt, yp = np.asarray(ev["yt"], float), np.asarray(ev["yp"], float)
    res = yp - yt
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())

    pva = go.Figure([
        go.Scattergl(x=yt, y=yp, mode="markers", marker=dict(size=4, color=point_density(yt, yp), colorscale="Viridis")),
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="black")),
    ]).update_layout(title="Predicted vs actual (colour = density)", xaxis_title="actual", yaxis_title="predicted", showlegend=False)

    rvf = go.Figure([
        go.Scattergl(x=yp, y=res, mode="markers", marker=dict(size=4, color=point_density(yp, res), colorscale="Viridis")),
        go.Scatter(x=[yp.min(), yp.max()], y=[0, 0], mode="lines", line=dict(dash="dash", color="black")),
    ]).update_layout(title="Residuals vs fitted", xaxis_title="predicted", yaxis_title="residual", showlegend=False)

    (osm, osr), (slope, inter, _) = stats.probplot(res, dist="norm")
    qq = go.Figure([
        go.Scattergl(x=osm, y=osr, mode="markers", marker=dict(size=4)),
        go.Scatter(x=osm, y=slope * osm + inter, mode="lines", line=dict(color="red")),
    ]).update_layout(title="Q-Q plot of residuals", xaxis_title="theoretical quantiles",
                     yaxis_title="ordered residuals", showlegend=False)
    return [pva, rvf, qq]


REPORT_CSS = """
*{box-sizing:border-box}
:root{--bg:#f4f6f8;--surface:#fff;--ink:#111827;--muted:#6b7280;--line:#e6e8ec;
  --accent:#3b6fb0;--track:#eef1f4;--ok:#15803d;--ok-bg:#e7f4ec;--bad:#b91c1c;--bad-bg:#fdeaea}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.5;
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 64px}
.eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:12px;color:var(--muted);font-weight:600}
h1{font-size:26px;margin:4px 0 6px;font-weight:700;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.meta{color:var(--muted);font-size:14px}.meta a{color:var(--accent);text-decoration:none}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:0 0 14px;font-weight:600}
.badge{font-size:12px;font-weight:700;padding:3px 10px;border-radius:999px;letter-spacing:.03em}
.badge.ok{background:var(--ok-bg);color:var(--ok)}.badge.bad{background:var(--bad-bg);color:var(--bad)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:24px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.t-val{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.t-lab{font-size:13px;color:var(--muted);margin-top:2px}.t-sub{font-size:12px;color:var(--muted);margin-top:6px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin-bottom:24px}
.tips{margin:0;padding-left:18px}.tips li{margin:4px 0}
.meters{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px 28px}
.m-top{display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px}
.m-val{color:var(--muted);font-variant-numeric:tabular-nums}
.m-track{height:8px;background:var(--track);border-radius:999px;overflow:hidden}
.m-fill{height:100%;background:var(--accent);border-radius:999px}
.m-code{color:var(--muted);font-weight:600;font-size:11px}
.m-desc{font-size:12px;color:var(--muted);margin-top:5px}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:8px;overflow:hidden}
details.panel summary{cursor:pointer;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
table.metrics{border-collapse:collapse;width:100%;margin-top:14px;font-size:14px}
table.metrics td{padding:6px 12px;border-bottom:1px solid var(--line)}
table.metrics td:first-child{color:var(--muted)}
table.metrics td:last-child{text-align:right;font-variant-numeric:tabular-nums}
"""


def stat_tile(label, value, sub=""):
    sub = "<div class='t-sub'>%s</div>" % sub if sub else ""
    return "<div class='tile'><div class='t-val'>%s</div><div class='t-lab'>%s</div>%s</div>" % (value, label, sub)


# short name + one-line "what it measures" per soft rule (higher score = better on all)
SOFT_RULES = {
    "S1": ("Uniqueness", "distinct from the other datasets"),
    "S2": ("IID", "no duplicate rows"),
    "S3": ("Data quality", "few missing / constant / outlier values"),
    "S4": ("Leakage", "no train/test target leakage"),
    "S5": ("Batch effects", "target not confounded by batch"),
    "S6": ("Class balance", "classes not too imbalanced"),
}


def score_meter(key, value):
    name, desc = SOFT_RULES.get(key, (key, ""))
    pct = max(0.0, min(1.0, value)) * 100
    return ("<div class='meter'>"
            "<div class='m-top'><span><b>%s</b> <span class='m-code'>%s</span></span>"
            "<span class='m-val'>%.2f</span></div>"
            "<div class='m-track'><div class='m-fill' style='width:%.0f%%'></div></div>"
            "<div class='m-desc'>%s</div></div>" % (name, key, value, pct, desc))


def build_html(ev, soft=None):
    # plotly figures fill their card (default_width=100% + responsive) instead of a fixed 700px
    figs = "".join("<div class='card'>%s</div>" % f.to_html(
        full_html=False, include_plotlyjs=(i == 0), default_width="100%", config={"responsive": True})
        for i, f in enumerate(figures(ev)))

    m = ev["metrics"]
    head = ("balanced accuracy", "%.3f" % m["balanced_accuracy"]) if "classification" in ev["task"] \
        else ("R²", "%.3f" % m["r2"])
    tiles = (stat_tile("rows (N)", "{:,}".format(ev["n"]))
             + stat_tile("features (P)", "{:,}".format(ev["p"]))
             + stat_tile(head[0], head[1], "%d-fold CV" % CV_FOLDS))

    # the dataset's final soft-rule score (composite of S1..S6 with the calibrated weights) + pass/fail
    badge, soft_html = "", ""
    if soft is not None:
        passed = str(soft.get("passed")).lower() == "true"
        badge = "<span class='badge %s'>%s</span>" % (("ok", "PASS") if passed else ("bad", "FAIL"))
        frac = soft["composite_fraction"]  # share of the 60 points
        tier = ("best" if frac >= 0.8 else "good" if frac >= 0.7 else "mediocre"
                if frac >= 0.6 else "poor" if frac >= 0.5 else "really bad")
        tiles += stat_tile("soft score", "%.0f/%d" % (soft["composite_score"], TOTAL_POINTS),
                           "%.0f%% · %s" % (frac * 100, tier))
        meters = "".join(score_meter(k, soft[k]) for k in soft
                         if k[:1] == "S" and k[1:].isdigit() and pd.notna(soft[k]))
        soft_html = "<section class='panel'><h2>Soft-rule scores</h2><div class='meters'>%s</div></section>" % meters

    tips = "".join("<li>%s</li>" % t for t in insights(ev))
    metric_rows = "".join("<tr><td>%s</td><td>%s</td></tr>" % (k, "n/a" if v is None else format(v, ".4f"))
                          for k, v in m.items())
    idlink = "<a href='%s'>%s</a>" % (ev["url"], ev["id"]) if ev.get("url") else ev["id"]

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{ev['id']}</title>
<style>{REPORT_CSS}</style></head><body><div class="wrap">
<header><div class="eyebrow">{ev['source']} &middot; {ev['task']}</div>
<h1>{ev['name']} {badge}</h1><div class="meta">{idlink}</div></header>
<section class="tiles">{tiles}</section>
<section class="panel"><h2>Insights</h2><ul class="tips">{tips}</ul></section>
{soft_html}
<section class="figs">{figs}</section>
<details class="panel"><summary>All metrics ({CV_FOLDS}-fold CV)</summary>
<table class="metrics">{metric_rows}</table></details>
</div></body></html>"""


def generate_reports(ds_dirs, load_dataset, results_dir="."):
    # all HTMLs go into one reports/ dir, each named after its dataset id
    reports_dir = Path(results_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # per-dataset soft scores (phase 4) + final composite score (phase 6), if those ran
    soft_path = Path(results_dir) / "soft_stats.csv"
    score_path = Path(results_dir) / "critic_scores.csv"
    soft_df = pd.read_csv(soft_path, index_col=0) if soft_path.exists() else pd.DataFrame()
    score_df = pd.read_csv(score_path, index_col=0) if score_path.exists() else pd.DataFrame()
    rows = []
    for i, d in enumerate(ds_dirs, 1):
        ds = load_dataset(d)
        log.info("[%d/%d] report %s", i, len(ds_dirs), ds.id)
        try:
            ev = evaluate_dataset(ds)
            soft = None
            if ev["id"] in score_df.index:
                soft = dict(score_df.loc[ev["id"]])
                if ev["id"] in soft_df.index:
                    soft.update(soft_df.loc[ev["id"]].to_dict())
            name = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(ev["id"]))  # filesystem-safe id
            (reports_dir / f"{name}.html").write_text(build_html(ev, soft), encoding="utf-8")
        except Exception:
            log.exception("  report failed for %s -- skipping (run continues)", ds.id)
            continue
        rows.append({"id": ev["id"], "source": ev["source"], "url": ev["url"], "task": ev["task"],
                     "n": ev["n"], "p": ev["p"], **ev["metrics"]})
    csv = str(Path(results_dir) / "metrics.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv
