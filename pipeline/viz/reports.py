# Per-dataset evaluation report (Phase 5). Trains the shared RF on each accepted
# dataset with out-of-fold CV and writes a self-contained HTML per dataset plus
# an aggregate metrics.csv. clf: acc/balanced/precision/recall/f1/auc; reg:
# r2/rmse/mae. Characterisation only, not a gate.

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


def build_html(ev, soft=None):
    figs = [f.to_html(full_html=False, include_plotlyjs=(i == 0)) for i, f in enumerate(figures(ev))]
    rows = "".join("<tr><td>%s</td><td>%s</td></tr>" % (k, "n/a" if v is None else format(v, ".4f"))
                   for k, v in ev["metrics"].items())
    tips = "".join("<li>%s</li>" % t for t in insights(ev))
    idlink = f"<a href='{ev['url']}'>{ev['id']}</a>" if ev.get("url") else ev["id"]

    # the dataset's final soft-rule score (composite of S1..S6 with the calibrated weights) + pass/fail
    soft_html = ""
    if soft is not None:
        verdict = "PASS" if str(soft.get("passed")).lower() == "true" else "FAIL"
        frac = soft["composite_fraction"]  # rate the dataset by its share of the 60 points
        tier = ("best" if frac >= 0.8 else "good" if frac >= 0.7 else "mediocre"
                if frac >= 0.6 else "poor" if frac >= 0.5 else "really bad")
        breakdown = "".join("<tr><td>%s</td><td>%.3f</td></tr>" % (k, soft[k])
                            for k in soft if k[:1] == "S" and k[1:].isdigit() and pd.notna(soft[k]))
        soft_html = (f"<h2>Soft-rule score</h2>"
                     f"<p><b>{soft['composite_score']:.1f} / {TOTAL_POINTS}</b> "
                     f"({frac:.0%}) &mdash; {verdict} &mdash; <b>{tier}</b></p>"
                     f"<table><tr><th>rule</th><th>score</th></tr>{breakdown}</table>")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{ev['id']}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:4px 10px}}</style></head><body>
<h1>{ev['name']}</h1><p>{idlink} - {ev['source']} - {ev['task']} - N={ev['n']} x P={ev['p']}</p>
{soft_html}
<h2>Insights</h2><ul>{tips}</ul>
<h2>Metrics ({CV_FOLDS}-fold CV)</h2><table><tr><th>metric</th><th>value</th></tr>{rows}</table>
<h2>Plots</h2>{''.join(figs)}</body></html>"""


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
