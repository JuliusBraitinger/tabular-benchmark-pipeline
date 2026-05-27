
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder


DATASETS_DIR = Path("data/datasets")
PLOT_DIR = Path("data/sanity_plots")
SUMMARY_CSV = Path("sanity_summary.csv")
TEST_SIZE = 0.2
CV_FOLDS = 5
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 10


def _load_dataset(folder: Path):
    X = pd.read_parquet(folder / "X.parquet")
    y = pd.read_parquet(folder / "y.parquet").squeeze()
    with open(folder / "meta.pkl", "rb") as fh:
        meta = pickle.load(fh)
    task = meta.get("task_type", "classification")
    return X, y, task


def _prep_features(X):
    if hasattr(X, "sparse"):
        X = X.sparse.to_dense()
    X = X.select_dtypes(include="number").fillna(0)
    return X


def _evaluate(folder: Path):
    X, y, task_type = _load_dataset(folder)
    X = _prep_features(X)

    is_classification = "classification" in task_type
    if is_classification:
        y = pd.Series(LabelEncoder().fit_transform(y), index=X.index)
        # drop rows whose class has only 1 sample -- can't be stratified
        counts = y.value_counts()
        rare = counts[counts < 2].index
        if len(rare):
            keep = ~y.isin(rare)
            X = X.loc[keep]
            y = y.loc[keep]
        n_classes = y.nunique()
        stratify = y
        model_cls = RandomForestClassifier
        scoring = "balanced_accuracy"
        test_metric = balanced_accuracy_score
    else:
        n_classes = None
        stratify = None
        model_cls = RandomForestRegressor
        scoring = "r2"
        test_metric = r2_score

    # 80/20 train/test split (stratified for classification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=stratify,
    )

    # 5-fold CV on training set (n_jobs=-1 -> use all CPU cores).
    # For high-dim data, force max_features="sqrt" -- RandomForestRegressor's
    # default of 1.0 (all features per split) is intractable when P >> 1000.
    max_features = "sqrt" if X_train.shape[1] > 1000 else 1.0
    model = model_cls(
        n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        max_features=max_features,
        random_state=42, n_jobs=-1,
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS,
                                scoring=scoring, n_jobs=-1)

    # Final fit + held-out test eval
    model.fit(X_train, y_train)
    test_score = test_metric(y_test, model.predict(X_test))

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if is_classification:
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax1, cmap="Blues")
        ax1.set_title(f"confusion matrix (test, n={len(y_test)})")
    else:
        ax1.scatter(y_test, model.predict(X_test), alpha=0.4, s=8)
        lo, hi = float(np.min(y_test)), float(np.max(y_test))
        ax1.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax1.set_xlabel("y_true")
        ax1.set_ylabel("y_pred")
        ax1.set_title(f"pred vs true (test, n={len(y_test)})")

    top = np.argsort(model.feature_importances_)[-15:]
    ax2.barh(range(len(top)), model.feature_importances_[top])
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(X.columns[top], fontsize=7)
    ax2.set_title("top 15 features (gini)")

    fig.suptitle(f"{str(folder.relative_to(DATASETS_DIR))}  -  CV {scoring}={cv_scores.mean():.3f}  test={test_score:.3f}")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = str(folder.relative_to(DATASETS_DIR)).replace("/", "_").replace(":", "_")
    fig.savefig(PLOT_DIR / f"{safe_name}.png", dpi=150)
    plt.close(fig)

    return {
        "dataset": str(folder.relative_to(DATASETS_DIR)),
        "task": "classification" if is_classification else "regression",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "test_score": round(test_score, 4),
        "metric": scoring,
        "verdict": "PASS" if test_score > (0.55 if is_classification else 0.05) else "WEAK",
        "error": "",
    }


def main():
    # Find every folder that contains an X.parquet, at any nesting depth.
    # Kaggle datasets are saved as data/datasets/kaggle:owner/slug/X.parquet
    # so a shallow iterdir() misses them.
    folders = sorted({p.parent for p in DATASETS_DIR.rglob("X.parquet")})

    # Resume support: load existing rows and skip datasets already in the CSV
    if SUMMARY_CSV.exists():
        existing_df = pd.read_csv(SUMMARY_CSV)
        rows = existing_df.to_dict("records")
        done = set(existing_df["dataset"].astype(str))
        print(f"resuming - found {len(done)} datasets already in {SUMMARY_CSV}")
    else:
        rows = []
        done = set()

    print(f"found {len(folders)} datasets total\n")

    for i, folder in enumerate(folders, 1):
        rel = str(folder.relative_to(DATASETS_DIR))
        if rel in done:
            print(f"[{i}/{len(folders)}] {rel} -- already done, skipping")
            continue

        print(f"[{i}/{len(folders)}] {rel}")
        try:
            row = _evaluate(folder)
        except Exception as e:
            print(f"  ERROR: {e}")
            row = {
                "dataset": rel, "task": "?", "n_samples": "?", "n_features": "?",
                "n_classes": "?", "cv_mean": "?", "cv_std": "?", "test_score": "?",
                "metric": "?", "verdict": "ERROR", "error": str(e)[:200],
            }
        rows.append(row)
        print(
            f"  task={row['task']} N={row['n_samples']} P={row['n_features']} "
            f"cv={row['cv_mean']}+-{row['cv_std']} test={row['test_score']} -> {row['verdict']}"
        )

        # Save after each dataset so a kill doesn't lose all progress
        pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    print(f"\nSaved summary to {SUMMARY_CSV}")
    print(f"Plots in {PLOT_DIR}/")


if __name__ == "__main__":
    main()
