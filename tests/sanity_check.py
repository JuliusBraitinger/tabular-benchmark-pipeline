# TODO: proper train/test split - hold out 20% before CV, only run CV on train, final eval on held-out test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

folders = sorted([f for f in Path("data/datasets").iterdir() if (f / "X.parquet").exists()])
print("found", len(folders), "datasets\n")

for folder in folders:
    X = pd.read_parquet(folder / "X.parquet")
    y = pd.read_parquet(folder / "y.parquet").squeeze()

    if hasattr(X, "sparse"):
        X = X.sparse.to_dense()
    X = X.select_dtypes(include="number").fillna(0)
    y = LabelEncoder().fit_transform(y)

    print(folder.name + ":", X.shape[0], "samples,", X.shape[1], "features,", len(set(y)), "classes")

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5, scoring="balanced_accuracy")
    print("  accuracy:", round(scores.mean(), 3), "+/-", round(scores.std(), 3))

    rf.fit(X, y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ConfusionMatrixDisplay.from_estimator(rf, X, y, ax=ax1, cmap="Blues")
    ax1.set_title("confusion matrix (not meaningfull - no train/test split)")

    top = np.argsort(rf.feature_importances_)[-15:]
    ax2.barh(range(15), rf.feature_importances_[top])
    ax2.set_yticks(range(15))
    ax2.set_yticklabels(X.columns[top], fontsize=7)
    ax2.set_title("top 15 features via gini index")

    fig.suptitle(folder.name + "  —  acc=" + str(round(scores.mean(), 3)))
    fig.tight_layout()
    fig.savefig(f"{folder.name}_sanity.png", dpi=150)
    plt.close()
    print("  saved", folder.name + "_sanity.png\n")
