"""Classical baselines on per-clip summary statistics, using the same folds as train.py.

Usage:
    python src/baseline.py --features_dir data/features --labels_csv data/labels.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def clip_stats(arr: np.ndarray) -> np.ndarray:
    """Mean, std, max and 95th percentile of every feature over the clip's real (non-padding) frames."""
    arr = arr[np.abs(arr).sum(axis=1) > 0]
    return np.concatenate([arr.mean(0), arr.std(0), arr.max(0), np.percentile(arr, 95, axis=0)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.labels_csv)
    df = df[df["filename"].apply(lambda f: os.path.exists(os.path.join(args.features_dir, f"{f}.npy")))]
    X = np.stack([clip_stats(np.load(os.path.join(args.features_dir, f"{f}.npy"))) for f in df["filename"]])
    y, groups = df["label"].values, df["subject_id"].values

    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=args.seed),
    }
    gkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for name, make in models.items():
        res = {"accuracy": [], "f1": [], "auc": []}
        for tr, te in gkf.split(X, y, groups):
            prob = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            res["accuracy"].append(accuracy_score(y[te], pred))
            res["f1"].append(f1_score(y[te], pred, zero_division=0))
            res["auc"].append(roc_auc_score(y[te], prob) if len(set(y[te])) > 1 else np.nan)
        print(f"[{name}] " + " | ".join(f"{k}: {np.nanmean(v):.3f} ± {np.nanstd(v):.3f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
