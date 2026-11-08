"""Collect training results and facial features into dashboard/data.json.

- Maps MediaPipe blendshapes to FACS action units and facial muscles (left/right merged).
- Per clip: out-of-fold model probabilities, muscle profile (z-score vs truthful clips),
  LSTM attention in 0.2 s windows and the muscles most raised at the top-attention moments.
- Overall: deceptive vs truthful muscle differences (Mann-Whitney U, Cliff's delta).
- Court cases are replaced by anonymous codes (case_01, ...).

Usage:
    cd src && ../.venv/bin/python build_dashboard_data.py --features_dir ../data/features \
        --labels_csv ../data/labels.csv --video_dir ../data/raw_videos --models_dir ../models --out ../dashboard/data.json
"""

import argparse
import json
import os
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from baseline import clip_stats

FPS = 30
WINDOW = 6  # 6 frames at 30 FPS = 0.2 s, the upper bound of a micro-expression

# muscle -> (action unit, MediaPipe blendshapes)
MUSCLES = {
    "Frontalis (medial)": ("AU1", ["browInnerUp"]),
    "Frontalis (lateral)": ("AU2", ["browOuterUpLeft", "browOuterUpRight"]),
    "Corrugator supercilii": ("AU4", ["browDownLeft", "browDownRight"]),
    "Levator palpebrae superioris": ("AU5", ["eyeWideLeft", "eyeWideRight"]),
    "Orbicularis oculi (orbital)": ("AU6", ["cheekSquintLeft", "cheekSquintRight"]),
    "Orbicularis oculi (palpebral)": ("AU7", ["eyeSquintLeft", "eyeSquintRight"]),
    "Orbicularis oculi (blink)": ("AU45", ["eyeBlinkLeft", "eyeBlinkRight"]),
    "Levator labii sup. alaeque nasi": ("AU9", ["noseSneerLeft", "noseSneerRight"]),
    "Levator labii superioris": ("AU10", ["mouthUpperUpLeft", "mouthUpperUpRight", "mouthShrugUpper"]),
    "Zygomaticus major": ("AU12", ["mouthSmileLeft", "mouthSmileRight"]),
    "Buccinator": ("AU14", ["mouthDimpleLeft", "mouthDimpleRight"]),
    "Depressor anguli oris": ("AU15", ["mouthFrownLeft", "mouthFrownRight"]),
    "Depressor labii inferioris": ("AU16", ["mouthLowerDownLeft", "mouthLowerDownRight"]),
    "Mentalis": ("AU17", ["mouthShrugLower"]),
    "Incisivus labii": ("AU18", ["mouthPucker"]),
    "Risorius / platysma": ("AU20", ["mouthStretchLeft", "mouthStretchRight"]),
    "Orbicularis oris (funnel)": ("AU22", ["mouthFunnel"]),
    "Orbicularis oris (press)": ("AU24", ["mouthPressLeft", "mouthPressRight", "mouthClose"]),
    "Orbicularis oris (roll)": ("AU28", ["mouthRollLower", "mouthRollUpper"]),
    "Masseter / digastric": ("AU26", ["jawOpen"]),
    "Lateral pterygoid": ("AU29-30", ["jawForward", "jawLeft", "jawRight"]),
    "Extraocular (gaze)": ("AU61-64", ["eyeLookDownLeft", "eyeLookDownRight", "eyeLookUpLeft",
                                       "eyeLookUpRight", "eyeLookInLeft", "eyeLookInRight",
                                       "eyeLookOutLeft", "eyeLookOutRight"]),
}

# Order of the first 52 feature columns written by extract_features.py
BLENDSHAPES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]


def muscle_series(arr: np.ndarray) -> np.ndarray:
    """(T, 58) features -> (T, n_muscles) muscle activity."""
    idx = {n: i for i, n in enumerate(BLENDSHAPES)}
    return np.stack([arr[:, [idx[b] for b in bs]].mean(1) for _, bs in MUSCLES.values()], axis=1)


def cliffs_delta(a, b):
    return float((np.sign(a[:, None] - b[None, :])).mean())


def video_fps(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    num, den = out.strip().split("/")
    return float(num) / float(den)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels_csv", default="data/labels.csv")
    parser.add_argument("--video_dir", default="data/raw_videos")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--out", default="dashboard/data.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels_csv)
    case_codes = {s: f"case_{i:02d}" for i, s in enumerate(labels["subject_id"].unique(), 1)}
    labels["case"] = labels["subject_id"].map(case_codes)
    have = labels["filename"].apply(lambda f: os.path.exists(os.path.join(args.features_dir, f"{f}.npy")))
    missing = labels.loc[~have, "filename"].tolist()
    labels = labels[have].reset_index(drop=True)

    feats = {f: np.load(os.path.join(args.features_dir, f"{f}.npy")) for f in labels["filename"]}
    valid = {f: np.abs(a).sum(1) > 0 for f, a in feats.items()}
    musc = {f: muscle_series(a[valid[f]]) for f, a in feats.items()}
    muscle_names = list(MUSCLES)

    # Random Forest out-of-fold predictions on the same split as train.py
    X = np.stack([clip_stats(feats[f]) for f in labels["filename"]])
    y, groups = labels["label"].values, labels["subject_id"].values
    rf_prob = np.zeros(len(y))
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for tr, te in gkf.split(X, y, groups):
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=args.seed).fit(X[tr], y[tr])
        rf_prob[te] = rf.predict_proba(X[te])[:, 1]

    oof = {m: pd.read_csv(os.path.join(args.models_dir, f"{m}_oof.csv")).set_index("filename")
           for m in ["lstm", "cnn"]}
    attn = np.load(os.path.join(args.models_dir, "lstm_attention.npz"))

    clip_means = np.stack([musc[f].mean(0) for f in labels["filename"]])
    truth_mu = clip_means[y == 0].mean(0)
    truth_sd = clip_means[y == 0].std(0) + 1e-6
    global_rows = []
    for j, m in enumerate(muscle_names):
        a, b = clip_means[y == 1, j], clip_means[y == 0, j]
        global_rows.append({
            "muscle": m, "au": MUSCLES[m][0],
            "mean_deceptive": float(a.mean()), "mean_truthful": float(b.mean()),
            "cliffs_delta": cliffs_delta(a, b), "p_value": float(mannwhitneyu(a, b).pvalue),
        })

    clips = []
    for i, row in labels.iterrows():
        f = row["filename"]
        ms = musc[f]
        z = (ms.mean(0) - truth_mu) / truth_sd
        a = attn[f][valid[f]] if f in attn else np.zeros(len(ms))
        n_win = len(a) // WINDOW
        win = a[: n_win * WINDOW].reshape(n_win, WINDOW).sum(1)
        win = win / win.sum() if win.sum() > 0 else win
        clip_mu, clip_sd = ms.mean(0), ms.std(0) + 1e-6
        moments = []
        for w in np.argsort(win)[::-1][:3]:
            mz = (ms[w * WINDOW:(w + 1) * WINDOW].mean(0) - clip_mu) / clip_sd
            top = np.argsort(mz)[::-1][:3]
            moments.append({"t": round(w * WINDOW / FPS, 1), "weight": float(win[w]),
                            "muscles": [{"muscle": muscle_names[k], "au": MUSCLES[muscle_names[k]][0],
                                         "z": round(float(mz[k]), 2)} for k in top]})
        clips.append({
            "id": f, "label": int(row["label"]), "case": row["case"],
            "fold": int(oof["lstm"].loc[f, "fold"]),
            "duration_s": round(len(ms) / FPS, 1),
            "source_fps": round(video_fps(os.path.join(args.video_dir, f"{f}.mp4")), 2),
            "prob": {"lstm": float(oof["lstm"].loc[f, "prob_deceptive"]),
                     "cnn": float(oof["cnn"].loc[f, "prob_deceptive"]),
                     "rf": float(rf_prob[i])},
            "muscle_z": [round(float(v), 2) for v in z],
            "attention": [round(float(v), 4) for v in win],
            "moments": moments,
        })

    def summarize(name, prob):
        pred = (prob >= 0.5).astype(int)
        return {"model": name, "accuracy": float((pred == y).mean()), "auc": float(roc_auc_score(y, prob))}

    order = labels["filename"].tolist()
    pooled = [summarize("LSTM + attention", np.array([oof["lstm"].loc[f, "prob_deceptive"] for f in order])),
              summarize("Temporal CNN", np.array([oof["cnn"].loc[f, "prob_deceptive"] for f in order])),
              summarize("Random Forest (baseline)", rf_prob)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"muscles": [{"muscle": m, "au": MUSCLES[m][0]} for m in muscle_names],
                   "global": global_rows, "models": pooled, "clips": clips,
                   "no_face": missing, "window_s": WINDOW / FPS, "fps": FPS}, fh)
    print(f"{len(clips)} clips -> {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")
    for p in pooled:
        print(p)


if __name__ == "__main__":
    main()
