"""Local analysis server for dashboard/analyze.html.

Extracts face/muscle features from an uploaded video with MediaPipe, runs the trained models
(5-fold LSTM and CNN ensembles, Random Forest) and returns JSON. The video is processed on this
machine only and deleted afterwards.

Usage:
    cd src && ../.venv/bin/python server.py          # http://localhost:8765
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, redirect, request
from sklearn.ensemble import RandomForestClassifier

from baseline import clip_stats
from build_dashboard_data import FPS, MUSCLES, WINDOW, muscle_series
from extract_features import extract_with_mediapipe, resample_to_fps, sequence_from_dataframe
from model import LSTMClassifier, TemporalCNN

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASH = os.path.join(ROOT, "dashboard")
MODELS = os.path.join(ROOT, "models")
FEATURES = os.path.join(ROOT, "data", "features")
LABELS = os.path.join(ROOT, "data", "labels.csv")
MP_MODEL = os.path.join(MODELS, "mediapipe", "face_landmarker.task")
MAX_LEN = 900  # same as training: 30 s at 30 FPS
ALLOWED = {".mp4", ".mov", ".avi", ".m4v", ".webm", ".mkv"}
MUSCLE_NAMES = list(MUSCLES)


def load_reference():
    """Fit the Random Forest and collect truthful-clip statistics for muscle z-scores."""
    labels = pd.read_csv(LABELS)
    labels = labels[labels["filename"].apply(lambda f: os.path.exists(os.path.join(FEATURES, f"{f}.npy")))]
    feats = {f: np.load(os.path.join(FEATURES, f"{f}.npy")) for f in labels["filename"]}
    series = {f: muscle_series(a[np.abs(a).sum(1) > 0]) for f, a in feats.items()}
    truth = labels.loc[labels["label"] == 0, "filename"]

    frames = np.concatenate([series[f] for f in truth])
    clip_means = np.stack([series[f].mean(0) for f in truth])
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=42)
    rf.fit(np.stack([clip_stats(feats[f]) for f in labels["filename"]]), labels["label"].values)
    return {
        "rf": rf,
        "frame_mu": frames.mean(0), "frame_sd": frames.std(0) + 1e-6,
        "clip_mu": clip_means.mean(0), "clip_sd": clip_means.std(0) + 1e-6,
    }


def load_folds(kind):
    folds = []
    for k in range(1, 6):
        w, n = os.path.join(MODELS, f"{kind}_fold{k}.pt"), os.path.join(MODELS, f"{kind}_fold{k}_norm.npz")
        if not (os.path.exists(w) and os.path.exists(n)):
            continue
        net = LSTMClassifier(n_features=58) if kind == "lstm" else TemporalCNN(n_features=58)
        net.load_state_dict(torch.load(w, map_location="cpu"))
        net.eval()
        norm = np.load(n)
        folds.append((net, norm["mean"], norm["std"]))
    if not folds:
        raise RuntimeError(f"{kind}: model weights or *_norm.npz missing; rerun train.py")
    return folds


@torch.no_grad()
def predict(folds, seq, mask):
    """Mean probability and mean attention (if any) across the fold models."""
    probs, attns = [], []
    for net, mu, sd in folds:
        x = (seq - mu) / sd
        x[~mask] = 0.0
        logits, attn = net(torch.tensor(x[None], dtype=torch.float32), torch.tensor(mask[None]))
        probs.append(torch.softmax(logits, 1)[0, 1].item())
        if attn is not None:
            attns.append(attn[0].numpy())
    return float(np.mean(probs)), (np.mean(attns, 0) if attns else None)


def analyze(path):
    df, times = extract_with_mediapipe(path, MP_MODEL)
    face_ratio = float((~df.isna().any(axis=1)).mean())
    source_fps = float(len(times) / times[-1]) if len(times) > 1 and times[-1] > 0 else float(FPS)
    full = resample_to_fps(df, times, FPS).values.astype(np.float32)

    seq = sequence_from_dataframe(pd.DataFrame(full), max_len=MAX_LEN)
    mask = np.abs(seq).sum(1) > 0
    p_lstm, attn = predict(LSTM, seq, mask)
    p_cnn, _ = predict(CNN, seq, mask)
    p_rf = float(REF["rf"].predict_proba(clip_stats(seq)[None])[0, 1])

    ms = muscle_series(full)
    n_win = len(ms) // WINDOW
    win = ms[: n_win * WINDOW].reshape(n_win, WINDOW, -1).mean(1)
    z = (win - REF["frame_mu"]) / REF["frame_sd"]
    timeline = [{"t": round(i * WINDOW / FPS, 2), "z": [round(float(v), 2) for v in z[i]]} for i in range(n_win)]
    profile = (ms.mean(0) - REF["clip_mu"]) / REF["clip_sd"]

    # attention covers only the first MAX_LEN frames, which is what the model sees
    attention, moments = [], []
    if attn is not None:
        a = attn[mask]
        n_a = len(a) // WINDOW
        aw = a[: n_a * WINDOW].reshape(n_a, WINDOW).sum(1)
        aw = aw / aw.sum() if aw.sum() > 0 else aw
        attention = [{"t": round(i * WINDOW / FPS, 2), "weight": float(aw[i])} for i in range(n_a)]
        mu_c, sd_c = ms.mean(0), ms.std(0) + 1e-6
        for w in np.argsort(aw)[::-1][:3]:
            seg = (ms[w * WINDOW:(w + 1) * WINDOW].mean(0) - mu_c) / sd_c
            top = np.argsort(seg)[::-1][:3]
            moments.append({"t": round(w * WINDOW / FPS, 2), "weight": float(aw[w]),
                            "muscles": [{"muscle": MUSCLE_NAMES[k], "au": MUSCLES[MUSCLE_NAMES[k]][0],
                                         "z": round(float(seg[k]), 2)} for k in top]})

    return {
        "duration_s": round(len(full) / FPS, 2),
        "model_window_s": round(min(len(full), MAX_LEN) / FPS, 2),
        "source_fps": round(source_fps, 2),
        "face_ratio": round(face_ratio, 3),
        "probs": {"lstm": p_lstm, "cnn": p_cnn, "rf": p_rf},
        "muscles": [{"muscle": m, "au": MUSCLES[m][0]} for m in MUSCLE_NAMES],
        "profile": [round(float(v), 2) for v in profile],
        "timeline": timeline,
        "attention": attention,
        "moments": moments,
        "window_s": WINDOW / FPS,
    }


app = Flask(__name__, static_folder=DASH, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


@app.get("/")
def home():
    return redirect("/local.html#video")


@app.post("/api/analyze")
def api_analyze():
    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify(error="No video file was sent."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Unsupported format ({ext}). Use one of: {', '.join(sorted(ALLOWED))}"), 400
    tmp = tempfile.mkdtemp(prefix="dd_upload_")
    try:
        path = os.path.join(tmp, "video" + ext)
        f.save(path)
        return jsonify(analyze(path))
    except RuntimeError:
        return jsonify(error="No face was found in the video. Use a clip where the face is clearly visible."), 422
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    REF = load_reference()
    LSTM, CNN = load_folds("lstm"), load_folds("cnn")
    print(f"Ready: http://localhost:{args.port}/local.html#video")
    app.run(host="127.0.0.1", port=args.port, debug=False)
