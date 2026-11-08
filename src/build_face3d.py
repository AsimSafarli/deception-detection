"""Build the data behind the 3D face-muscle view.

1. Mean 3D face: frontal frames from data/mesh3d, centred, scaled and averaged.
2. Triangles: 3-cliques of the MediaPipe face tesselation edges.
3. Muscle regions: anatomical anchor landmarks per muscle; each vertex gets a Gaussian weight by distance.
4. Clip timeline: activity of the 22 muscle groups per 0.2 s window, as z-scores against truthful frames.

Usage:
    cd src && ../.venv/bin/python build_face3d.py --out_dir ../dashboard/upload
    ../.venv/bin/python build_face3d.py --preview face_check.png   # plot anchors to check placement
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from build_dashboard_data import MUSCLES, WINDOW, FPS, muscle_series

# muscle -> landmarks marking the centre of its anatomical region (both sides)
ANCHORS = {
    "Frontalis (medial)": [151, 108, 337],
    "Frontalis (lateral)": [104, 333, 67, 297],
    "Corrugator supercilii": [55, 285, 107, 336],
    "Levator palpebrae superioris": [159, 386],
    "Orbicularis oculi (orbital)": [117, 346, 111, 340, 35, 265],
    "Orbicularis oculi (palpebral)": [145, 374],
    "Orbicularis oculi (blink)": [159, 145, 386, 374],
    "Levator labii sup. alaeque nasi": [129, 358, 198, 420],
    "Levator labii superioris": [206, 426, 39, 269],
    "Zygomaticus major": [207, 427, 187, 411],
    "Buccinator": [57, 287, 212, 432],
    "Depressor anguli oris": [204, 424],
    "Depressor labii inferioris": [84, 314, 181, 405],
    "Mentalis": [152, 175, 199],
    "Incisivus labii": [0, 17],
    "Risorius / platysma": [214, 434, 135, 364],
    "Orbicularis oris (funnel)": [0, 17, 37, 267, 84, 314],
    "Orbicularis oris (press)": [13, 14, 82, 312, 87, 317],
    "Orbicularis oris (roll)": [12, 15, 72, 302],
    "Masseter / digastric": [172, 397, 136, 365],
    "Lateral pterygoid": [132, 361, 58, 288],
    "Extraocular (gaze)": [468, 473],
}
assert list(ANCHORS) == list(MUSCLES)


def triangles_from_edges(edges):
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    tris = set()
    for a, b in edges:
        for c in adj[a] & adj[b]:
            tris.add(tuple(sorted((a, b, c))))
    return sorted(tris)


def mean_face(mesh_dir, video_aspect, max_frames_per_clip=20):
    faces = []
    for path in sorted(glob.glob(os.path.join(mesh_dir, "*_mesh.npy"))):
        name = os.path.basename(path).replace("_mesh.npy", "")
        m = np.load(path)
        m = m[~np.isnan(m).any(axis=(1, 2))]
        if not len(m):
            continue
        p = m.copy()
        p[:, :, 1] /= video_aspect.get(name, 16 / 9)  # x and z are in width units, y in height units
        # frontal: both cheeks (234/454) at equal depth and the nose tip (1) centred between them
        width = np.linalg.norm(p[:, 234, :2] - p[:, 454, :2], axis=1)
        yaw = np.abs(p[:, 234, 2] - p[:, 454, 2]) / width
        mid = np.abs(p[:, 1, 0] - (p[:, 234, 0] + p[:, 454, 0]) / 2) / width
        good = np.argsort(yaw + mid)[:max_frames_per_clip]
        good = good[(yaw[good] < 0.08) & (mid[good] < 0.05)]
        for f in p[good]:
            f = f - f.mean(0)
            f = f / np.linalg.norm(f[234, :2] - f[454, :2])
            faces.append(f)
    face = np.mean(faces, axis=0)
    face[:, 1] *= -1  # image y points down
    face[:, 2] *= -1  # MediaPipe z is negative towards the camera
    return face, len(faces)


def muscle_weights(face, sigma=0.075):
    w = {}
    for m, idx in ANCHORS.items():
        d = np.min(np.linalg.norm(face[:, None, :] - face[idx][None, :, :], axis=2), axis=1)
        w[m] = np.exp(-0.5 * (d / sigma) ** 2)
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", default="../data/mesh3d")
    parser.add_argument("--features_dir", default="../data/features")
    parser.add_argument("--labels_csv", default="../data/labels.csv")
    parser.add_argument("--video_dir", default="../data/raw_videos")
    parser.add_argument("--out_dir", default="../dashboard/upload")
    parser.add_argument("--preview", default=None, help="write a PNG of the anchors instead of the data files")
    args = parser.parse_args()

    import cv2
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    aspect = {}
    for v in glob.glob(os.path.join(args.video_dir, "*.mp4")):
        cap = cv2.VideoCapture(v)
        aspect[os.path.splitext(os.path.basename(v))[0]] = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()

    face, n = mean_face(args.mesh_dir, aspect)
    tris = triangles_from_edges([(c.start, c.end) for c in C.FACE_LANDMARKS_TESSELATION])
    weights = muscle_weights(face)
    print(f"mean face from {n} frontal frames, {len(tris)} triangles")

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(16, 9))
        for ax, (i, j, title) in zip(axes, [(0, 1, "front"), (2, 1, "side")]):
            ax.scatter(face[:, i], face[:, j], s=3, c="#bbb")
            for k, (m, idx) in enumerate(ANCHORS.items()):
                col = plt.cm.tab20(k % 20)
                ax.scatter(face[idx, i], face[idx, j], s=30, color=col)
                if i == 0:
                    ax.annotate(m.split(" (")[0][:14] + (" " + m.split("(")[1][:4] if "(" in m else ""),
                                face[idx[0], [i, j]], fontsize=7, color=col)
            ax.set_aspect("equal")
            ax.set_title(title)
        plt.savefig(args.preview, dpi=110, bbox_inches="tight")
        print("preview ->", args.preview)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    names = list(MUSCLES)
    verts = []
    for v in range(len(face)):
        ws = {m: round(float(weights[m][v]), 2) for m in names if weights[m][v] >= 0.12}
        verts.append({"v": v, "x": round(float(face[v, 0]), 4), "y": round(float(face[v, 1]), 4),
                      "z": round(float(face[v, 2]), 4), "w": ws})
    tri_rows = [{"a": a, "b": b, "c": c} for a, b, c in tris]
    anchor_rows = []
    for m, idx in ANCHORS.items():
        pts = face[idx]
        one_side = pts[pts[:, 0] <= 0.02]  # label bilateral muscles on one side, not the midline
        c = (one_side if len(one_side) else pts).mean(0)
        anchor_rows.append({"muscle": m, "au": MUSCLES[m][0], "x": round(float(c[0]), 4),
                            "y": round(float(c[1]), 4), "z": round(float(c[2]), 4)})

    labels = pd.read_csv(args.labels_csv)
    series = {}
    for f in labels["filename"]:
        p = os.path.join(args.features_dir, f"{f}.npy")
        if os.path.exists(p):
            a = np.load(p)
            series[f] = muscle_series(a[np.abs(a).sum(1) > 0])
    truth = np.concatenate([series[f] for f in labels.loc[labels["label"] == 0, "filename"] if f in series])
    mu, sd = truth.mean(0), truth.std(0) + 1e-6
    timeline = []
    for f, s in series.items():
        n_win = len(s) // WINDOW
        win = s[: n_win * WINDOW].reshape(n_win, WINDOW, -1).mean(1)
        z = (win - mu) / sd
        for i in range(n_win):
            timeline.append({"id": f, "t": round(i * WINDOW / FPS, 1), "z": [round(float(v), 1) for v in z[i]]})

    for name, rows in [("face_vertices.json", verts), ("face_triangles.json", tri_rows),
                       ("face_anchors.json", anchor_rows), ("clip_muscle_timeline.json", timeline)]:
        with open(os.path.join(args.out_dir, name), "w") as fh:
            json.dump(rows, fh, separators=(",", ":"))
        print(name, len(rows), f"{os.path.getsize(os.path.join(args.out_dir, name)) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
