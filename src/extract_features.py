"""Extract per-frame facial features from videos and save fixed-length sequences (.npy).

Backends:
    mediapipe  52 blendshapes + 6 head-pose values per frame, every frame, resampled to a common FPS;
               optionally the 478-point 3D face mesh.
    pyfeat     Py-Feat AU intensities and emotions.

Usage:
    python src/extract_features.py --backend mediapipe --video_dir data/raw_videos \
        --out_dir data/features --mesh_dir data/mesh3d --max_len 900
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

POSE_COLUMNS = ["yaw", "pitch", "roll", "tx", "ty", "tz"]


def extract_with_pyfeat(video_path: str, skip_frames: int = 1) -> pd.DataFrame:
    from feat import Detector

    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        device="cpu",
    )
    result = detector.detect_video(video_path, skip_frames=skip_frames)
    merged = pd.concat([result.aus.copy(), result.emotions.copy()], axis=1)
    merged.insert(0, "frame", range(len(merged)))
    return merged


def extract_with_openface_csv(openface_csv_path: str) -> pd.DataFrame:
    """Convert an OpenFace 2.0 CSV (AU*_r intensity columns) to this pipeline's format."""
    df = pd.read_csv(openface_csv_path)
    df.columns = [c.strip() for c in df.columns]
    au_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
    out = df[au_cols].copy()
    out.columns = [c.replace("_r", "") for c in out.columns]
    out.insert(0, "frame", range(len(out)))
    return out


def extract_with_mediapipe(video_path: str, model_path: str, mesh_out: str = None) -> tuple:
    """Run MediaPipe Face Landmarker on every frame.

    Returns (features DataFrame, frame timestamps in seconds). Frames without a face are NaN.
    If mesh_out is given, the (n_frames, 478, 3) mesh is saved there.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, times, meshes = [], [], []
    bs_names = None

    with FaceLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(image, int(round(t * 1000)))

            if res.face_blendshapes:
                cats = res.face_blendshapes[0]
                if bs_names is None:
                    bs_names = [c.category_name for c in cats]
                feats = [c.score for c in cats] + list(_pose_from_matrix(res.facial_transformation_matrixes[0]))
                mesh = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32)
            else:
                feats, mesh = None, np.full((478, 3), np.nan, dtype=np.float32)

            rows.append(feats)
            times.append(t)
            meshes.append(mesh)
            idx += 1
    cap.release()

    if bs_names is None:
        raise RuntimeError("no face found in any frame")

    cols = bs_names + POSE_COLUMNS
    data = np.array([r if r is not None else [np.nan] * len(cols) for r in rows], dtype=np.float32)
    if mesh_out:
        np.save(mesh_out, np.stack(meshes))
    return pd.DataFrame(data, columns=cols), np.array(times)


def _pose_from_matrix(m) -> tuple:
    """Euler angles (degrees) and translation from a 4x4 facial transformation matrix."""
    m = np.asarray(m)
    r = m[:3, :3]
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = np.degrees(np.arcsin(-np.clip(r[2, 0], -1, 1)))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll, m[0, 3], m[1, 3], m[2, 3]


def resample_to_fps(df: pd.DataFrame, times: np.ndarray, target_fps: float = 30.0) -> pd.DataFrame:
    """Resample to a common time grid so 0.2 s (micro-expression scale) is the same frame count in every video.

    Frames without a face are linearly interpolated. Upsampling a low-FPS video adds no information.
    """
    grid = np.arange(0, times[-1] + 1e-9, 1.0 / target_fps)
    out = {}
    for col in df.columns:
        v = df[col].values
        valid = ~np.isnan(v)
        out[col] = np.interp(grid, times[valid], v[valid]) if valid.any() else np.zeros_like(grid)
    return pd.DataFrame(out)


def sequence_from_dataframe(df: pd.DataFrame, max_len: int = 150) -> np.ndarray:
    """Zero-pad or truncate to a (max_len, n_features) array."""
    feature_cols = [c for c in df.columns if c != "frame"]
    arr = df[feature_cols].fillna(0).values.astype(np.float32)
    if len(arr) >= max_len:
        return arr[:max_len]
    return np.vstack([arr, np.zeros((max_len - len(arr), arr.shape[1]), dtype=np.float32)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_len", type=int, default=150, help="sequence length in frames")
    parser.add_argument("--skip_frames", type=int, default=1, help="pyfeat only: process every n-th frame")
    parser.add_argument("--backend", choices=["pyfeat", "openface", "mediapipe"], default="pyfeat")
    parser.add_argument("--mp_model", default="models/mediapipe/face_landmarker.task")
    parser.add_argument("--target_fps", type=float, default=30.0, help="mediapipe: common FPS (>= 25 for micro-expressions)")
    parser.add_argument("--mesh_dir", default=None, help="mediapipe: where to save the 3D face mesh per video")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.mesh_dir:
        os.makedirs(args.mesh_dir, exist_ok=True)

    video_paths = sorted(
        glob.glob(os.path.join(args.video_dir, "*.mp4"))
        + glob.glob(os.path.join(args.video_dir, "*.avi"))
        + glob.glob(os.path.join(args.video_dir, "*.mov"))
    )
    if not video_paths:
        print(f"[!] No videos found in {args.video_dir}.")
        return

    for vp in tqdm(video_paths, desc="Feature extraction"):
        name = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(args.out_dir, f"{name}.npy")
        if os.path.exists(out_path):
            continue
        try:
            if args.backend == "pyfeat":
                df = extract_with_pyfeat(vp, skip_frames=args.skip_frames)
            elif args.backend == "mediapipe":
                mesh_out = os.path.join(args.mesh_dir, f"{name}_mesh.npy") if args.mesh_dir else None
                df, times = extract_with_mediapipe(vp, args.mp_model, mesh_out)
                df = resample_to_fps(df, times, args.target_fps)
            else:
                raise ValueError("OpenFace mode: convert its CSV with extract_with_openface_csv()")
            np.save(out_path, sequence_from_dataframe(df, max_len=args.max_len))
        except Exception as e:
            print(f"[ERROR] {vp}: {e}")


if __name__ == "__main__":
    main()
