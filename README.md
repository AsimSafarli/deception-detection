# Facial Muscle Deception Detection

A research pipeline that studies whether facial muscle activity over time separates deceptive
from truthful speech. It extracts a 3D face mesh and 52 muscle-related signals from every video
frame, trains sequence models with a subject-independent evaluation, and shows the results on an
interactive 3D face.

> **For educational purposes only.** This is a research prototype, not a lie detector, and must
> not be used to judge real people. Contact: asim.safarlii@gmail.com

## Results on Real-life Trial

115 clips (6 had no detectable face), 5-fold `StratifiedGroupKFold` grouped by court case, pooled
out-of-fold scores:

| Model                    | AUC  | Accuracy |
|--------------------------|------|----------|
| Random Forest (baseline) | 0.57 | 52%      |
| Temporal CNN             | 0.47 | 46%      |
| LSTM + attention         | 0.45 | 48%      |

Several muscles differ between deceptive and truthful clips at clip level (AU6, AU20, AU9,
p < 0.001), but no model carries this over to unseen people. The differences mostly reflect
individual faces: one case alone contributes 33 clips. Published results on this dataset that
reach 75–90% often allow the same person in training and test.

## Layout

```
deception-detection/
├── data/
│   ├── raw_videos/            input videos
│   ├── features/              (900, 58) feature sequence per video
│   ├── mesh3d/                (n_frames, 478, 3) face mesh per video
│   └── labels_template.csv
├── src/
│   ├── extract_features.py    MediaPipe (or Py-Feat) feature extraction
│   ├── make_labels_rltrial.py labels.csv from the Real-life Trial README
│   ├── dataset.py             PyTorch dataset with per-fold normalization
│   ├── model.py               BiLSTM + attention, temporal CNN
│   ├── train.py               subject-independent K-fold training
│   ├── baseline.py            logistic regression / random forest baselines
│   ├── build_dashboard_data.py
│   ├── build_face3d.py        mean 3D face, muscle regions, muscle timelines
│   └── server.py              local video analysis server
├── dashboard/
│   ├── index.html             dashboard page (claude.ai artifact and local)
│   ├── local.html             runs index.html locally
│   ├── analyze.html           upload a video and analyze it
│   ├── face3d.js              3D face renderer
│   └── upload/                data files the pages read
└── models/                    checkpoints, normalization stats, out-of-fold predictions
```

## Setup

Python 3.12 is required (MediaPipe has no wheels for 3.14 yet).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models/mediapipe && curl -L -o models/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Pipeline

```bash
# 1. Dataset (GPL, University of Michigan)
curl -L -o data/download/RealLifeDeceptionDetection.2016.zip \
  https://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip
unzip -q data/download/RealLifeDeceptionDetection.2016.zip -x '__MACOSX/*' -d data/download
cp data/download/Real-life_Deception_Detection_2016/Clips/*/*.mp4 data/raw_videos/

# 2. Features: every frame, resampled to 30 FPS, 900 frames = 30 s
.venv/bin/python src/extract_features.py --backend mediapipe --video_dir data/raw_videos \
  --out_dir data/features --mesh_dir data/mesh3d --max_len 900

# 3. Labels, baselines, models
.venv/bin/python src/make_labels_rltrial.py \
  --readme data/download/Real-life_Deception_Detection_2016/README.txt --out data/labels.csv
.venv/bin/python src/baseline.py --features_dir data/features --labels_csv data/labels.csv
.venv/bin/python src/train.py --features_dir data/features --labels_csv data/labels.csv --model lstm
.venv/bin/python src/train.py --features_dir data/features --labels_csv data/labels.csv --model cnn

# 4. Dashboard data
cd src
../.venv/bin/python build_dashboard_data.py --features_dir ../data/features --labels_csv ../data/labels.csv \
  --video_dir ../data/raw_videos --models_dir ../models --out ../dashboard/data.json
../.venv/bin/python build_face3d.py --out_dir ../dashboard/upload
```

## Local apps

```bash
cd src && ../.venv/bin/python server.py
```

- http://localhost:8765/analyze.html: upload a video. It is processed on this machine and deleted
  afterwards. The page plays the video next to a 3D face whose muscle regions follow it, with the
  model probabilities, LSTM attention and the strongest moments.
- http://localhost:8765/local.html: the dataset dashboard (model results, 3D muscle map, clip
  explorer, all clips).

## Method notes

1. **Subject-independent split.** Groups are court cases, so no person appears in both training
   and test. Without this, models learn to recognise faces rather than behaviour.
2. **No leakage in training.** Z-score statistics and early stopping use only the training groups
   of each fold; the test fold is scored once. Statistics are saved per fold so new videos are
   normalized the same way.
3. **Micro-expressions.** They last 1/25 to 1/5 s. No frames are skipped and every video is
   resampled to 30 FPS, so 0.2 s is always 6 frames. 13 clips are recorded at 10 FPS; upsampling
   them adds no information.
4. **Muscles.** MediaPipe blendshapes are mapped to FACS action units and facial muscles
   (for example AU12 zygomaticus major, AU4 corrugator supercilii). A camera estimates muscle
   activity from skin-surface motion; it does not measure muscles.
5. **Attention.** LSTM attention is close to uniform on this data, so the "key moments" are not
   yet reliable evidence.

## Other datasets

MDPE (193 subjects), Bag-of-Lies, MU3D and DOLOS need a signed licence agreement from their
authors. MDPE is the most useful next step: https://github.com/cai-cong/MDPE
