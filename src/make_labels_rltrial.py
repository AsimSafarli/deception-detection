"""Build labels.csv (filename, label, subject_id) from the Real-life Trial README tables.

The dataset has no official subject ids, so the court case in the "Role / trial name" column is
used as the group. Everyone in one case lands in the same fold, which is stricter than a
per-person split and rules out subject leakage.

Usage:
    python src/make_labels_rltrial.py \
        --readme data/download/Real-life_Deception_Detection_2016/README.txt --out data/labels.csv
"""

import argparse
import re
import pandas as pd

ROW_RE = re.compile(r"^\|\s*(trial_(lie|truth)_\d{3})\.mp4\s*\|\s*([^|]+?)\s*\|")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", required=True)
    parser.add_argument("--out", default="data/labels.csv")
    args = parser.parse_args()

    rows = []
    with open(args.readme, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ROW_RE.match(line)
            if not m:
                continue
            name, kind, role_trial = m.groups()
            trial = role_trial.split("/", 1)[-1].strip()
            subject = re.sub(r"[^a-z0-9]+", "_", trial.lower()).strip("_")
            rows.append({"filename": name, "label": int(kind == "lie"), "subject_id": subject})

    df = pd.DataFrame(rows)
    # The README lists trial_truth_008 twice where 009 is meant.
    for i in df.index[df["filename"].duplicated()]:
        num = int(df.at[i, "filename"][-3:]) + 1
        df.at[i, "filename"] = df.at[i, "filename"][:-3] + f"{num:03d}"
    assert not df["filename"].duplicated().any()

    df.to_csv(args.out, index=False)
    print(f"{len(df)} clips, {df['label'].sum()} deceptive, {df['subject_id'].nunique()} groups -> {args.out}")
    print(df.groupby("subject_id")["label"].agg(["count", "sum"]).sort_values("count", ascending=False).to_string())


if __name__ == "__main__":
    main()
