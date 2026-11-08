"""Train and evaluate a deception classifier with subject-independent K-fold cross-validation.

Clips of one subject never appear in both train and test; otherwise the model learns to
recognise faces instead of behaviour. labels.csv therefore needs a subject_id column.

Usage:
    python src/train.py --features_dir data/features --labels_csv data/labels.csv \
        --model lstm --epochs 50 --folds 5
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd

from dataset import DeceptionDataset
from model import LSTMClassifier, TemporalCNN


def train_one_fold(model, train_loader, val_loader, device, epochs, lr, patience=10):
    """Early stopping on an inner validation split of the training groups, never on the test fold."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits, _ = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={total_loss/len(train_loader.dataset):.4f} "
                  f"| val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.3f}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping (epoch {epoch})")
            break

    model.load_state_dict(best_state)
    return model, best_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    for x, y, mask in loader:
        x, mask = x.to(device), mask.to(device)
        logits, _ = model(x, mask)
        total_loss += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
        all_probs.extend(probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # only one class present
    cm = confusion_matrix(all_labels, all_preds)

    return {"accuracy": acc, "f1": f1, "auc": auc, "confusion_matrix": cm,
            "loss": total_loss / len(all_labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--labels_csv", required=True,
                         help="columns: filename,label,subject_id")
    parser.add_argument("--model", choices=["lstm", "cnn"], default="lstm")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DeceptionDataset(args.features_dir, args.labels_csv)
    labels_df = dataset.labels_df
    n_features = dataset.n_features

    if "subject_id" not in labels_df.columns:
        raise ValueError("labels_csv has no 'subject_id' column; it is required for a subject-independent split.")

    groups = labels_df["subject_id"].values
    y_all = labels_df["label"].values

    gkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []
    oof_rows, oof_attn = [], {}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(len(y_all)), y_all, groups), 1):
        print(f"\n=== Fold {fold}/{args.folds} ===")
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=args.seed)
        in_tr, in_val = next(inner.split(train_idx, y_all[train_idx], groups[train_idx]))
        inner_train_idx, inner_val_idx = train_idx[in_tr], train_idx[in_val]
        print(f"  train={len(inner_train_idx)} val={len(inner_val_idx)} test={len(test_idx)} clips")

        dataset.fit_normalization(inner_train_idx)
        # saved so inference on new videos (server.py) uses the same normalization
        np.savez(os.path.join(args.out_dir, f"{args.model}_fold{fold}_norm.npz"), mean=dataset.mean, std=dataset.std)
        train_loader = DataLoader(Subset(dataset, inner_train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, inner_val_idx), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

        if args.model == "lstm":
            model = LSTMClassifier(n_features=n_features).to(device)
        else:
            model = TemporalCNN(n_features=n_features).to(device)

        model, _ = train_one_fold(model, train_loader, val_loader, device, args.epochs, args.lr)
        metrics = evaluate(model, test_loader, device)
        print(f"  Fold {fold} final: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auc={metrics['auc']:.3f}")
        print(f"  Confusion matrix:\n{metrics['confusion_matrix']}")

        fold_results.append(metrics)
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"{args.model}_fold{fold}.pt"))

        # out-of-fold predictions and attention for analysis and the dashboard
        model.eval()
        with torch.no_grad():
            for i in test_idx:
                x, y, mask = dataset[i]
                logits, attn = model(x[None].to(device), mask[None].to(device))
                fname = labels_df.iloc[i]["filename"]
                oof_rows.append({"filename": fname, "label": int(y), "fold": fold,
                                 "prob_deceptive": torch.softmax(logits, 1)[0, 1].item()})
                if attn is not None:
                    oof_attn[fname] = attn[0].cpu().numpy()

    print("\n=== Overall (mean ± std across folds) ===")
    for key in ["accuracy", "f1", "auc"]:
        vals = [r[key] for r in fold_results]
        print(f"{key}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    pd.DataFrame(oof_rows).to_csv(os.path.join(args.out_dir, f"{args.model}_oof.csv"), index=False)
    if oof_attn:
        np.savez(os.path.join(args.out_dir, f"{args.model}_attention.npz"), **oof_attn)


if __name__ == "__main__":
    main()
