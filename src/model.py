"""Sequence classifiers for deceptive/truthful prediction from facial feature time series."""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Additive attention that learns a weight per time step."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM with attention pooling; returns logits and per-step attention."""

    def __init__(self, n_features: int, hidden_dim: int = 128, n_layers: int = 2,
                 n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.pool = AttentionPooling(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
        pooled, attn_weights = self.pool(out, mask=mask)
        return self.classifier(pooled), attn_weights


class TemporalCNN(nn.Module):
    """1D CNN over time: faster and smaller than the LSTM."""

    def __init__(self, n_features: int, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, mask=None):
        feat = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(feat), None
