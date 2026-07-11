from __future__ import annotations

import torch
import torch.nn as nn


class CNNOnlyClassifier(nn.Module):
    """
    CNN-only baseline.

    Input shape:
        x: (batch, sequence_length, input_dim)

    This follows the paper's comparison idea:
        CNN vs LSTM vs ConvLSTM.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        cnn_channels: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: B, T, F -> B, F, T
        x = x.transpose(1, 2)
        x = self.net(x)
        return self.classifier(x)


class LSTMOnlyClassifier(nn.Module):
    """
    LSTM-only baseline.

    Input shape:
        x: (batch, sequence_length, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last)


class SequentialConvLSTMClassifier(nn.Module):
    """
    Adapted ConvLSTM-style model from Yadav et al.

    The paper describes a sequential fusion:
        CNN -> LSTM -> Fully Connected -> Softmax

    This implementation uses Conv1D layers over temporal skeleton features
    to extract spatial/feature patterns, then feeds the CNN output sequence
    into an LSTM for temporal modeling.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        cnn_channels: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.30,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: B, T, F -> B, F, T
        x = x.transpose(1, 2)

        # CNN feature filtering
        x = self.cnn(x)

        # B, C, T -> B, T, C
        x = x.transpose(1, 2)

        # LSTM temporal modeling
        out, _ = self.lstm(x)
        last = out[:, -1, :]

        return self.classifier(last)


def build_model(
    model_name: str,
    input_dim: int,
    num_classes: int = 2,
    cnn_channels: int = 128,
    lstm_hidden: int = 128,
    lstm_layers: int = 2,
    dropout: float = 0.30,
):
    model_name = model_name.lower().strip()

    if model_name == "cnn":
        return CNNOnlyClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            cnn_channels=cnn_channels,
            dropout=dropout,
        )

    if model_name == "lstm":
        return LSTMOnlyClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True,
        )

    if model_name in ["convlstm", "sequential_convlstm"]:
        return SequentialConvLSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True,
        )

    raise ValueError(f"Unknown model_name: {model_name}")
