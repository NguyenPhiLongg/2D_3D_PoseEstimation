import torch
import torch.nn as nn


class FallCNNLSTM(nn.Module):
    def __init__(
        self,
        input_dim=40,
        num_classes=2,
        cnn_channels=128,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.3
    ):
        super(FallCNNLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if lstm_layers == 1 else dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, feature_dim)
        x = x.transpose(1, 2)   # (batch, feature_dim, sequence_length)

        x = self.cnn(x)         # (batch, channels, sequence_length)

        x = x.transpose(1, 2)   # (batch, sequence_length, channels)

        lstm_out, _ = self.lstm(x)

        last_out = lstm_out[:, -1, :]

        logits = self.classifier(last_out)

        return logits