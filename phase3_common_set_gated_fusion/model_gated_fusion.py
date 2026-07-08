"""
Gated Fusion model for Phase 3.

Purpose:
- Learn separate temporal representations for 2D and estimated 3D skeleton features.
- Learn an adaptive gate to control the contribution of each stream.
- Compare against concat fusion on the same common-set split.

Input:
    x2d: (batch_size, sequence_length, 40)
    x3d: (batch_size, sequence_length, 59)

Output:
    logits: (batch_size, num_classes)

Gated fusion formula:
    h2d = Encoder2D(x2d)
    h3d = Encoder3D(x3d)

    gate = sigmoid(MLP([h2d, h3d]))

    fused = gate * h2d + (1 - gate) * h3d

    logits = Classifier(fused)
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class TemporalPoseEncoder(nn.Module):
    """
    CNN1D + BiLSTM encoder for skeleton sequence features.

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output shape:
        (batch_size, lstm_hidden * 2)

    Why CNN1D + BiLSTM:
    - CNN1D extracts local temporal patterns from short frame windows.
    - BiLSTM captures longer temporal dependencies in both directions.
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=64,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.output_dim = lstm_hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor with shape (B, T, D)

        Returns:
            sequence representation with shape (B, 2 * lstm_hidden)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}")

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {x.shape[-1]}"
            )

        # CNN1D expects (B, D, T)
        x = x.transpose(1, 2)

        x = self.cnn(x)

        # LSTM expects (B, T, C)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)

        # Use the last time step representation.
        h = lstm_out[:, -1, :]

        return h


class GatedFusionCNNLSTM(nn.Module):
    """
    Gated Fusion model for 2D and estimated 3D skeleton sequences.

    Instead of directly concatenating 2D and 3D frame features, this model:
    1. Encodes 2D features with a 2D temporal encoder.
    2. Encodes 3D features with a 3D temporal encoder.
    3. Learns a gate value to balance the two representations.
    4. Classifies the fused representation.

    Gate interpretation:
        gate close to 1.0 -> model relies more on 2D stream.
        gate close to 0.0 -> model relies more on 3D stream.
        gate around 0.5    -> model balances both streams.
    """

    def __init__(
        self,
        input_dim_2d: int = 40,
        input_dim_3d: int = 59,
        num_classes: int = 2,
        cnn_channels: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        gate_hidden: int = 128,
        use_vector_gate: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim_2d = input_dim_2d
        self.input_dim_3d = input_dim_3d
        self.num_classes = num_classes
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.gate_hidden = gate_hidden
        self.use_vector_gate = use_vector_gate

        self.encoder_2d = TemporalPoseEncoder(
            input_dim=input_dim_2d,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )

        self.encoder_3d = TemporalPoseEncoder(
            input_dim=input_dim_3d,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )

        self.representation_dim = lstm_hidden * 2

        if use_vector_gate:
            gate_output_dim = self.representation_dim
        else:
            gate_output_dim = 1

        self.gate_network = nn.Sequential(
            nn.Linear(self.representation_dim * 2, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, gate_output_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.representation_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x2d: torch.Tensor,
        x3d: torch.Tensor,
        return_gate: bool = False,
    ):
        """
        Forward pass.

        Args:
            x2d:
                Tensor with shape (B, T, 40)

            x3d:
                Tensor with shape (B, T, 59)

            return_gate:
                If True, return both logits and gate values.

        Returns:
            logits or (logits, gate)
        """
        h2d = self.encoder_2d(x2d)
        h3d = self.encoder_3d(x3d)

        gate_input = torch.cat([h2d, h3d], dim=1)
        gate = self.gate_network(gate_input)

        if gate.shape[1] == 1:
            gate_for_fusion = gate.expand_as(h2d)
        else:
            gate_for_fusion = gate

        fused = gate_for_fusion * h2d + (1.0 - gate_for_fusion) * h3d

        logits = self.classifier(fused)

        if return_gate:
            return logits, gate

        return logits

    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Return total and trainable parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return total_params, trainable_params


def build_gated_fusion_model(
    task: str = "binary",
    cnn_channels: int = 128,
    lstm_hidden: int = 128,
    lstm_layers: int = 1,
    dropout: float = 0.3,
    gate_hidden: int = 128,
    use_vector_gate: bool = True,
) -> GatedFusionCNNLSTM:
    """
    Helper function to build gated fusion model by task.

    task:
        binary -> 2 classes
        action -> 4 classes
    """
    task = task.lower().strip()

    if task == "binary":
        num_classes = 2
    elif task == "action":
        num_classes = 4
    else:
        raise ValueError(f"Invalid task: {task}. Expected 'binary' or 'action'.")

    model = GatedFusionCNNLSTM(
        input_dim_2d=40,
        input_dim_3d=59,
        num_classes=num_classes,
        cnn_channels=cnn_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
        gate_hidden=gate_hidden,
        use_vector_gate=use_vector_gate,
    )

    return model


def smoke_test() -> None:
    """
    Simple test to verify input/output shapes.
    Run:
        python phase3_common_set_gated_fusion/model_gated_fusion.py
    """
    batch_size = 4
    sequence_length = 60

    x2d = torch.randn(batch_size, sequence_length, 40)
    x3d = torch.randn(batch_size, sequence_length, 59)

    model = build_gated_fusion_model(task="binary")

    logits, gate = model(x2d, x3d, return_gate=True)

    print("=" * 80)
    print("GatedFusionCNNLSTM smoke test")
    print("=" * 80)
    print("x2d shape:", tuple(x2d.shape))
    print("x3d shape:", tuple(x3d.shape))
    print("logits shape:", tuple(logits.shape))
    print("gate shape:", tuple(gate.shape))

    total_params, trainable_params = model.get_num_parameters()

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print("=" * 80)

    assert logits.shape == (batch_size, 2)
    assert gate.shape[0] == batch_size

    print("Smoke test passed.")


if __name__ == "__main__":
    smoke_test()