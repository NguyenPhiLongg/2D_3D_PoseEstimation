import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Phase 4 - Quality-Aware 2D-3D Fusion Model.

Main idea:
    2D sequence      -> 2D temporal encoder -> h2d
    3D sequence      -> 3D temporal encoder -> h3d
    quality features -> quality encoder     -> hq

    gate = sigmoid(MLP([h2d, h3d, hq]))

    fused = gate * h2d + (1 - gate) * h3d

    logits = classifier(fused)

Why this is different from Phase 3:
    Phase 3 Gated Fusion only used encoded 2D and 3D features.
    Phase 4 adds quality features such as:
        - keypoint confidence
        - missing joint ratio
        - temporal jitter
        - bone length stability
        - 3D z instability

    Therefore, the gate can learn not only from pose features,
    but also from pose reliability indicators.
"""


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalAttentionPooling(nn.Module):
    """
    Attention pooling over time.

    Input:
        x: [B, T, D]

    Output:
        pooled: [B, D]
        attention_weights: [B, T]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)

        return pooled, weights


class TemporalPoseEncoder(nn.Module):
    """
    CNN1D + BiLSTM encoder for pose sequences.

    Input:
        x: [B, T, input_dim]

    Output:
        h: [B, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: int = 128,
        lstm_hidden: int = 128,
        output_dim: int = 128,
        num_lstm_layers: int = 1,
        dropout: float = 0.3,
        pooling: str = "attention",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim
        self.pooling = pooling

        self.input_norm = nn.LayerNorm(input_dim)

        self.conv = nn.Sequential(
            Conv1DBlock(
                in_channels=input_dim,
                out_channels=cnn_channels,
                kernel_size=3,
                dropout=dropout,
            ),
            Conv1DBlock(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=3,
                dropout=dropout,
            ),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden * 2

        if pooling == "attention":
            self.attention_pool = TemporalAttentionPooling(lstm_output_dim)
            pooled_dim = lstm_output_dim
        elif pooling == "mean":
            self.attention_pool = None
            pooled_dim = lstm_output_dim
        elif pooling == "last":
            self.attention_pool = None
            pooled_dim = lstm_output_dim
        elif pooling == "mean_max":
            self.attention_pool = None
            pooled_dim = lstm_output_dim * 2
        else:
            raise ValueError(f"Invalid pooling mode: {pooling}")

        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        x = self.input_norm(x)

        # Conv1D expects [B, D, T]
        x = x.transpose(1, 2)
        x = self.conv(x)

        # Back to [B, T, C]
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)

        attention_weights = None

        if self.pooling == "attention":
            pooled, attention_weights = self.attention_pool(lstm_out)

        elif self.pooling == "mean":
            pooled = torch.mean(lstm_out, dim=1)

        elif self.pooling == "last":
            pooled = lstm_out[:, -1, :]

        elif self.pooling == "mean_max":
            mean_pool = torch.mean(lstm_out, dim=1)
            max_pool = torch.max(lstm_out, dim=1).values
            pooled = torch.cat([mean_pool, max_pool], dim=1)

        else:
            raise ValueError(f"Invalid pooling mode: {self.pooling}")

        h = self.proj(pooled)

        return h, {
            "attention_weights": attention_weights,
        }


class QualityEncoder(nn.Module):
    """
    Encoder for sequence-level quality features.

    Input:
        q: [B, quality_dim]

    Output:
        hq: [B, output_dim]
    """

    def __init__(
        self,
        quality_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.quality_dim = quality_dim

        self.net = nn.Sequential(
            nn.LayerNorm(quality_dim),
            nn.Linear(quality_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q)


class QualityAwareGatedFusion(nn.Module):
    """
    Quality-aware adaptive fusion model.

    Input:
        x2d:     [B, T, 40]
        x3d:     [B, T, 59]
        quality: [B, Q]

    Output:
        {
            "logits": [B, num_classes],
            "gate": [B, fusion_dim] or [B, 1],
            ...
        }
    """

    def __init__(
        self,
        input_dim_2d: int = 40,
        input_dim_3d: int = 59,
        quality_dim: int = 33,
        num_classes: int = 2,
        encoder_dim: int = 128,
        cnn_channels: int = 128,
        lstm_hidden: int = 128,
        quality_hidden: int = 128,
        fusion_dim: int = 128,
        dropout: float = 0.3,
        gate_type: str = "vector",
        pooling: str = "attention",
    ):
        super().__init__()

        if gate_type not in ["vector", "scalar"]:
            raise ValueError("gate_type must be 'vector' or 'scalar'")

        self.input_dim_2d = input_dim_2d
        self.input_dim_3d = input_dim_3d
        self.quality_dim = quality_dim
        self.num_classes = num_classes
        self.encoder_dim = encoder_dim
        self.fusion_dim = fusion_dim
        self.gate_type = gate_type

        self.encoder_2d = TemporalPoseEncoder(
            input_dim=input_dim_2d,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            output_dim=encoder_dim,
            dropout=dropout,
            pooling=pooling,
        )

        self.encoder_3d = TemporalPoseEncoder(
            input_dim=input_dim_3d,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            output_dim=encoder_dim,
            dropout=dropout,
            pooling=pooling,
        )

        self.quality_encoder = QualityEncoder(
            quality_dim=quality_dim,
            hidden_dim=quality_hidden,
            output_dim=encoder_dim,
            dropout=dropout,
        )

        self.project_2d = nn.Sequential(
            nn.Linear(encoder_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

        self.project_3d = nn.Sequential(
            nn.Linear(encoder_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

        self.project_quality = nn.Sequential(
            nn.Linear(encoder_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

        gate_input_dim = fusion_dim * 3

        if gate_type == "vector":
            gate_output_dim = fusion_dim
        else:
            gate_output_dim = 1

        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fusion_dim // 2, gate_output_dim),
            nn.Sigmoid(),
        )

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(
        self,
        x2d: torch.Tensor,
        x3d: torch.Tensor,
        quality: torch.Tensor,
    ) -> dict:
        h2d_raw, aux2d = self.encoder_2d(x2d)
        h3d_raw, aux3d = self.encoder_3d(x3d)
        hq_raw = self.quality_encoder(quality)

        h2d = self.project_2d(h2d_raw)
        h3d = self.project_3d(h3d_raw)
        hq = self.project_quality(hq_raw)

        gate_input = torch.cat([h2d, h3d, hq], dim=1)
        gate = self.gate_net(gate_input)

        if self.gate_type == "scalar":
            fused = gate * h2d + (1.0 - gate) * h3d
        else:
            fused = gate * h2d + (1.0 - gate) * h3d

        fused = self.fusion_norm(fused)

        # Keep quality representation in the classifier input.
        # This allows the final classifier to use both fused pose representation
        # and explicit reliability information.
        classifier_input = torch.cat([fused, hq], dim=1)

        logits = self.classifier(classifier_input)

        return {
            "logits": logits,
            "gate": gate,
            "mean_gate": gate.mean(dim=1),
            "h2d": h2d,
            "h3d": h3d,
            "hq": hq,
            "attention_2d": aux2d.get("attention_weights"),
            "attention_3d": aux3d.get("attention_weights"),
        }


class ConcatWithQualityFusion(nn.Module):
    """
    Optional ablation model:
        concat h2d + h3d + quality embedding, no gate.

    This is useful later if you want to check whether the gain comes from:
        - quality features only
        - or quality-aware gate
    """

    def __init__(
        self,
        input_dim_2d: int = 40,
        input_dim_3d: int = 59,
        quality_dim: int = 33,
        num_classes: int = 2,
        encoder_dim: int = 128,
        cnn_channels: int = 128,
        lstm_hidden: int = 128,
        quality_hidden: int = 128,
        fusion_dim: int = 128,
        dropout: float = 0.3,
        pooling: str = "attention",
    ):
        super().__init__()

        self.encoder_2d = TemporalPoseEncoder(
            input_dim=input_dim_2d,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            output_dim=encoder_dim,
            dropout=dropout,
            pooling=pooling,
        )

        self.encoder_3d = TemporalPoseEncoder(
            input_dim=input_dim_3d,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            output_dim=encoder_dim,
            dropout=dropout,
            pooling=pooling,
        )

        self.quality_encoder = QualityEncoder(
            quality_dim=quality_dim,
            hidden_dim=quality_hidden,
            output_dim=encoder_dim,
            dropout=dropout,
        )

        self.project = nn.Sequential(
            nn.Linear(encoder_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(
        self,
        x2d: torch.Tensor,
        x3d: torch.Tensor,
        quality: torch.Tensor,
    ) -> dict:
        h2d, aux2d = self.encoder_2d(x2d)
        h3d, aux3d = self.encoder_3d(x3d)
        hq = self.quality_encoder(quality)

        fused = torch.cat([h2d, h3d, hq], dim=1)
        fused = self.project(fused)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "gate": torch.full(
                size=(x2d.shape[0], 1),
                fill_value=0.5,
                device=x2d.device,
                dtype=x2d.dtype,
            ),
            "mean_gate": torch.full(
                size=(x2d.shape[0],),
                fill_value=0.5,
                device=x2d.device,
                dtype=x2d.dtype,
            ),
            "attention_2d": aux2d.get("attention_weights"),
            "attention_3d": aux3d.get("attention_weights"),
        }


# Alias for easier import in training script.
QualityAwareFusionModel = QualityAwareGatedFusion


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def smoke_test() -> None:
    print("Running smoke test for model_quality_aware_fusion.py")

    batch_size = 4
    sequence_length = 60
    input_dim_2d = 40
    input_dim_3d = 59
    quality_dim = 33
    num_classes = 2

    x2d = torch.randn(batch_size, sequence_length, input_dim_2d)
    x3d = torch.randn(batch_size, sequence_length, input_dim_3d)
    quality = torch.randn(batch_size, quality_dim)

    model = QualityAwareGatedFusion(
        input_dim_2d=input_dim_2d,
        input_dim_3d=input_dim_3d,
        quality_dim=quality_dim,
        num_classes=num_classes,
        gate_type="vector",
    )

    output = model(x2d, x3d, quality)

    print(f"Logits shape: {output['logits'].shape}")
    print(f"Gate shape  : {output['gate'].shape}")
    print(f"Mean gate   : {output['gate'].mean().item():.4f}")
    print(f"Parameters  : {count_parameters(model):,}")

    assert output["logits"].shape == (batch_size, num_classes)
    assert output["gate"].shape == (batch_size, model.fusion_dim)

    print("Smoke test passed.")


if __name__ == "__main__":
    smoke_test()