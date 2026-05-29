"""
Shared Mamba model definitions.

This module is the single source of truth for all Mamba network structures
used in the project:
- feature-based classifier: MambaClassifier
- sequence-based classifier: MambaSequenceClassifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence models."""

    def __init__(self, d_model, dropout=0.1, max_len=8192):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MambaBlock(nn.Module):
    """Core selective state-space block used by both Mamba classifiers."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(1).repeat(1, self.d_inner)
        a = a / d_state
        self.A_log = nn.Parameter(torch.log(a + 1e-4))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x):
        _, seq_len, _ = x.shape

        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        x = self.activation(x)
        ssm_out = self.ssm(x)
        out = ssm_out * self.activation(res)
        return self.out_proj(out)

    def ssm(self, x):
        a = -torch.exp(self.A_log.float())
        d = self.D.float()

        x_dbl = self.x_proj(x)
        b, c = x_dbl.split([self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(x))

        x_expanded = x.unsqueeze(2)
        b_expanded = b.unsqueeze(3)
        h = b_expanded * x_expanded
        h = h * torch.exp(a.unsqueeze(0).unsqueeze(0))
        c_expanded = c.unsqueeze(3)
        y = (h * c_expanded).sum(dim=2)
        y = y * delta
        y = y + x * d.unsqueeze(0).unsqueeze(0)
        return y


class ResidualBlock(nn.Module):
    """LayerNorm + MambaBlock + dropout with residual connection."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaClassifier(nn.Module):
    """Feature-based Mamba classifier that reshapes flat features into short sequences."""

    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = 8
        self.input_dim = input_dim

        feature_per_step = (input_dim + self.seq_len - 1) // self.seq_len
        self.embedding = nn.Sequential(
            nn.Linear(feature_per_step, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [ResidualBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        batch_size, input_dim = x.shape
        seq_len = self.seq_len

        if input_dim % seq_len != 0:
            pad_size = seq_len - (input_dim % seq_len)
            x = F.pad(x, (0, pad_size), mode="constant", value=0)
            input_dim = x.shape[1]

        x = x.view(batch_size, seq_len, input_dim // seq_len)
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=1)
        return self.classifier(x)


class MambaSequenceClassifier(nn.Module):
    """Sequence-based Mamba classifier for inputs shaped as (batch, seq_len, channels)."""

    def __init__(
        self,
        input_channels,
        num_classes,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [ResidualBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class MambaSequenceEncoder(nn.Module):
    """Encode one sequence modality into a pooled high-level feature vector."""

    def __init__(
        self,
        input_channels,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [ResidualBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(n_layers)]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.output_norm(x)


class DualBranchMambaSequenceClassifier(nn.Module):
    """
    Dual-branch Mamba classifier for paired EMG and IMU sequences.

    Each modality is encoded by an independent Mamba stack. The resulting
    feature vectors are concatenated before the dense classification head.
    """

    def __init__(
        self,
        emg_input_channels,
        imu_input_channels,
        num_classes,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        fusion_hidden_dim=None,
    ):
        super().__init__()
        self.emg_encoder = MambaSequenceEncoder(
            input_channels=emg_input_channels,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.imu_encoder = MambaSequenceEncoder(
            input_channels=imu_input_channels,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        fused_dim = d_model * 2
        hidden_dim = fusion_hidden_dim if fusion_hidden_dim is not None else d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, emg_x, imu_x):
        emg_features = self.emg_encoder(emg_x)
        imu_features = self.imu_encoder(imu_x)
        fused_features = torch.cat([emg_features, imu_features], dim=-1)
        return self.classifier(fused_features)


class TransformerSequenceClassifier(nn.Module):
    """Sequence Transformer classifier for inputs shaped as (batch, seq_len, channels)."""

    def __init__(
        self,
        input_channels,
        num_classes,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class TransformerSequenceEncoder(nn.Module):
    """Encode one sequence modality with a Transformer into a pooled feature vector."""

    def __init__(
        self,
        input_channels,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.output_norm(x)


class DualBranchTransformerSequenceClassifier(nn.Module):
    """
    Dual-branch Transformer classifier for paired EMG and IMU sequences.

    Each modality is encoded by an independent Transformer encoder. The pooled
    features are concatenated before the dense classification head.
    """

    def __init__(
        self,
        emg_input_channels,
        imu_input_channels,
        num_classes,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        fusion_hidden_dim=None,
    ):
        super().__init__()
        self.emg_encoder = TransformerSequenceEncoder(
            input_channels=emg_input_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.imu_encoder = TransformerSequenceEncoder(
            input_channels=imu_input_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        fused_dim = d_model * 2
        hidden_dim = fusion_hidden_dim if fusion_hidden_dim is not None else d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, emg_x, imu_x):
        emg_features = self.emg_encoder(emg_x)
        imu_features = self.imu_encoder(imu_x)
        fused_features = torch.cat([emg_features, imu_features], dim=-1)
        return self.classifier(fused_features)


__all__ = [
    "PositionalEncoding",
    "MambaBlock",
    "ResidualBlock",
    "MambaClassifier",
    "MambaSequenceClassifier",
    "MambaSequenceEncoder",
    "DualBranchMambaSequenceClassifier",
    "TransformerSequenceClassifier",
    "TransformerSequenceEncoder",
    "DualBranchTransformerSequenceClassifier",
]
