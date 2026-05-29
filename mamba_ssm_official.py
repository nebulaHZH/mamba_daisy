"""
Official mamba-ssm encoder adapters.

This module intentionally wraps the installed `mamba-ssm` package so project
training scripts can switch away from the local pure-PyTorch reference
implementation in `mamba_ssm_pytorch.py`.
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError(
        "The official mamba-ssm package is not available in this Python "
        "environment. Install the CUDA-enabled package, or run with "
        "--ssm-backend pytorch to use the local reference implementation."
    ) from exc


class MambaSSMResidualBlock(nn.Module):
    """Pre-norm residual wrapper around the official fused Mamba block."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dropout=0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=int(d_model),
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            dt_rank=dt_rank,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaSSMSequenceEncoder(nn.Module):
    """Encode one sensor sequence into a pooled feature vector."""

    def __init__(
        self,
        input_channels,
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                MambaSSMResidualBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.output_norm(x)


class MambaSSMSequenceClassifier(nn.Module):
    """Single-modality sequence classifier built from official Mamba blocks."""

    def __init__(
        self,
        input_channels,
        num_classes,
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = MambaSSMSequenceEncoder(
            input_channels=input_channels,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


class DualBranchMambaSSMSequenceClassifier(nn.Module):
    """
    Dual-branch classifier using official Mamba blocks.

    The constructor mirrors the local pure-PyTorch implementation so existing
    training wrappers can swap backends with minimal code changes.
    """

    def __init__(
        self,
        emg_input_channels,
        imu_input_channels,
        num_classes,
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dropout=0.1,
        fusion_hidden_dim=None,
    ):
        super().__init__()
        self.emg_encoder = MambaSSMSequenceEncoder(
            input_channels=emg_input_channels,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )
        self.imu_encoder = MambaSSMSequenceEncoder(
            input_channels=imu_input_channels,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
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
    "MambaSSMResidualBlock",
    "MambaSSMSequenceEncoder",
    "MambaSSMSequenceClassifier",
    "DualBranchMambaSSMSequenceClassifier",
]
