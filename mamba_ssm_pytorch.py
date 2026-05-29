"""
Pure PyTorch selective state-space Mamba blocks.

This file intentionally does not replace mamba_models.py. It provides a
smaller, more Mamba-like implementation that can be imported by a separate
training script when you want to compare against the current project model.

Notes:
- The selective scan here is implemented with a Python loop for portability.
  It follows the recurrent state-update structure, but it is not as fast as
  the fused CUDA kernels used by the official mamba-ssm package.
- Parameter count is reduced by using a low-rank dt projection.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_dt_rank(d_model):
    """Default dt rank used by common Mamba implementations."""
    return math.ceil(d_model / 16)


class SelectiveScanMambaBlock(nn.Module):
    """
    Mamba-style block with causal depthwise convolution and selective scan.

    Input shape: (batch, seq_len, d_model)
    Output shape: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = infer_dt_rank(self.d_model) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        a = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        a = a.unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.activation = nn.SiLU()
        self._init_dt(dt_min=dt_min, dt_max=dt_max, dt_init_floor=dt_init_floor)

    def _init_dt(self, dt_min, dt_max, dt_init_floor):
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.activation(x_branch)

        y = self.selective_scan(x_branch)
        y = y * self.activation(z_branch)
        return self.out_proj(y)

    def selective_scan(self, x):
        batch_size, seq_len, _ = x.shape

        x_dbl = self.x_proj(x)
        dt, b, c = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))

        a = -torch.exp(self.A_log.float())
        d = self.D.float()
        state = x.new_zeros(batch_size, self.d_inner, self.d_state)
        outputs = []

        for step in range(seq_len):
            x_t = x[:, step]
            dt_t = dt[:, step]
            b_t = b[:, step]
            c_t = c[:, step]

            delta_a = torch.exp(dt_t.unsqueeze(-1) * a.unsqueeze(0))
            delta_b_x = dt_t.unsqueeze(-1) * b_t.unsqueeze(1) * x_t.unsqueeze(-1)
            state = state * delta_a + delta_b_x
            y_t = torch.einsum("bdn,bn->bd", state, c_t)
            y_t = y_t + x_t * d.unsqueeze(0)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class MambaSSMResidualBlock(nn.Module):
    """Pre-norm residual wrapper around SelectiveScanMambaBlock."""

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
        self.mamba = SelectiveScanMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
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
    """Single-modality sequence classifier built from Mamba SSM blocks."""

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
    Dual-branch Mamba SSM classifier for paired EMG and IMU windows.

    The constructor mirrors DualBranchMambaSequenceClassifier so it can be
    swapped into a training script with minimal changes.
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
    "infer_dt_rank",
    "SelectiveScanMambaBlock",
    "MambaSSMResidualBlock",
    "MambaSSMSequenceEncoder",
    "MambaSSMSequenceClassifier",
    "DualBranchMambaSSMSequenceClassifier",
]
