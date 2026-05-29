"""
Train a dual-branch Mamba-SSM model with adaptive cross-modal fusion.

This is a second, independent multimodal experiment entry point. It keeps the
same paired-data pipeline as train_mamba_emg_imu_dual_branch.py, but replaces
the concat fusion head with:

- token-level bidirectional cross-attention between EMG and IMU encodings
- modality-specific fallback classifiers
- a learned late-fusion gate over EMG logits, IMU logits, and fused logits

The design goal is to reduce the risk that a weak/noisy modality drags down a
stronger one. It still does not guarantee higher accuracy; use fold averages for
the final conclusion.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

import train_mamba_emg_imu_dual_branch as dual_base
from mamba_ssm_pytorch import MambaSSMResidualBlock

OUTPUT_DIR = Path("mamba_ssm_results")


def has_option(argv, option):
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv[1:])


def append_option_if_missing(argv, option, value):
    if has_option(argv, option):
        return argv
    return argv + [option, str(value)]


def resolve_attention_heads(d_model, requested_heads=4):
    requested_heads = max(1, int(requested_heads))
    if d_model % requested_heads == 0:
        return requested_heads
    for candidate in (8, 4, 2, 1):
        if candidate <= d_model and d_model % candidate == 0:
            return candidate
    return 1


class TokenMambaSSMSequenceEncoder(nn.Module):
    """Mamba-SSM encoder that keeps per-time-step tokens for cross-attention."""

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
        return self.output_norm(x)


class AdaptiveFusionMambaSSMSequenceClassifier(nn.Module):
    """
    Dual-branch Mamba-SSM classifier with cross-attention and gated late fusion.

    Output logits are a learned weighted sum of:
    - EMG-only logits
    - IMU-only logits
    - cross-modal fused logits
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
        attention_heads=4,
        modality_dropout=0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.modality_dropout = float(modality_dropout)
        self.attention_heads = resolve_attention_heads(self.d_model, attention_heads)

        self.emg_encoder = TokenMambaSSMSequenceEncoder(
            input_channels=emg_input_channels,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )
        self.imu_encoder = TokenMambaSSMSequenceEncoder(
            input_channels=imu_input_channels,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )

        self.emg_to_imu_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.imu_to_emg_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.emg_cross_norm = nn.LayerNorm(self.d_model)
        self.imu_cross_norm = nn.LayerNorm(self.d_model)

        self.emg_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_classes),
        )
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_classes),
        )

        hidden_dim = fusion_hidden_dim if fusion_hidden_dim is not None else self.d_model
        fusion_dim = self.d_model * 4
        self.fused_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.logit_gate = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 3),
        )

    def apply_modality_dropout(self, emg_features, imu_features):
        if not self.training or self.modality_dropout <= 0:
            return emg_features, imu_features

        batch_size = emg_features.size(0)
        device = emg_features.device
        drop_emg = torch.rand(batch_size, 1, device=device) < self.modality_dropout
        drop_imu = torch.rand(batch_size, 1, device=device) < self.modality_dropout
        drop_both = drop_emg & drop_imu
        drop_emg = drop_emg & ~drop_both
        drop_imu = drop_imu & ~drop_both

        emg_features = emg_features.masked_fill(drop_emg, 0.0)
        imu_features = imu_features.masked_fill(drop_imu, 0.0)
        return emg_features, imu_features

    def forward(self, emg_x, imu_x):
        emg_tokens = self.emg_encoder(emg_x)
        imu_tokens = self.imu_encoder(imu_x)

        emg_cross, _ = self.emg_to_imu_attention(
            emg_tokens,
            imu_tokens,
            imu_tokens,
            need_weights=False,
        )
        imu_cross, _ = self.imu_to_emg_attention(
            imu_tokens,
            emg_tokens,
            emg_tokens,
            need_weights=False,
        )
        emg_tokens = self.emg_cross_norm(emg_tokens + emg_cross)
        imu_tokens = self.imu_cross_norm(imu_tokens + imu_cross)

        emg_features = emg_tokens.mean(dim=1)
        imu_features = imu_tokens.mean(dim=1)
        emg_features, imu_features = self.apply_modality_dropout(emg_features, imu_features)

        interaction = torch.cat(
            [
                emg_features,
                imu_features,
                torch.abs(emg_features - imu_features),
                emg_features * imu_features,
            ],
            dim=-1,
        )

        emg_logits = self.emg_head(emg_features)
        imu_logits = self.imu_head(imu_features)
        fused_logits = self.fused_head(interaction)
        gate = torch.softmax(self.logit_gate(interaction), dim=-1)

        stacked_logits = torch.stack([emg_logits, imu_logits, fused_logits], dim=1)
        return torch.sum(gate.unsqueeze(-1) * stacked_logits, dim=1)


def resolve_adaptive_output_paths(explicit_prefix=None):
    prefix = explicit_prefix if explicit_prefix else "mamba_ssm_emg_imu_adaptive_5fold"
    prefix = Path(prefix).stem
    if prefix.endswith(".pth"):
        prefix = prefix[:-4]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        str(OUTPUT_DIR / f"best_model_{prefix}.pth"),
        str(OUTPUT_DIR / f"training_history_{prefix}.png"),
        str(OUTPUT_DIR / f"confusion_matrix_{prefix}.png"),
    )


def prepare_argv(argv):
    prepared = list(argv)
    prepared = append_option_if_missing(prepared, "--cv-mode", "kfold")
    prepared = append_option_if_missing(prepared, "--num-folds", 5)
    prepared = append_option_if_missing(prepared, "--fold-seed", 42)
    prepared = append_option_if_missing(prepared, "--emg-temporal-pooling", "rms")
    prepared = append_option_if_missing(prepared, "--emg-temporal-pool-size", 5)
    prepared = append_option_if_missing(prepared, "--num-epochs", 12)
    return prepared


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-prefix", type=str, default=None)
    wrapper_args, _ = parser.parse_known_args(sys.argv[1:])

    dual_base.DualBranchMambaSequenceClassifier = AdaptiveFusionMambaSSMSequenceClassifier
    dual_base.resolve_output_paths = resolve_adaptive_output_paths
    sys.argv = prepare_argv(sys.argv)

    if wrapper_args.output_prefix is None:
        print("Output prefix: mamba_ssm_emg_imu_adaptive_5fold")
    print("Model: dual-branch Mamba-SSM with adaptive cross-modal fusion")
    print("Fusion: bidirectional cross-attention + EMG/IMU/fused late-logit gate")
    dual_base.main()


if __name__ == "__main__":
    main()
