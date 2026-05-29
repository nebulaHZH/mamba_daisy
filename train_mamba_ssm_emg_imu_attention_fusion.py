"""
Train a dual-branch Mamba-SSM model with attention-based EMG/IMU fusion.

This entry point keeps the same data pipeline and training loop as
train_mamba_emg_imu_dual_branch.py, but replaces the simple concat fusion head
with a small modality-attention fusion module.

Important: attention fusion is a better experimental design than plain concat,
but it cannot guarantee higher accuracy than single-modality baselines. Use
the fold averages to support the final conclusion.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

import train_mamba_emg_imu_dual_branch as dual_base
from mamba_ssm_pytorch import MambaSSMSequenceEncoder

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


class AttentionFusionMambaSSMSequenceClassifier(nn.Module):
    """
    Dual-branch Mamba-SSM classifier with modality attention fusion.

    Flow:
    - encode sEMG and IMU independently with Mamba-SSM encoders
    - run self-attention over the two modality tokens
    - use a learned query token to attend to the modality tokens
    - use a learned soft gate as a residual reliability weighting
    - classify the fused representation
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
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.attention_heads = resolve_attention_heads(self.d_model, attention_heads)

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

        self.modality_type_embedding = nn.Parameter(torch.zeros(1, 2, self.d_model))
        nn.init.normal_(self.modality_type_embedding, mean=0.0, std=0.02)

        self.modality_self_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)
        self.query_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.token_norm = nn.LayerNorm(self.d_model)
        self.context_norm = nn.LayerNorm(self.d_model)
        self.gate = nn.Sequential(
            nn.LayerNorm(self.d_model * 4),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 2),
        )

        hidden_dim = fusion_hidden_dim if fusion_hidden_dim is not None else self.d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model * 2),
            nn.Linear(self.d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, emg_x, imu_x):
        emg_features = self.emg_encoder(emg_x)
        imu_features = self.imu_encoder(imu_x)

        modality_tokens = torch.stack([emg_features, imu_features], dim=1)
        modality_tokens = modality_tokens + self.modality_type_embedding

        attended_tokens, _ = self.modality_self_attention(
            modality_tokens,
            modality_tokens,
            modality_tokens,
            need_weights=False,
        )
        modality_tokens = self.token_norm(modality_tokens + attended_tokens)

        query = self.query_token.expand(emg_x.size(0), -1, -1)
        context, _ = self.query_attention(
            query,
            modality_tokens,
            modality_tokens,
            need_weights=False,
        )
        attention_context = self.context_norm(context.squeeze(1))

        gate_input = torch.cat(
            [
                modality_tokens[:, 0],
                modality_tokens[:, 1],
                torch.abs(modality_tokens[:, 0] - modality_tokens[:, 1]),
                modality_tokens[:, 0] * modality_tokens[:, 1],
            ],
            dim=-1,
        )
        modality_weights = torch.softmax(self.gate(gate_input), dim=-1)
        gated_context = (
            modality_weights[:, 0:1] * modality_tokens[:, 0]
            + modality_weights[:, 1:2] * modality_tokens[:, 1]
        )

        fused_features = torch.cat([attention_context, gated_context], dim=-1)
        return self.classifier(fused_features)


def resolve_attention_output_paths(explicit_prefix=None):
    prefix = explicit_prefix if explicit_prefix else "mamba_ssm_emg_imu_attention_5fold"
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

    dual_base.DualBranchMambaSequenceClassifier = AttentionFusionMambaSSMSequenceClassifier
    dual_base.resolve_output_paths = resolve_attention_output_paths
    sys.argv = prepare_argv(sys.argv)

    if wrapper_args.output_prefix is None:
        print("Output prefix: mamba_ssm_emg_imu_attention_5fold")
    print("Model: dual-branch Mamba-SSM with attention fusion")
    print("Fusion: modality self-attention + learned query attention + residual soft gate")
    dual_base.main()


if __name__ == "__main__":
    main()
