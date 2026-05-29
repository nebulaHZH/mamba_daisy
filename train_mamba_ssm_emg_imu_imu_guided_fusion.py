"""
Train an IMU-guided multimodal Mamba-SSM model.

This experiment is designed for the case observed in the current results:
IMU is the stronger and more stable modality, while sEMG can be noisy across
subjects. The model therefore treats IMU as the main path and lets sEMG provide
a gated residual correction instead of forcing an equal concat fusion.

Key ideas:
- paired sEMG + IMU data, same 5-fold subject-grouped CV pipeline
- sEMG RMS pooling x5, so both modalities enter as 100-step sequences
- IMU-guided residual feature fusion
- late gate over IMU-only, sEMG-only, and fused logits, initialized toward IMU
- auxiliary branch losses during training so each branch learns useful logits

This improves the experimental design, but it still must be evaluated by fold
averages rather than assumed to beat the IMU-only baseline.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

import train_mamba_emg_imu_dual_branch as dual_base

OUTPUT_DIR = Path("mamba_ssm_results")


def get_option_value(argv, option, default=None):
    for index, arg in enumerate(argv[1:], start=1):
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(f"{option}="):
            return arg.split("=", 1)[1]
    return default


SSM_BACKEND = str(get_option_value(sys.argv, "--ssm-backend", "official")).lower()
if SSM_BACKEND == "official":
    from mamba_ssm_official import MambaSSMSequenceEncoder
elif SSM_BACKEND == "pytorch":
    from mamba_ssm_pytorch import MambaSSMSequenceEncoder
else:
    raise ValueError("--ssm-backend must be either 'official' or 'pytorch'.")


def has_option(argv, option):
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv[1:])


def remove_option(argv, option):
    filtered = [argv[0]]
    skip_next = False
    for index, arg in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            skip_next = index + 1 < len(argv)
            continue
        if arg.startswith(f"{option}="):
            continue
        filtered.append(arg)
    return filtered


def append_option_if_missing(argv, option, value):
    if has_option(argv, option):
        return argv
    return argv + [option, str(value)]


class IMUGuidedFusionMambaSSMSequenceClassifier(nn.Module):
    """Dual-branch Mamba-SSM classifier with IMU-guided residual fusion."""

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
        aux_loss_weight=0.2,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.aux_loss_weight = float(aux_loss_weight)

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

        self.emg_residual_adapter = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.residual_gate = nn.Sequential(
            nn.LayerNorm(self.d_model * 4),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid(),
        )

        self.fused_norm = nn.LayerNorm(self.d_model)
        self.emg_head = self._make_head(num_classes, dropout)
        self.imu_head = self._make_head(num_classes, dropout)
        self.fused_head = self._make_head(num_classes, dropout)

        hidden_dim = fusion_hidden_dim if fusion_hidden_dim is not None else self.d_model
        gate_input_dim = self.d_model * 5
        self.logit_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self._init_imu_preferred_gate()

    def _make_head(self, num_classes, dropout):
        return nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_classes),
        )

    def _init_imu_preferred_gate(self):
        last_linear = self.logit_gate[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            with torch.no_grad():
                # Gate order: IMU-only, EMG-only, fused. Start close to IMU,
                # then learn whether EMG/fused evidence should be trusted.
                last_linear.bias.copy_(torch.tensor([2.0, -2.0, 0.0]))

    def forward(self, emg_x, imu_x, return_aux=False):
        emg_features = self.emg_encoder(emg_x)
        imu_features = self.imu_encoder(imu_x)

        relation = torch.cat(
            [
                imu_features,
                emg_features,
                torch.abs(imu_features - emg_features),
                imu_features * emg_features,
            ],
            dim=-1,
        )
        emg_residual = self.emg_residual_adapter(emg_features)
        residual_weight = self.residual_gate(relation)
        fused_features = self.fused_norm(imu_features + residual_weight * emg_residual)

        imu_logits = self.imu_head(imu_features)
        emg_logits = self.emg_head(emg_features)
        fused_logits = self.fused_head(fused_features)

        gate_input = torch.cat(
            [
                imu_features,
                emg_features,
                fused_features,
                torch.abs(imu_features - emg_features),
                imu_features * emg_features,
            ],
            dim=-1,
        )
        logit_weights = torch.softmax(self.logit_gate(gate_input), dim=-1)
        stacked_logits = torch.stack([imu_logits, emg_logits, fused_logits], dim=1)
        final_logits = torch.sum(logit_weights.unsqueeze(-1) * stacked_logits, dim=1)

        if return_aux:
            return {
                "logits": final_logits,
                "imu_logits": imu_logits,
                "emg_logits": emg_logits,
                "fused_logits": fused_logits,
                "logit_weights": logit_weights,
            }
        return final_logits


def compute_loss(outputs, labels, criterion, aux_loss_weight=0.2):
    final_loss = criterion(outputs["logits"], labels)
    aux_loss = (
        criterion(outputs["imu_logits"], labels)
        + criterion(outputs["emg_logits"], labels)
        + criterion(outputs["fused_logits"], labels)
    ) / 3.0
    return final_loss + aux_loss_weight * aux_loss


def train_epoch(model, dataloader, criterion, optimizer, device, amp_enabled=False, grad_scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train")
    for emg_sequences, imu_sequences, labels in pbar:
        emg_sequences = emg_sequences.to(device, non_blocking=amp_enabled)
        imu_sequences = imu_sequences.to(device, non_blocking=amp_enabled)
        labels = labels.to(device, non_blocking=amp_enabled)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(emg_sequences, imu_sequences, return_aux=True)
            loss = compute_loss(
                outputs,
                labels,
                criterion,
                aux_loss_weight=getattr(model, "aux_loss_weight", 0.2),
            )

        if amp_enabled and grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        logits = outputs["logits"]
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"})

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device, amp_enabled=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for emg_sequences, imu_sequences, labels in dataloader:
            emg_sequences = emg_sequences.to(device, non_blocking=amp_enabled)
            imu_sequences = imu_sequences.to(device, non_blocking=amp_enabled)
            labels = labels.to(device, non_blocking=amp_enabled)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(emg_sequences, imu_sequences)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), 100.0 * correct / total, all_preds, all_labels


def resolve_output_paths(explicit_prefix=None):
    default_prefix = f"mamba_ssm_{SSM_BACKEND}_emg_imu_imu_guided_5fold"
    prefix = explicit_prefix if explicit_prefix else default_prefix
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
    prepared = remove_option(list(argv), "--ssm-backend")
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
    parser.add_argument("--ssm-backend", choices=("official", "pytorch"), default="official")
    wrapper_args, _ = parser.parse_known_args(sys.argv[1:])

    dual_base.DualBranchMambaSequenceClassifier = IMUGuidedFusionMambaSSMSequenceClassifier
    dual_base.train_epoch = train_epoch
    dual_base.evaluate = evaluate
    dual_base.resolve_output_paths = resolve_output_paths
    sys.argv = prepare_argv(sys.argv)

    if wrapper_args.output_prefix is None:
        print(f"Output prefix: mamba_ssm_{SSM_BACKEND}_emg_imu_imu_guided_5fold")
    print("Model: IMU-guided multimodal Mamba-SSM")
    print(f"SSM backend: {SSM_BACKEND}")
    print("Fusion: IMU main path + gated sEMG residual + auxiliary branch losses")
    dual_base.main()


if __name__ == "__main__":
    main()
