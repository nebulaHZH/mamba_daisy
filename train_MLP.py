from __future__ import annotations

import argparse
import copy
import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_mamba_emg_imu import (
    FileAwareBatchSampler,
    LABEL_DISPLAY_NAMES,
    LABEL_NAMES,
    build_leave_one_subject_out_folds,
    build_subject_group_kfold_folds,
    describe_training_device,
    resolve_training_device,
    summarize_epoch_windows,
    summarize_split,
)
from train_mamba_emg_imu_dual_branch import (
    PairedSequenceWindowDataset,
    count_model_parameters,
    synchronize_device,
)
from train_svm import (
    compute_metrics,
    configure_style,
    estimate_fit_memory_gb,
    export_feature_excel,
    fit_single_scaler,
    get_feature_names,
    materialize_single,
    plot_confusion,
    plot_fold_accuracy,
    resolve_dual_setup,
    resolve_feature_excel_path,
    resolve_single_setup,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


warnings.filterwarnings("ignore")
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold MLP baseline on paired sEMG/IMU sequences.")
    parser.add_argument("--input-mode", choices=("single", "dual"), default="dual")
    parser.add_argument("--window-ms", type=float, default=500.0)
    parser.add_argument("--stride-ms", type=float, default=250.0)
    parser.add_argument("--max-files-per-class", type=int, default=None)
    parser.add_argument("--index-cache", type=str, default=None)
    parser.add_argument("--rebuild-index-cache", action="store_true")
    parser.add_argument("--max-train-windows-per-file", type=int, default=8)
    parser.add_argument("--max-test-windows-per-file", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=128)
    parser.add_argument("--max-fit-memory-gb", type=float, default=6.0)
    parser.add_argument("--emg-zc-threshold", type=float, default=1e-3)
    parser.add_argument("--emg-ssc-threshold", type=float, default=1e-3)
    parser.add_argument("--imu-zc-threshold", type=float, default=1e-4)
    parser.add_argument("--imu-ssc-threshold", type=float, default=1e-4)

    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--sensor-folder", type=str, default="emg")
    parser.add_argument("--sample-rate-hz", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--feature-mask", type=str, default=None)
    parser.add_argument("--expected-channels", type=int, default=None)

    parser.add_argument("--emg-data-root", type=str, default=None)
    parser.add_argument("--imu-data-root", type=str, default=None)
    parser.add_argument("--emg-sample-rate-hz", type=float, default=None)
    parser.add_argument("--imu-sample-rate-hz", type=float, default=None)
    parser.add_argument("--emg-window-size", type=int, default=None)
    parser.add_argument("--imu-window-size", type=int, default=None)
    parser.add_argument("--emg-stride", type=int, default=None)
    parser.add_argument("--imu-stride", type=int, default=None)
    parser.add_argument("--emg-feature-mask", type=str, default=None)
    parser.add_argument("--imu-feature-mask", type=str, default=None)
    parser.add_argument("--emg-downsample-mode", choices=("mean", "rms"), default="mean")
    parser.add_argument("--no-align-emg-to-imu", action="store_true")
    parser.add_argument("--emg-expected-channels", type=int, default=None)
    parser.add_argument("--imu-expected-channels", type=int, default=None)

    parser.add_argument("--stair-up-token", choices=("l", "r"), default="l")
    parser.add_argument("--ramp-up-token", choices=("l", "r"), default="l")

    parser.add_argument("--hidden-dims", type=str, default="64,32")
    parser.add_argument("--mlp-pool-steps", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-subject-mode", choices=("rotate", "first", "last"), default="rotate")
    parser.add_argument("--cv-mode", choices=("loso", "kfold"), default="kfold")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--require-cuda", action="store_true")

    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--export-feature-excel", action="store_true")
    parser.add_argument("--feature-excel-path", type=Path, default=None)
    parser.add_argument("--export-feature-max-windows-per-file", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def format_ms_tag(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    w = format_ms_tag(args.window_ms)
    s = format_ms_tag(args.stride_ms)
    if args.input_mode == "dual":
        return Path.cwd() / f"MLP{w}ms滑窗{s}ms步长sEMG和imu结果图"
    sensor_tag = "imu" if str(args.sensor_folder).lower() == "imu" else "sEMG"
    return Path.cwd() / f"MLP{w}ms滑窗{s}ms步长{sensor_tag}结果图"


def parse_hidden_dims(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    hidden_dims = [int(part) for part in parts]
    if not hidden_dims:
        raise ValueError("--hidden-dims must contain at least one integer.")
    return hidden_dims


def resolve_mlp_pool_steps(requested_steps: int | None, sequence_length: int) -> int:
    requested_steps = 0 if requested_steps is None else int(requested_steps)
    sequence_length = int(sequence_length)
    if requested_steps <= 0:
        return sequence_length
    return max(1, min(requested_steps, sequence_length))


def resolve_device(device_arg: str, require_cuda: bool = False) -> torch.device:
    return resolve_training_device(device_arg, require_cuda)


def resolve_cv_folds(records: list, args: argparse.Namespace) -> tuple[list, str, str]:
    if args.cv_mode == "loso":
        folds = build_leave_one_subject_out_folds(records)
        if not folds:
            raise ValueError(
                "LOSO-CV requires at least two subjects. Increase --max-files-per-class if your subset keeps only one subject."
            )
        return folds, "LOSO", "loso"

    folds = build_subject_group_kfold_folds(
        records,
        num_folds=int(args.num_folds),
        random_state=int(args.fold_seed),
    )
    if not folds:
        raise ValueError("K-fold CV requires records from at least two subject groups.")
    return folds, f"{int(args.num_folds)}-fold subject-grouped CV", f"kfold{int(args.num_folds)}"


def resolve_emg_alignment(setup: dict[str, object], args: argparse.Namespace) -> dict[str, int | str | bool]:
    emg_window_size = int(setup["emg_window_size"])
    imu_window_size = int(setup["imu_window_size"])
    emg_stride = int(setup["emg_stride"])
    imu_stride = int(setup["imu_stride"])
    if args.no_align_emg_to_imu:
        return {
            "enabled": False,
            "mode": "none",
            "factor": 1,
            "emg_model_window_size": emg_window_size,
        }

    if emg_window_size < imu_window_size:
        raise ValueError(
            "Cannot downsample sEMG to IMU length because "
            f"emg_window_size={emg_window_size} is smaller than imu_window_size={imu_window_size}."
        )
    if emg_window_size % imu_window_size != 0:
        raise ValueError(
            "Automatic sEMG/IMU alignment requires emg_window_size to be divisible by "
            f"imu_window_size, got {emg_window_size} and {imu_window_size}."
        )

    factor = emg_window_size // imu_window_size
    if emg_stride % factor != 0 or emg_stride // factor != imu_stride:
        raise ValueError(
            "Automatic sEMG/IMU alignment requires the stride ratio to match the window "
            f"downsample factor. Got emg_stride={emg_stride}, imu_stride={imu_stride}, factor={factor}."
        )

    mode = "none" if factor == 1 else str(args.emg_downsample_mode)
    return {
        "enabled": factor != 1,
        "mode": mode,
        "factor": factor,
        "emg_model_window_size": imu_window_size,
    }


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DualSequenceMLPClassifier(nn.Module):
    def __init__(
        self,
        emg_window_size: int,
        emg_input_channels: int,
        imu_window_size: int,
        imu_input_channels: int,
        pool_steps: int,
        hidden_dims: list[int],
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emg_pool_steps = resolve_mlp_pool_steps(pool_steps, emg_window_size)
        self.imu_pool_steps = resolve_mlp_pool_steps(pool_steps, imu_window_size)
        self.emg_pool = nn.AdaptiveAvgPool1d(self.emg_pool_steps)
        self.imu_pool = nn.AdaptiveAvgPool1d(self.imu_pool_steps)
        input_dim = (
            self.emg_pool_steps * int(emg_input_channels)
            + self.imu_pool_steps * int(imu_input_channels)
        )
        self.mlp = MLPClassifier(input_dim, hidden_dims, num_classes, dropout)

    def forward(self, emg_sequences: torch.Tensor, imu_sequences: torch.Tensor) -> torch.Tensor:
        emg_pooled = self.emg_pool(emg_sequences.transpose(1, 2)).transpose(1, 2)
        imu_pooled = self.imu_pool(imu_sequences.transpose(1, 2)).transpose(1, 2)
        emg_flat = emg_pooled.flatten(start_dim=1)
        imu_flat = imu_pooled.flatten(start_dim=1)
        return self.mlp(torch.cat([emg_flat, imu_flat], dim=1))


def build_mlp(input_dim: int, args: argparse.Namespace) -> MLPClassifier:
    return MLPClassifier(
        input_dim=input_dim,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        num_classes=len(LABEL_NAMES),
        dropout=float(args.dropout),
    )


def build_sequence_mlp(setup: dict[str, object], args: argparse.Namespace) -> DualSequenceMLPClassifier:
    alignment = setup["emg_alignment"]
    return DualSequenceMLPClassifier(
        emg_window_size=int(alignment["emg_model_window_size"]),
        emg_input_channels=int(setup["selected_emg_channels"]),
        imu_window_size=int(setup["imu_window_size"]),
        imu_input_channels=int(setup["selected_imu_channels"]),
        pool_steps=int(args.mlp_pool_steps),
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        num_classes=len(LABEL_NAMES),
        dropout=float(args.dropout),
    )


def get_sequence_mlp_input_dim(setup: dict[str, object], args: argparse.Namespace) -> tuple[int, int, int]:
    emg_steps = resolve_mlp_pool_steps(
        int(args.mlp_pool_steps),
        int(setup["emg_alignment"]["emg_model_window_size"]),
    )
    imu_steps = resolve_mlp_pool_steps(int(args.mlp_pool_steps), int(setup["imu_window_size"]))
    input_dim = (
        emg_steps * int(setup["selected_emg_channels"])
        + imu_steps * int(setup["selected_imu_channels"])
    )
    return input_dim, emg_steps, imu_steps


def create_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_num_workers(requested_workers: int | None) -> int:
    if requested_workers is not None:
        return max(0, int(requested_workers))
    return 0


def build_loader_kwargs(device: torch.device, num_workers: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def create_sequence_dataset(
    records: list,
    setup: dict[str, object],
    args: argparse.Namespace,
    fit_scaler: bool = False,
    emg_scaler=None,
    imu_scaler=None,
    max_windows_per_file: int | None = None,
) -> PairedSequenceWindowDataset:
    alignment = setup["emg_alignment"]
    return PairedSequenceWindowDataset(
        records,
        emg_window_size=int(setup["emg_window_size"]),
        emg_stride=int(setup["emg_stride"]),
        imu_window_size=int(setup["imu_window_size"]),
        imu_stride=int(setup["imu_stride"]),
        emg_scaler=emg_scaler,
        imu_scaler=imu_scaler,
        fit_scaler=fit_scaler,
        emg_feature_mask=setup["emg_feature_mask"],
        imu_feature_mask=setup["imu_feature_mask"],
        cache_size=int(args.cache_size),
        max_windows_per_file=max_windows_per_file,
        emg_temporal_pooling=str(alignment["mode"]),
        emg_temporal_pool_size=int(alignment["factor"]),
    )


def create_sequence_loaders(
    train_dataset: PairedSequenceWindowDataset,
    eval_dataset: PairedSequenceWindowDataset,
    args: argparse.Namespace,
    loader_kwargs: dict[str, object],
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=FileAwareBatchSampler(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle_files=True,
        ),
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, eval_loader


def run_sequence_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    grad_scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for emg_sequences, imu_sequences, labels in loader:
        emg_sequences = emg_sequences.to(device, non_blocking=amp_enabled)
        imu_sequences = imu_sequences.to(device, non_blocking=amp_enabled)
        labels = labels.to(device, non_blocking=amp_enabled)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(emg_sequences, imu_sequences)
            loss = criterion(logits, labels)

        if amp_enabled and grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * int(labels.size(0))
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += int(labels.size(0))
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def run_sequence_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    preds: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    with torch.no_grad():
        for emg_sequences, imu_sequences, labels in loader:
            emg_sequences = emg_sequences.to(device, non_blocking=amp_enabled)
            imu_sequences = imu_sequences.to(device, non_blocking=amp_enabled)
            labels = labels.to(device, non_blocking=amp_enabled)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(emg_sequences, imu_sequences)
                loss = criterion(logits, labels)

            predicted = logits.argmax(dim=1)
            total_loss += float(loss.item()) * int(labels.size(0))
            total_correct += int((predicted == labels).sum().item())
            total_count += int(labels.size(0))
            preds.append(predicted.cpu().numpy())
            labels_out.append(labels.cpu().numpy())

    return (
        total_loss / max(total_count, 1),
        total_correct / max(total_count, 1),
        np.concatenate(preds, axis=0).astype(np.int64, copy=False),
        np.concatenate(labels_out, axis=0).astype(np.int64, copy=False),
    )


def fit_sequence_with_early_stopping(
    train_dataset: PairedSequenceWindowDataset,
    val_dataset: PairedSequenceWindowDataset,
    setup: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    loader_kwargs: dict[str, object],
    seed_offset: int,
) -> tuple[int, list[dict[str, float]], dict[str, torch.Tensor]]:
    torch.manual_seed(RANDOM_SEED + seed_offset)
    np.random.seed(RANDOM_SEED + seed_offset)

    model = build_sequence_mlp(setup, args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_loader, val_loader = create_sequence_loaders(train_dataset, val_dataset, args, loader_kwargs)

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.max_epochs) + 1):
        train_loss, train_acc = run_sequence_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp_enabled=amp_enabled,
            grad_scaler=grad_scaler,
        )
        val_loss, val_acc, _, _ = run_sequence_eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
        )

        if val_loss < best_val_loss - float(args.min_delta):
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(args.patience):
                break

    return best_epoch, history, best_state


def train_sequence_fixed_epochs(
    train_dataset: PairedSequenceWindowDataset,
    setup: dict[str, object],
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    loader_kwargs: dict[str, object],
    seed_offset: int,
) -> nn.Module:
    torch.manual_seed(RANDOM_SEED + seed_offset)
    np.random.seed(RANDOM_SEED + seed_offset)

    model = build_sequence_mlp(setup, args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=FileAwareBatchSampler(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle_files=True,
        ),
        **loader_kwargs,
    )

    for _ in range(max(int(epochs), 1)):
        run_sequence_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp_enabled=amp_enabled,
            grad_scaler=grad_scaler,
        )
    return model


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
        batch_y = batch_y.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(batch_y.size(0))
        total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total_count += int(batch_y.size(0))
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            batch_y = batch_y.to(device, non_blocking=device.type == "cuda")
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * int(batch_y.size(0))
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total_count += int(batch_y.size(0))
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def predict_in_batches(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            logits = model(batch_x)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.int64, copy=False)


def choose_validation_subject(train_records: list, fold_index: int, mode: str) -> str:
    subject_ids = sorted({record.subject_id for record in train_records})
    if not subject_ids:
        raise ValueError("No training records are available for validation split.")
    if mode == "first":
        return subject_ids[0]
    if mode == "last":
        return subject_ids[-1]
    return subject_ids[(fold_index - 1) % len(subject_ids)]


def split_train_val_records(train_records: list, fold_index: int, mode: str) -> tuple[list, list, str]:
    unique_subjects = sorted({record.subject_id for record in train_records})
    if len(unique_subjects) >= 2:
        val_subject_id = choose_validation_subject(train_records, fold_index, mode)
        val_records = [record for record in train_records if record.subject_id == val_subject_id]
        inner_train_records = [record for record in train_records if record.subject_id != val_subject_id]
        return inner_train_records, val_records, val_subject_id

    split_index = max(1, int(round(len(train_records) * 0.2)))
    val_records = list(train_records[:split_index])
    inner_train_records = list(train_records[split_index:])
    if not inner_train_records:
        inner_train_records = list(train_records[:-1])
        val_records = [train_records[-1]]
    return inner_train_records, val_records, "record_split"


def fit_with_early_stopping(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> tuple[int, list[dict[str, float]], dict[str, torch.Tensor]]:
    torch.manual_seed(RANDOM_SEED + seed_offset)
    np.random.seed(RANDOM_SEED + seed_offset)

    model = build_mlp(int(x_train.shape[1]), args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    train_loader = create_loader(x_train, y_train, int(args.batch_size), shuffle=True)
    val_loader = create_loader(x_val, y_val, int(args.batch_size), shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.max_epochs) + 1):
        train_loss, train_acc = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_eval_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
        )

        if val_loss < best_val_loss - float(args.min_delta):
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(args.patience):
                break

    return best_epoch, history, best_state


def train_fixed_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> nn.Module:
    torch.manual_seed(RANDOM_SEED + seed_offset)
    np.random.seed(RANDOM_SEED + seed_offset)

    model = build_mlp(int(x_train.shape[1]), args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    train_loader = create_loader(x_train, y_train, int(args.batch_size), shuffle=True)

    for _ in range(max(int(epochs), 1)):
        run_train_epoch(model, train_loader, criterion, optimizer, device)
    return model


def plot_training_history(history_df: pd.DataFrame, title: str, output_path: Path) -> None:
    if plt is None or sns is None or history_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#2F5C85")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#D97A2B")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_acc"] * 100.0, label="Train Acc", color="#2F5C85")
    axes[1].plot(history_df["epoch"], history_df["val_acc"] * 100.0, label="Val Acc", color="#D97A2B")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle(title, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_history(histories: list[pd.DataFrame], output_path: Path) -> None:
    if plt is None or sns is None or not histories:
        return
    max_len = max(len(history) for history in histories)
    epochs = np.arange(1, max_len + 1, dtype=np.int64)
    metrics = ["train_loss", "val_loss", "train_acc", "val_acc"]
    padded = {metric: np.full((len(histories), max_len), np.nan, dtype=np.float64) for metric in metrics}
    for row_idx, history in enumerate(histories):
        for metric in metrics:
            values = history[metric].to_numpy(dtype=np.float64, copy=False)
            padded[metric][row_idx, : len(values)] = values

    aggregate_df = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": np.nanmean(padded["train_loss"], axis=0),
            "val_loss": np.nanmean(padded["val_loss"], axis=0),
            "train_acc": np.nanmean(padded["train_acc"], axis=0),
            "val_acc": np.nanmean(padded["val_acc"], axis=0),
        }
    )
    plot_training_history(aggregate_df, "MLP Aggregate Training History", output_path)


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    summary: dict[str, object],
    fold_df: pd.DataFrame,
) -> None:
    cv_name = "5-fold subject-grouped cross-validation" if args.cv_mode == "kfold" else "leave-one-subject-out cross-validation"
    lines = [f"# MLP {cv_name} Experiment Summary", "", "## Protocol"]
    if args.input_mode == "dual":
        lines.append("- Input: paired sEMG and IMU sequences.")
        lines.append("- Model input: aligned raw windows are flattened and passed directly into the MLP.")
        alignment = summary.get("emg_alignment") or {}
        if alignment:
            lines.append(
                "- Alignment: sEMG windows are downsampled before model input "
                f"using {alignment.get('mode')} x{alignment.get('factor')}."
            )
        lines.append(f"- Compression: adaptive temporal pooling to {summary.get('mlp_pool_steps')} steps before flattening.")
    else:
        lines.append("- Input: single-modality windows transformed into handcrafted channel-wise features.")
    lines.append(f"- Evaluation: {cv_name}.")
    lines.append("- Leakage control: scaler(s) fit only on inner-training records within each CV fold.")
    lines.append("- Leakage control: MLP is re-instantiated for every fold and again for final retraining.")
    lines.append("- Early stopping: validation loss is monitored on a validation subject sampled from the training subjects only.")
    lines.append("- Final test prediction uses a fresh model retrained on the full training fold for the selected best epoch.")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append(f"- Folds completed: {summary['folds_completed']}")
    lines.append(f"- Mean accuracy: {summary['mean_accuracy_pct']:.2f}% +/- {summary['std_accuracy_pct']:.2f}%")
    lines.append(f"- Mean weighted F1: {summary['mean_f1_weighted_pct']:.2f}% +/- {summary['std_f1_weighted_pct']:.2f}%")
    lines.append(f"- Mean macro F1: {summary['mean_f1_macro_pct']:.2f}% +/- {summary['std_f1_macro_pct']:.2f}%")
    lines.append(f"- Aggregate accuracy: {summary['aggregate_accuracy_pct']:.2f}%")
    lines.append(f"- Mean selected epoch: {summary['mean_best_epoch']:.2f}")
    lines.append(f"- Mean train time per fold: {summary['mean_train_time_sec']:.2f}s")
    if not fold_df.empty:
        best_row = fold_df.loc[fold_df["accuracy_pct"].idxmax()]
        worst_row = fold_df.loc[fold_df["accuracy_pct"].idxmin()]
        lines.extend(
            [
                "",
                "## Subject-level Extremes",
                f"- Best subject: {best_row['subject_id']} ({best_row['accuracy_pct']:.2f}%)",
                f"- Worst subject: {worst_row['subject_id']} ({worst_row['accuracy_pct']:.2f}%)",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_tag = f"kfold{int(args.num_folds)}" if args.cv_mode == "kfold" else "loso"
    (output_dir / f"mlp_{cv_tag}_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    setup = resolve_dual_setup(args) if args.input_mode == "dual" else resolve_single_setup(args)
    if args.input_mode == "dual":
        setup["emg_alignment"] = resolve_emg_alignment(setup, args)
    artifact_prefix = (
        f"mlp_emg_imu_dual_{cv_tag}"
        if args.input_mode == "dual"
        else f"mlp_{setup['sensor_folder']}_{cv_tag}"
    )
    feature_names = [] if args.input_mode == "dual" else get_feature_names(setup)
    device = resolve_device(args.device, args.require_cuda)
    amp_enabled = device.type == "cuda"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    resolved_num_workers = resolve_num_workers(args.num_workers)
    loader_kwargs = build_loader_kwargs(device, resolved_num_workers)
    cv_folds, cv_label, cv_tag = resolve_cv_folds(setup["records"], args)

    print("=" * 80)
    headline = "paired sEMG/IMU sequences" if args.input_mode == "dual" else "handcrafted window features"
    print(f"MLP baseline on {headline} with {cv_label}")
    print("=" * 80)
    print(f"Input mode: {args.input_mode}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {describe_training_device(device)}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"DataLoader workers: {resolved_num_workers}")
    print("Important: scaler(s) are fit inside each fold, early stopping uses training subjects only, and the final MLP is retrained per fold.")
    print(f"Subjects found: {len(setup['subject_ids'])} -> {', '.join(setup['subject_ids'])}")
    print(f"Classes: {', '.join(LABEL_DISPLAY_NAMES)}")
    if args.input_mode == "dual":
        alignment = setup["emg_alignment"]
        print(
            "sEMG -> IMU alignment: "
            f"{alignment['mode']} x{alignment['factor']} -> "
            f"{alignment['emg_model_window_size']} sEMG time steps"
        )
        print(f"sEMG window/stride: {setup['emg_window_size']} / {setup['emg_stride']} samples")
        print(f"IMU window/stride: {setup['imu_window_size']} / {setup['imu_stride']} samples")

    if args.export_feature_excel:
        if args.input_mode == "dual":
            print("Feature Excel export skipped: dual-mode MLP now trains directly on sequences.")
        else:
            feature_excel_path = resolve_feature_excel_path(args, output_dir)
            print(f"Exporting extracted features to Excel: {feature_excel_path}")
            export_feature_excel(setup, args, feature_names, feature_excel_path)
            print(f"Feature Excel saved: {feature_excel_path}")
    elif args.input_mode == "dual":
        print("Feature extraction: disabled for dual-mode MLP; raw aligned sequences are used.")

    print(f"CV folds: {len(cv_folds)}")
    if args.cv_mode == "kfold":
        for fold_index, (fold_id, _, held_out_records) in enumerate(cv_folds, start=1):
            held_out_subjects = sorted({record.subject_id for record in held_out_records})
            print(f"  Fold {fold_index:02d} ({fold_id}): held-out subjects {', '.join(held_out_subjects)}")

    fold_results: list[dict[str, object]] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    fold_histories: list[pd.DataFrame] = []

    for fold_index, (subject_id, train_records, test_records) in enumerate(cv_folds, start=1):
        print("\n" + "=" * 80)
        held_out_subjects = sorted({record.subject_id for record in test_records})
        print(f"Fold {fold_index:02d}/{len(cv_folds)} - Held-out group: {subject_id}")
        print(f"Held-out subjects: {', '.join(held_out_subjects)}")
        print("=" * 80)
        summarize_split("Train", train_records)
        summarize_split("Test", test_records)

        inner_train_records, val_records, val_subject_id = split_train_val_records(train_records, fold_index, args.val_subject_mode)
        print(f"Validation subject for early stopping: {val_subject_id}")
        summarize_split("Inner Train", inner_train_records)
        summarize_split("Validation", val_records)

        search_start = perf_counter()
        if args.input_mode == "dual":
            print("\nBuilding sequence datasets for early stopping...")
            inner_train_dataset = create_sequence_dataset(
                inner_train_records,
                setup,
                args,
                fit_scaler=True,
                max_windows_per_file=args.max_train_windows_per_file,
            )
            val_dataset = create_sequence_dataset(
                val_records,
                setup,
                args,
                emg_scaler=inner_train_dataset.emg_scaler,
                imu_scaler=inner_train_dataset.imu_scaler,
                max_windows_per_file=args.max_train_windows_per_file,
            )
            summarize_epoch_windows("Inner Train", inner_train_dataset)
            summarize_epoch_windows("Validation", val_dataset)
            input_dim, emg_pool_steps, imu_pool_steps = get_sequence_mlp_input_dim(setup, args)
            preview_model = build_sequence_mlp(setup, args)
            model_parameters, trainable_parameters = count_model_parameters(preview_model)
            print(f"Sequence MLP input dim: {input_dim}")
            print(f"MLP temporal pool steps: EMG={emg_pool_steps}, IMU={imu_pool_steps}")
            print(f"Model parameters: {model_parameters:,} (trainable: {trainable_parameters:,})")
            synchronize_device(device)
            best_epoch, history_rows, _ = fit_sequence_with_early_stopping(
                train_dataset=inner_train_dataset,
                val_dataset=val_dataset,
                setup=setup,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                loader_kwargs=loader_kwargs,
                seed_offset=fold_index,
            )
        else:
            print("\nFitting scaler on inner-training windows only...")
            scaler = fit_single_scaler(inner_train_records, setup, args.max_train_windows_per_file, args)
            x_inner_train, y_inner_train = materialize_single(
                inner_train_records,
                setup,
                args.max_train_windows_per_file,
                scaler,
                args,
            )
            x_val, y_val = materialize_single(
                val_records,
                setup,
                args.max_train_windows_per_file,
                scaler,
                args,
            )

            estimated_fit_memory_gb = estimate_fit_memory_gb(x_inner_train)
            print(f"Inner-train matrix: {x_inner_train.shape}")
            print(f"Validation matrix: {x_val.shape}")
            print(f"Estimated matrix memory if materialized as float64: {estimated_fit_memory_gb:.2f} GiB")
            if estimated_fit_memory_gb > float(args.max_fit_memory_gb):
                raise MemoryError(
                    "The training matrix is too large for this configuration. "
                    f"Estimated float64 matrix memory is {estimated_fit_memory_gb:.2f} GiB, "
                    f"which exceeds --max-fit-memory-gb={args.max_fit_memory_gb}. "
                    "Try lowering --max-train-windows-per-file or using a smaller subset first."
                )
            best_epoch, history_rows, _ = fit_with_early_stopping(
                x_train=x_inner_train,
                y_train=y_inner_train,
                x_val=x_val,
                y_val=y_val,
                args=args,
                device=device,
                seed_offset=fold_index,
            )
        synchronize_device(device)
        search_time_sec = perf_counter() - search_start
        history_df = pd.DataFrame(history_rows)
        fold_histories.append(history_df)
        plot_training_history(
            history_df,
            f"MLP Training History - Fold {fold_index:02d} ({subject_id})",
            output_dir / f"training_history_{artifact_prefix}_fold_{fold_index:02d}_{subject_id}.png",
        )
        print(f"Selected best epoch from validation loss: {best_epoch}")

        final_train_start = perf_counter()
        if args.input_mode == "dual":
            print("\nBuilding full-train/test sequence datasets...")
            train_full_dataset = create_sequence_dataset(
                train_records,
                setup,
                args,
                fit_scaler=True,
                max_windows_per_file=args.max_train_windows_per_file,
            )
            test_dataset = create_sequence_dataset(
                test_records,
                setup,
                args,
                emg_scaler=train_full_dataset.emg_scaler,
                imu_scaler=train_full_dataset.imu_scaler,
                max_windows_per_file=args.max_test_windows_per_file,
            )
            summarize_epoch_windows("Full Train", train_full_dataset)
            summarize_epoch_windows("Test", test_dataset)
            synchronize_device(device)
            model = train_sequence_fixed_epochs(
                train_dataset=train_full_dataset,
                setup=setup,
                epochs=best_epoch,
                args=args,
                device=device,
                amp_enabled=amp_enabled,
                loader_kwargs=loader_kwargs,
                seed_offset=1000 + fold_index,
            )
            synchronize_device(device)
            final_train_time_sec = perf_counter() - final_train_start

            inference_start = perf_counter()
            test_loader = DataLoader(
                test_dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                **loader_kwargs,
            )
            _, _, y_pred, y_test = run_sequence_eval_epoch(
                model,
                test_loader,
                nn.CrossEntropyLoss(),
                device,
                amp_enabled=amp_enabled,
            )
            synchronize_device(device)
            inference_time_sec = perf_counter() - inference_start
            train_windows_used = len(train_full_dataset)
            test_windows_used = len(test_dataset)
            model_input_dim, _, _ = get_sequence_mlp_input_dim(setup, args)
            scaler_payload = {
                "emg_scaler": train_full_dataset.emg_scaler,
                "imu_scaler": train_full_dataset.imu_scaler,
            }
        else:
            full_scaler = fit_single_scaler(train_records, setup, args.max_train_windows_per_file, args)
            x_train_full, y_train_full = materialize_single(
                train_records,
                setup,
                args.max_train_windows_per_file,
                full_scaler,
                args,
            )
            x_test, y_test = materialize_single(
                test_records,
                setup,
                args.max_test_windows_per_file,
                full_scaler,
                args,
            )
            scaler_payload = {"scaler": full_scaler}
            print(f"Full-train matrix: {x_train_full.shape}")
            print(f"Test matrix: {x_test.shape}")
            model = train_fixed_epochs(
                x_train=x_train_full,
                y_train=y_train_full,
                epochs=best_epoch,
                args=args,
                device=device,
                seed_offset=1000 + fold_index,
            )
            final_train_time_sec = perf_counter() - final_train_start

            inference_start = perf_counter()
            y_pred = predict_in_batches(model, x_test, int(args.batch_size), device)
            inference_time_sec = perf_counter() - inference_start
            train_windows_used = int(x_train_full.shape[0])
            test_windows_used = int(x_test.shape[0])
            model_input_dim = int(x_train_full.shape[1])

        metrics = compute_metrics(y_test, y_pred)
        total_train_time_sec = search_time_sec + final_train_time_sec
        print(
            f"Fold accuracy: {metrics['accuracy'] * 100:.2f}% | "
            f"weighted F1: {metrics['f1_weighted'] * 100:.2f}% | "
            f"macro F1: {metrics['f1_macro'] * 100:.2f}%"
        )
        print(
            f"Search time: {search_time_sec:.2f}s | "
            f"Retrain time: {final_train_time_sec:.2f}s | "
            f"Inference time: {inference_time_sec:.4f}s"
        )

        fold_results.append(
            {
                "fold_index": fold_index,
                "subject_id": subject_id,
                "held_out_subjects": ",".join(held_out_subjects),
                "validation_subject_id": val_subject_id,
                "train_files": len(train_records),
                "test_files": len(test_records),
                "train_windows_used": int(train_windows_used),
                "test_windows_used": int(test_windows_used),
                "input_dim": int(model_input_dim),
                "best_epoch": int(best_epoch),
                "accuracy_pct": metrics["accuracy"] * 100.0,
                "f1_weighted_pct": metrics["f1_weighted"] * 100.0,
                "f1_macro_pct": metrics["f1_macro"] * 100.0,
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "train_time_sec": total_train_time_sec,
                "inference_time_sec": inference_time_sec,
            }
        )
        all_preds.extend(np.asarray(y_pred, dtype=np.int64).tolist())
        all_labels.extend(np.asarray(y_test, dtype=np.int64).tolist())

        plot_confusion(
            np.asarray(y_test, dtype=np.int64),
            np.asarray(y_pred, dtype=np.int64),
            f"MLP Confusion Matrix - Fold {fold_index:02d} ({subject_id})",
            output_dir / f"confusion_matrix_{artifact_prefix}_fold_{fold_index:02d}_{subject_id}.png",
        )

        if args.save_fold_models:
            model_dir = output_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "state_dict": model.state_dict(),
                "fold_index": fold_index,
                "subject_id": subject_id,
                "validation_subject_id": val_subject_id,
                "best_epoch": int(best_epoch),
                "input_mode": args.input_mode,
                "input_dim": int(model_input_dim),
                "hidden_dims": parse_hidden_dims(args.hidden_dims),
                "dropout": float(args.dropout),
                "label_names": LABEL_NAMES,
                "label_display_names": LABEL_DISPLAY_NAMES,
                "cv_mode": args.cv_mode,
                "emg_alignment": setup.get("emg_alignment"),
                **scaler_payload,
            }
            torch.save(payload, model_dir / f"{artifact_prefix}_fold_{fold_index:02d}_{subject_id}.pt")

    fold_df = pd.DataFrame(fold_results).sort_values(["fold_index", "subject_id"]).reset_index(drop=True)
    fold_df.to_csv(output_dir / f"mlp_{cv_tag}_fold_results.csv", index=False, encoding="utf-8-sig")

    y_true_all = np.asarray(all_labels, dtype=np.int64)
    y_pred_all = np.asarray(all_preds, dtype=np.int64)
    agg = compute_metrics(y_true_all, y_pred_all)
    summary = {
        "input_mode": args.input_mode,
        "cv_mode": args.cv_mode,
        "num_folds": int(args.num_folds) if args.cv_mode == "kfold" else len(cv_folds),
        "emg_alignment": setup.get("emg_alignment"),
        "mlp_pool_steps": int(args.mlp_pool_steps),
        "hidden_dims": parse_hidden_dims(args.hidden_dims),
        "folds_completed": int(len(fold_df)),
        "mean_accuracy_pct": float(fold_df["accuracy_pct"].mean()),
        "std_accuracy_pct": float(fold_df["accuracy_pct"].std(ddof=0)),
        "mean_f1_weighted_pct": float(fold_df["f1_weighted_pct"].mean()),
        "std_f1_weighted_pct": float(fold_df["f1_weighted_pct"].std(ddof=0)),
        "mean_f1_macro_pct": float(fold_df["f1_macro_pct"].mean()),
        "std_f1_macro_pct": float(fold_df["f1_macro_pct"].std(ddof=0)),
        "aggregate_accuracy_pct": float(agg["accuracy"] * 100.0),
        "aggregate_f1_weighted_pct": float(agg["f1_weighted"] * 100.0),
        "aggregate_f1_macro_pct": float(agg["f1_macro"] * 100.0),
        "mean_best_epoch": float(fold_df["best_epoch"].mean()),
        "mean_train_time_sec": float(fold_df["train_time_sec"].mean()),
        "mean_inference_time_sec": float(fold_df["inference_time_sec"].mean()),
    }
    (output_dir / f"mlp_{cv_tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"mlp_{cv_tag}_classification_report.txt").write_text(
        classification_report(
            y_true_all,
            y_pred_all,
            labels=list(range(len(LABEL_NAMES))),
            target_names=LABEL_DISPLAY_NAMES,
            digits=4,
            zero_division=0,
        ),
        encoding="utf-8",
    )

    plot_confusion(
        y_true_all,
        y_pred_all,
        f"MLP Aggregate Confusion Matrix - {cv_label}",
        output_dir / f"confusion_matrix_{artifact_prefix}.png",
    )
    plot_fold_accuracy(
        fold_df,
        f"MLP {cv_label} Accuracy by Held-out Group",
        output_dir / f"{artifact_prefix}_fold_accuracy.png",
    )
    plot_aggregate_history(fold_histories, output_dir / f"training_history_{artifact_prefix}.png")
    write_report(output_dir / f"mlp_{cv_tag}_report.md", args, summary, fold_df)

    print("\n" + "=" * 80)
    print(f"MLP {cv_label} complete")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Mean accuracy: {summary['mean_accuracy_pct']:.2f}% +/- {summary['std_accuracy_pct']:.2f}%")
    print(f"Mean weighted F1: {summary['mean_f1_weighted_pct']:.2f}% +/- {summary['std_f1_weighted_pct']:.2f}%")
    print(f"Mean selected epoch: {summary['mean_best_epoch']:.2f}")
    print("Leakage check: scaler(s) fit inside each fold, validation comes from training subjects only, and test subject is never used for early stopping.")


if __name__ == "__main__":
    main()
