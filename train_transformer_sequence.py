"""
Train a dual-branch Transformer classifier that fuses sEMG and IMU windows.

This keeps the same paired-data / early-stopping protocol as the dual-branch
Mamba script, using subject-grouped 5-fold cross-validation by default so the
comparison stays fair without mixing subjects across train/validation splits.
"""

import argparse
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from mamba_models import DualBranchTransformerSequenceClassifier
from train_mamba_emg_imu import (
    CLASS_TO_ID,
    LABEL_DISPLAY_NAMES,
    LABEL_NAMES,
    EarlyStopping,
    FileAwareBatchSampler,
    build_artifact_path,
    build_leave_one_subject_out_folds,
    build_subject_group_kfold_folds,
    describe_training_device,
    parse_feature_mask,
    plot_confusion_matrix,
    plot_training_history,
    resolve_training_device,
    set_seed,
    summarize_cv_results,
    summarize_epoch_windows,
    summarize_split,
)
from train_mamba_emg_imu_dual_branch import (
    PairedSequenceWindowDataset,
    benchmark_single_window_inference,
    collect_available_fold_results,
    count_model_parameters,
    evaluate,
    fold_artifacts_exist,
    load_fold_result_from_checkpoint,
    load_paired_dataset,
    resolve_pair_index_cache_path,
    select_loso_fold_specs,
    summarize_runtime_results,
    synchronize_device,
    train_epoch,
    write_runtime_metrics_csv,
)

set_seed(42, deterministic=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a dual-branch Transformer model with paired sEMG and IMU windows."
    )
    parser.add_argument("--emg-data-root", type=str, default=None, help="Path to the processed sEMG dataset root.")
    parser.add_argument("--imu-data-root", type=str, default=None, help="Path to the processed IMU dataset root.")
    parser.add_argument("--window-ms", type=float, default=500.0, help="Sliding window length in milliseconds.")
    parser.add_argument("--stride-ms", type=float, default=250.0, help="Sliding stride in milliseconds.")
    parser.add_argument("--emg-sample-rate-hz", type=float, default=None, help="sEMG sampling rate. Defaults to 1000 Hz.")
    parser.add_argument("--imu-sample-rate-hz", type=float, default=None, help="IMU sampling rate. Defaults to 200 Hz.")
    parser.add_argument("--emg-window-size", type=int, default=None, help="sEMG window size in samples.")
    parser.add_argument("--imu-window-size", type=int, default=None, help="IMU window size in samples.")
    parser.add_argument("--emg-stride", type=int, default=None, help="sEMG stride in samples.")
    parser.add_argument("--imu-stride", type=int, default=None, help="IMU stride in samples.")
    parser.add_argument("--emg-feature-mask", type=str, default=None, help='Optional sEMG channel mask, e.g. "1,1,0,...".')
    parser.add_argument("--imu-feature-mask", type=str, default=None, help='Optional IMU channel mask, e.g. "1,1,1,...".')
    parser.add_argument("--emg-downsample-mode", choices=("mean", "rms"), default="mean", help="sEMG temporal downsampling mode used before model input.")
    parser.add_argument("--no-align-emg-to-imu", action="store_true", help="Disable automatic sEMG downsampling to the IMU sequence length.")
    parser.add_argument("--emg-expected-channels", type=int, default=None, help="Optional expected sEMG channel count.")
    parser.add_argument("--imu-expected-channels", type=int, default=None, help="Optional expected IMU channel count.")
    parser.add_argument("--max-files-per-class", type=int, default=None, help="Limit paired files per class.")
    parser.add_argument("--stair-up-token", choices=("l", "r"), default="l", help="Filename token mapped to stair ascent.")
    parser.add_argument("--ramp-up-token", choices=("l", "r"), default="l", help="Filename token mapped to ramp ascent.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-epochs", type=int, default=20, help="Maximum number of epochs per fold.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--patience", type=int, choices=tuple(range(5, 9)), default=5, help="Early stopping patience on validation loss.")
    parser.add_argument("--device", type=str, default="cuda:0", help='Training device: "cuda:0", "cuda", "cpu", or "auto" (default: cuda:0).')
    parser.add_argument("--require-cuda", action="store_true", help="Exit instead of falling back to CPU when CUDA is unavailable.")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer hidden size per branch.")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of Transformer encoder layers per branch.")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Transformer feedforward size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout.")
    parser.add_argument("--fusion-hidden-dim", type=int, default=None, help="Hidden size of the fusion dense layer.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--cache-size", type=int, default=128, help="Dataset cache size.")
    parser.add_argument("--max-train-windows-per-file", type=int, default=64, help="Cap training windows per paired file.")
    parser.add_argument("--max-val-windows-per-file", type=int, default=None, help="Cap validation windows per paired file.")
    parser.add_argument("--benchmark-warmup", type=int, default=20, help="Warmup iterations for single-window inference timing.")
    parser.add_argument("--benchmark-repeat", type=int, default=200, help="Measured iterations for single-window inference timing.")
    parser.add_argument("--cv-mode", choices=("loso", "kfold"), default="kfold", help="Cross-validation mode: LOSO or subject-grouped K-fold.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of subject-grouped folds when --cv-mode kfold is used.")
    parser.add_argument("--fold-seed", type=int, default=42, help="Random seed for assigning subjects to K-fold groups.")
    parser.add_argument("--index-cache", type=str, default=None, help="Optional paired-file index cache path.")
    parser.add_argument("--rebuild-index-cache", action="store_true", help="Force rebuilding the paired-file index cache.")
    parser.add_argument("--output-prefix", type=str, default=None, help="Optional output prefix.")
    parser.add_argument("--start-subject", type=str, default=None, help="Optional validation subject/group id to start from.")
    parser.add_argument("--end-subject", type=str, default=None, help="Optional validation subject/group id to stop at.")
    parser.add_argument("--skip-existing-folds", action="store_true", help="Skip folds whose artifacts already exist.")
    return parser.parse_args()


def resolve_default_data_roots():
    from train_mamba_emg_imu import resolve_default_data_root

    return resolve_default_data_root("emg"), resolve_default_data_root("imu")


def resolve_sample_rates(emg_sample_rate_hz, imu_sample_rate_hz):
    from train_mamba_emg_imu import SENSOR_DEFAULT_SAMPLE_RATE, ms_to_samples

    return (
        float(emg_sample_rate_hz) if emg_sample_rate_hz is not None else SENSOR_DEFAULT_SAMPLE_RATE["emg"],
        float(imu_sample_rate_hz) if imu_sample_rate_hz is not None else SENSOR_DEFAULT_SAMPLE_RATE["imu"],
        ms_to_samples,
    )


def resolve_num_workers(requested_workers):
    from train_mamba_emg_imu import resolve_num_workers as _resolve_num_workers

    return _resolve_num_workers(requested_workers)


def resolve_emg_imu_alignment(
    emg_window_size,
    emg_stride,
    imu_window_size,
    imu_stride,
    align_emg_to_imu=True,
    downsample_mode="mean",
):
    if not align_emg_to_imu:
        return "none", 1, emg_window_size

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

    downsample_factor = emg_window_size // imu_window_size
    if downsample_factor < 1:
        raise ValueError("sEMG downsample factor must be at least 1.")

    if emg_stride % downsample_factor != 0 or emg_stride // downsample_factor != imu_stride:
        raise ValueError(
            "Automatic sEMG/IMU alignment requires the stride ratio to match the window "
            f"downsample factor. Got emg_stride={emg_stride}, imu_stride={imu_stride}, "
            f"downsample_factor={downsample_factor}."
        )

    if downsample_factor == 1:
        return "none", 1, imu_window_size
    return downsample_mode, downsample_factor, imu_window_size


def resolve_output_paths(explicit_prefix=None, cv_mode="kfold"):
    prefix = explicit_prefix if explicit_prefix else f"transformer_emg_imu_dual_{cv_mode}"
    if prefix.endswith(".pth"):
        prefix = prefix[:-4]
    checkpoint_path = f"best_model_{prefix}.pth"
    history_path = f"training_history_{prefix}.png"
    confusion_matrix_path = f"confusion_matrix_{prefix}.png"
    return checkpoint_path, history_path, confusion_matrix_path


def main():
    args = parse_args()
    default_emg_root, default_imu_root = resolve_default_data_roots()
    emg_data_root = args.emg_data_root if args.emg_data_root else default_emg_root
    imu_data_root = args.imu_data_root if args.imu_data_root else default_imu_root
    emg_sample_rate_hz, imu_sample_rate_hz, ms_to_samples_fn = resolve_sample_rates(
        args.emg_sample_rate_hz,
        args.imu_sample_rate_hz,
    )
    emg_window_size = int(args.emg_window_size) if args.emg_window_size is not None else ms_to_samples_fn(args.window_ms, emg_sample_rate_hz)
    imu_window_size = int(args.imu_window_size) if args.imu_window_size is not None else ms_to_samples_fn(args.window_ms, imu_sample_rate_hz)
    emg_stride = int(args.emg_stride) if args.emg_stride is not None else ms_to_samples_fn(args.stride_ms, emg_sample_rate_hz)
    imu_stride = int(args.imu_stride) if args.imu_stride is not None else ms_to_samples_fn(args.stride_ms, imu_sample_rate_hz)
    (
        emg_temporal_pooling,
        emg_temporal_pool_size,
        emg_model_window_size,
    ) = resolve_emg_imu_alignment(
        emg_window_size=emg_window_size,
        emg_stride=emg_stride,
        imu_window_size=imu_window_size,
        imu_stride=imu_stride,
        align_emg_to_imu=not args.no_align_emg_to_imu,
        downsample_mode=args.emg_downsample_mode,
    )
    resolved_num_workers = resolve_num_workers(args.num_workers)
    checkpoint_path, history_path, confusion_matrix_path = resolve_output_paths(
        args.output_prefix,
        cv_mode=args.cv_mode,
    )

    config = {
        "emg_data_root": emg_data_root,
        "imu_data_root": imu_data_root,
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "emg_sample_rate_hz": emg_sample_rate_hz,
        "imu_sample_rate_hz": imu_sample_rate_hz,
        "emg_window_size": emg_window_size,
        "emg_model_window_size": emg_model_window_size,
        "imu_window_size": imu_window_size,
        "emg_stride": emg_stride,
        "imu_stride": imu_stride,
        "align_emg_to_imu": not args.no_align_emg_to_imu,
        "emg_temporal_pooling": emg_temporal_pooling,
        "emg_temporal_pool_size": emg_temporal_pool_size,
        "num_classes": len(CLASS_TO_ID),
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "device": args.device,
        "require_cuda": args.require_cuda,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "n_layers": args.n_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "max_files_per_class": args.max_files_per_class,
        "max_train_windows_per_file": args.max_train_windows_per_file,
        "max_val_windows_per_file": args.max_val_windows_per_file,
        "emg_feature_mask": args.emg_feature_mask,
        "imu_feature_mask": args.imu_feature_mask,
        "emg_expected_channels": args.emg_expected_channels,
        "imu_expected_channels": args.imu_expected_channels,
        "stair_up_token": args.stair_up_token,
        "ramp_up_token": args.ramp_up_token,
        "num_workers": resolved_num_workers,
        "cache_size": args.cache_size,
        "benchmark_warmup": args.benchmark_warmup,
        "benchmark_repeat": args.benchmark_repeat,
        "cv_mode": args.cv_mode,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "confusion_matrix_path": confusion_matrix_path,
        "index_cache": str(
            resolve_pair_index_cache_path(
                emg_root=emg_data_root,
                imu_root=imu_data_root,
                emg_window_size=emg_window_size,
                emg_stride=emg_stride,
                imu_window_size=imu_window_size,
                imu_stride=imu_stride,
                max_files_per_class=args.max_files_per_class,
                explicit_path=args.index_cache,
            )
        ),
    }

    print("=" * 60)
    print("Dual-branch Transformer training on paired sEMG + IMU data")
    print("=" * 60)

    device = resolve_training_device(args.device, args.require_cuda)
    amp_enabled = device.type == "cuda"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print(f"\nUsing device: {describe_training_device(device)}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"DataLoader workers: {config['num_workers']}")
    print(f"EMG root: {config['emg_data_root']}")
    print(f"IMU root: {config['imu_data_root']}")
    print(
        f"Direction mapping: stair {config['stair_up_token']}=up, "
        f"ramp {config['ramp_up_token']}=up"
    )
    print(
        f"EMG window: {config['window_ms']} ms / {config['stride_ms']} ms "
        f"-> {config['emg_window_size']} / {config['emg_stride']} samples "
        f"(sample rate: {config['emg_sample_rate_hz']} Hz)"
    )
    if config["emg_temporal_pooling"] != "none":
        print(
            f"EMG -> IMU alignment: {config['emg_temporal_pooling']} downsample "
            f"x{config['emg_temporal_pool_size']} -> model sequence length "
            f"{config['emg_model_window_size']}"
        )
    else:
        print(f"EMG -> IMU alignment: none -> model sequence length {config['emg_model_window_size']}")
    print(
        f"IMU window: {config['window_ms']} ms / {config['stride_ms']} ms "
        f"-> {config['imu_window_size']} / {config['imu_stride']} samples "
        f"(sample rate: {config['imu_sample_rate_hz']} Hz)"
    )
    print(f"Early stopping patience: {config['patience']} (monitor: val_loss)")
    print(f"Checkpoint path: {config['checkpoint_path']}")

    (
        file_records,
        detected_emg_channel_names,
        detected_imu_channel_names,
        subject_ids,
    ) = load_paired_dataset(
        emg_root=config["emg_data_root"],
        imu_root=config["imu_data_root"],
        emg_window_size=config["emg_window_size"],
        emg_stride=config["emg_stride"],
        imu_window_size=config["imu_window_size"],
        imu_stride=config["imu_stride"],
        max_files_per_class=config["max_files_per_class"],
        emg_expected_channels=config["emg_expected_channels"],
        imu_expected_channels=config["imu_expected_channels"],
        cache_path=config["index_cache"],
        rebuild_index_cache=args.rebuild_index_cache,
        stair_up_token=config["stair_up_token"],
        ramp_up_token=config["ramp_up_token"],
    )

    original_emg_channels = len(detected_emg_channel_names)
    original_imu_channels = len(detected_imu_channel_names)
    emg_feature_mask = parse_feature_mask(config["emg_feature_mask"], original_emg_channels)
    imu_feature_mask = parse_feature_mask(config["imu_feature_mask"], original_imu_channels)

    selected_emg_feature_indices = list(range(original_emg_channels))
    selected_imu_feature_indices = list(range(original_imu_channels))
    if emg_feature_mask is not None:
        selected_emg_feature_indices = np.where(emg_feature_mask == 1)[0].tolist()
        print(
            f"\nEMG feature mask applied: keep {len(selected_emg_feature_indices)}/"
            f"{original_emg_channels} channels"
        )
        print(f"Selected EMG channel indices: {selected_emg_feature_indices}")
    else:
        print(f"\nNo EMG feature mask specified, using all {original_emg_channels} channels")

    if imu_feature_mask is not None:
        selected_imu_feature_indices = np.where(imu_feature_mask == 1)[0].tolist()
        print(
            f"IMU feature mask applied: keep {len(selected_imu_feature_indices)}/"
            f"{original_imu_channels} channels"
        )
        print(f"Selected IMU channel indices: {selected_imu_feature_indices}")
    else:
        print(f"No IMU feature mask specified, using all {original_imu_channels} channels")

    print(f"\nEMG channel names: {', '.join(detected_emg_channel_names)}")
    print(f"IMU channel names: {', '.join(detected_imu_channel_names)}")
    print(f"All detected subjects: {len(subject_ids)}")
    print("Classes: " + ", ".join(f"{idx}={name}" for idx, name in enumerate(LABEL_DISPLAY_NAMES)))

    loader_kwargs = {
        "num_workers": config["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    if config["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    emg_input_channels = len(selected_emg_feature_indices)
    imu_input_channels = len(selected_imu_feature_indices)
    print(f"EMG input shape: (batch, {config['emg_model_window_size']}, {emg_input_channels})")
    print(f"IMU input shape: (batch, {config['imu_window_size']}, {imu_input_channels})")

    if config["cv_mode"] == "loso":
        cv_folds = build_leave_one_subject_out_folds(file_records)
        cv_label = "LOSO"
        print(f"\nLeave-one-subject-out folds: {len(cv_folds)}")
        if not cv_folds:
            raise ValueError(
                "Leave-one-subject-out cross-validation requires records from at least two subjects."
            )
    else:
        cv_folds = build_subject_group_kfold_folds(
            file_records,
            num_folds=config["num_folds"],
            random_state=config["fold_seed"],
        )
        cv_label = f"{config['num_folds']}-fold subject-grouped CV"
        print(f"\nSubject-grouped K-fold folds: {len(cv_folds)}")
        for fold_index, (fold_id, _, val_records) in enumerate(cv_folds, start=1):
            val_subjects = sorted({record.subject_id for record in val_records})
            print(f"  Fold {fold_index:02d} ({fold_id}): validation subjects {', '.join(val_subjects)}")
    if not cv_folds:
        raise ValueError(
            "Cross-validation requires records from at least two subject groups."
        )

    selected_fold_specs = select_loso_fold_specs(
        cv_folds,
        start_subject=args.start_subject,
        end_subject=args.end_subject,
    )
    if not selected_fold_specs:
        raise ValueError("No CV folds selected after applying the requested fold range.")

    print(
        "Selected folds: "
        + ", ".join(
            f"{fold_index:02d}:{subject_id}" for fold_index, subject_id, _, _ in selected_fold_specs
        )
    )

    print("\n" + "=" * 60)
    print(f"Start {cv_label} cross-validation...")
    print("=" * 60)

    all_cv_preds = []
    all_cv_labels = []
    fold_results = []
    best_overall_fold = None

    for fold_index, val_subject_id, train_records, val_records in selected_fold_specs:
        print("\n" + "=" * 60)
        val_subjects = sorted({record.subject_id for record in val_records})
        print(f"Fold {fold_index}/{len(cv_folds)} - Validation group: {val_subject_id}")
        print(f"Validation subjects: {', '.join(val_subjects)}")
        print("=" * 60)

        fold_checkpoint_path = build_artifact_path(
            config["checkpoint_path"],
            f"fold_{fold_index:02d}_{val_subject_id}",
        )
        fold_history_path = build_artifact_path(
            config["history_path"],
            f"fold_{fold_index:02d}_{val_subject_id}",
        )
        fold_confusion_path = build_artifact_path(
            config["confusion_matrix_path"],
            f"fold_{fold_index:02d}_{val_subject_id}",
        )

        if args.skip_existing_folds and fold_artifacts_exist(
            fold_checkpoint_path,
            fold_history_path,
            fold_confusion_path,
        ):
            print("Fold artifacts already exist. Skipping this fold.")
            existing_result = load_fold_result_from_checkpoint(
                fold_checkpoint_path,
                fold_index=fold_index,
            )
            fold_results.append(existing_result)
            if best_overall_fold is None or existing_result["val_loss"] < best_overall_fold["val_loss"]:
                best_overall_fold = existing_result
            continue

        summarize_split("Train", train_records)
        summarize_split("Val", val_records)

        train_dataset = PairedSequenceWindowDataset(
            train_records,
            emg_window_size=config["emg_window_size"],
            emg_stride=config["emg_stride"],
            imu_window_size=config["imu_window_size"],
            imu_stride=config["imu_stride"],
            fit_scaler=True,
            emg_feature_mask=emg_feature_mask,
            imu_feature_mask=imu_feature_mask,
            cache_size=config["cache_size"],
            max_windows_per_file=config["max_train_windows_per_file"],
            emg_temporal_pooling=config["emg_temporal_pooling"],
            emg_temporal_pool_size=config["emg_temporal_pool_size"],
        )
        val_dataset = PairedSequenceWindowDataset(
            val_records,
            emg_window_size=config["emg_window_size"],
            emg_stride=config["emg_stride"],
            imu_window_size=config["imu_window_size"],
            imu_stride=config["imu_stride"],
            emg_scaler=train_dataset.emg_scaler,
            imu_scaler=train_dataset.imu_scaler,
            emg_feature_mask=emg_feature_mask,
            imu_feature_mask=imu_feature_mask,
            cache_size=config["cache_size"],
            max_windows_per_file=config["max_val_windows_per_file"],
            emg_temporal_pooling=config["emg_temporal_pooling"],
            emg_temporal_pool_size=config["emg_temporal_pool_size"],
        )

        summarize_epoch_windows("Train", train_dataset)
        summarize_epoch_windows("Val", val_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=FileAwareBatchSampler(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle_files=True,
            ),
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            **loader_kwargs,
        )

        model = DualBranchTransformerSequenceClassifier(
            emg_input_channels=emg_input_channels,
            imu_input_channels=imu_input_channels,
            num_classes=config["num_classes"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["n_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            fusion_hidden_dim=config["fusion_hidden_dim"],
        ).to(device)
        model_parameters, trainable_parameters = count_model_parameters(model)
        print(f"Model parameters: {model_parameters:,} (trainable: {trainable_parameters:,})")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"],
        )
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        early_stopping = EarlyStopping(
            patience=config["patience"],
            checkpoint_path=fold_checkpoint_path,
        )

        synchronize_device(device)
        fold_train_start = perf_counter()
        for epoch in range(config["num_epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                amp_enabled=amp_enabled,
                grad_scaler=grad_scaler,
            )
            val_loss, val_acc, _, _ = evaluate(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
            )
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            improved, should_stop = early_stopping.step(
                val_loss,
                epoch,
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "fold_subject_id": val_subject_id,
                    "config": config,
                    "emg_scaler": train_dataset.emg_scaler,
                    "imu_scaler": train_dataset.imu_scaler,
                    "emg_feature_mask": emg_feature_mask.tolist() if emg_feature_mask is not None else None,
                    "imu_feature_mask": imu_feature_mask.tolist() if imu_feature_mask is not None else None,
                    "selected_emg_feature_indices": selected_emg_feature_indices,
                    "selected_imu_feature_indices": selected_imu_feature_indices,
                    "original_emg_channels": original_emg_channels,
                    "original_imu_channels": original_imu_channels,
                    "emg_channel_names": detected_emg_channel_names,
                    "imu_channel_names": detected_imu_channel_names,
                    "label_names": LABEL_NAMES,
                    "label_display_names": LABEL_DISPLAY_NAMES,
                    "model_family": "DualBranchTransformer",
                },
            )
            if improved:
                print(
                    f"Saved best fold model (val loss: {val_loss:.4f}, "
                    f"val acc: {val_acc:.2f}%)"
                )
            if should_stop:
                print(
                    f"\nEarly stopping triggered! Val loss did not improve for "
                    f"{config['patience']} epochs"
                )
                break
        synchronize_device(device)
        train_time_sec = perf_counter() - fold_train_start

        best_checkpoint = early_stopping.restore_best_weights(model, map_location=device)
        inference_benchmark = benchmark_single_window_inference(
            model,
            emg_window_size=config["emg_model_window_size"],
            emg_input_channels=emg_input_channels,
            imu_window_size=config["imu_window_size"],
            imu_input_channels=imu_input_channels,
            device=device,
            amp_enabled=amp_enabled,
            warmup=config["benchmark_warmup"],
            repeat=config["benchmark_repeat"],
        )
        best_val_loss, best_val_acc, val_preds, val_labels = evaluate(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
        )

        plot_training_history(history, save_path=fold_history_path)
        plot_confusion_matrix(
            val_labels,
            val_preds,
            LABEL_DISPLAY_NAMES,
            save_path=fold_confusion_path,
        )

        fold_result = {
            "fold_index": fold_index,
            "subject_id": val_subject_id,
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "best_epoch": int(best_checkpoint["epoch"]),
            "checkpoint_path": fold_checkpoint_path,
            "model_parameters": model_parameters,
            "trainable_parameters": trainable_parameters,
            "train_time_sec": train_time_sec,
            "single_inference_ms": inference_benchmark["single_inference_ms"],
            "inference_benchmark": inference_benchmark,
            "history": history,
        }
        best_checkpoint.update(
            {
                "model_parameters": model_parameters,
                "trainable_parameters": trainable_parameters,
                "train_time_sec": train_time_sec,
                "single_inference_ms": inference_benchmark["single_inference_ms"],
                "inference_benchmark": inference_benchmark,
                "model_family": "DualBranchTransformer",
            }
        )
        torch.save(best_checkpoint, fold_checkpoint_path)
        fold_results.append(fold_result)
        all_cv_preds.extend(val_preds)
        all_cv_labels.extend(val_labels)

        print("\nFold result:")
        print(f"  validation group: {val_subject_id}")
        print(f"  best epoch: {fold_result['best_epoch'] + 1}")
        print(f"  best val loss: {best_val_loss:.4f}")
        print(f"  best val acc: {best_val_acc:.2f}%")
        print(f"  train time: {train_time_sec:.2f} s")
        print(f"  single-window inference: {inference_benchmark['single_inference_ms']:.4f} ms")
        print(f"  checkpoint: {fold_checkpoint_path}")

        runtime_metrics_path = write_runtime_metrics_csv(fold_results, config["checkpoint_path"])
        print(f"  runtime metrics: {runtime_metrics_path}")

        if best_overall_fold is None or fold_result["val_loss"] < best_overall_fold["val_loss"]:
            best_overall_fold = fold_result

    summarize_cv_results(fold_results)
    summarize_runtime_results(fold_results)
    runtime_metrics_path = write_runtime_metrics_csv(fold_results, config["checkpoint_path"])
    print(f"Runtime metrics saved to: {runtime_metrics_path}")

    if all_cv_labels and all_cv_preds:
        print(f"\nOverall {cv_label} classification report:")
        print(
            classification_report(
                all_cv_labels,
                all_cv_preds,
                labels=list(range(len(LABEL_NAMES))),
                target_names=LABEL_DISPLAY_NAMES,
                digits=4,
                zero_division=0,
            )
        )

        plot_confusion_matrix(
            all_cv_labels,
            all_cv_preds,
            LABEL_DISPLAY_NAMES,
            save_path=config["confusion_matrix_path"],
        )

        if best_overall_fold is not None and best_overall_fold["history"] is not None:
            plot_training_history(best_overall_fold["history"], save_path=config["history_path"])
            best_overall_checkpoint = torch.load(
                best_overall_fold["checkpoint_path"],
                weights_only=False,
            )
            torch.save(best_overall_checkpoint, config["checkpoint_path"])

    available_fold_results = collect_available_fold_results(
        config["checkpoint_path"],
        [
            (fold_index, subject_id, train_records, val_records)
            for fold_index, (subject_id, train_records, val_records) in enumerate(cv_folds, start=1)
        ],
    )
    if available_fold_results:
        print("\nAvailable checkpoint summary across all completed folds:")
        summarize_cv_results(available_fold_results)
        summarize_runtime_results(available_fold_results)
        print(f"Completed folds available on disk: {len(available_fold_results)} / {len(cv_folds)}")
        if len(available_fold_results) == len(cv_folds):
            best_available_fold = min(available_fold_results, key=lambda item: item["val_loss"])
            best_available_checkpoint = torch.load(
                best_available_fold["checkpoint_path"],
                weights_only=False,
            )
            torch.save(best_available_checkpoint, config["checkpoint_path"])
            print(
                "All fold checkpoints are now available. "
                f"Best checkpoint copied to: {config['checkpoint_path']}"
            )

    print("\n" + "=" * 60)
    print("Cross-validation complete!")
    if best_overall_fold is not None:
        print(
            f"Best fold: {best_overall_fold['fold_index']} "
            f"(validation group {best_overall_fold['subject_id']})"
        )
        print(f"Best fold val loss: {best_overall_fold['val_loss']:.4f}")
        print(f"Best fold val acc: {best_overall_fold['val_acc']:.2f}%")
    if all_cv_labels and all_cv_preds:
        print(f"Aggregate confusion matrix saved to: {config['confusion_matrix_path']}")
        print(f"Best overall fold checkpoint copied to: {config['checkpoint_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
