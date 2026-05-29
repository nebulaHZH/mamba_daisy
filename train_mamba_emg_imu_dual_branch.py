"""
Train a dual-branch Mamba classifier that fuses sEMG and IMU windows.

This script keeps the training style from train_mamba_emg_imu.py:
- early stopping on validation loss
- leave-one-subject-out cross-validation
- file-backed sliding-window loading

The difference is the model/data path:
- one Mamba branch encodes sEMG windows
- one Mamba branch encodes IMU windows
- their pooled feature vectors are concatenated before the dense classifier
"""

import argparse
import csv
import warnings
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mamba_models import DualBranchMambaSequenceClassifier
from train_mamba_emg_imu import (
    CLASS_TO_ID,
    INDEX_CACHE_VERSION,
    LABEL_DISPLAY_NAMES,
    LABEL_NAMES,
    SENSOR_DEFAULT_SAMPLE_RATE,
    EarlyStopping,
    FileAwareBatchSampler,
    apply_feature_mask,
    build_artifact_path,
    build_leave_one_subject_out_folds,
    extract_label_from_path,
    extract_subject_id,
    load_index_cache,
    load_mat_file,
    ms_to_samples,
    parse_feature_mask,
    plot_confusion_matrix,
    plot_training_history,
    describe_training_device,
    resolve_default_data_root,
    resolve_num_workers,
    resolve_temporal_pooling,
    resolve_training_device,
    save_index_cache,
    set_seed,
    summarize_cv_results,
    summarize_epoch_windows,
    summarize_split,
)

warnings.filterwarnings("ignore")
set_seed(42, deterministic=False)

PAIR_INDEX_CACHE_VERSION = INDEX_CACHE_VERSION + 100


@dataclass(frozen=True)
class PairedFileRecord:
    pair_key: str
    emg_file_path: str
    imu_file_path: str
    label_id: int
    label_name: str
    subject_id: str
    emg_num_steps: int
    imu_num_steps: int
    emg_num_windows: int
    imu_num_windows: int
    num_windows: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a dual-branch Mamba model with paired sEMG and IMU windows."
    )
    parser.add_argument("--emg-data-root", type=str, default=None, help="Path to the processed sEMG dataset root.")
    parser.add_argument("--imu-data-root", type=str, default=None, help="Path to the processed IMU dataset root.")
    parser.add_argument(
        "--window-ms",
        type=float,
        default=500.0,
        help="Sliding window length in milliseconds shared by both modalities.",
    )
    parser.add_argument(
        "--stride-ms",
        type=float,
        default=250.0,
        help="Sliding stride in milliseconds shared by both modalities.",
    )
    parser.add_argument("--emg-sample-rate-hz", type=float, default=None, help="sEMG sampling rate. Defaults to 1000 Hz.")
    parser.add_argument("--imu-sample-rate-hz", type=float, default=None, help="IMU sampling rate. Defaults to 200 Hz.")
    parser.add_argument("--emg-window-size", type=int, default=None, help="sEMG window size in samples. Overrides --window-ms.")
    parser.add_argument("--imu-window-size", type=int, default=None, help="IMU window size in samples. Overrides --window-ms.")
    parser.add_argument("--emg-stride", type=int, default=None, help="sEMG stride in samples. Overrides --stride-ms.")
    parser.add_argument("--imu-stride", type=int, default=None, help="IMU stride in samples. Overrides --stride-ms.")
    parser.add_argument("--emg-feature-mask", type=str, default=None, help='Optional sEMG channel mask, e.g. "1,1,0,1,...".')
    parser.add_argument("--imu-feature-mask", type=str, default=None, help='Optional IMU channel mask, e.g. "1,1,1,0,...".')
    parser.add_argument("--emg-temporal-pooling", choices=("none", "mean", "rms"), default="none", help="Optional sEMG temporal pooling applied to each window before scaling/model input.")
    parser.add_argument("--emg-temporal-pool-size", type=int, default=1, help="Number of consecutive sEMG time points pooled into one step.")
    parser.add_argument("--emg-expected-channels", type=int, default=None, help="Optional expected sEMG channel count.")
    parser.add_argument("--imu-expected-channels", type=int, default=None, help="Optional expected IMU channel count.")
    parser.add_argument("--max-files-per-class", type=int, default=None, help="Limit paired files per class for quick experiments.")
    parser.add_argument("--stair-up-token", choices=("l", "r"), default="l", help="Filename token mapped to stair ascent.")
    parser.add_argument("--ramp-up-token", choices=("l", "r"), default="l", help="Filename token mapped to ramp ascent.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--num-epochs", type=int, default=20, help="Maximum number of epochs per LOSO fold.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--patience", type=int, choices=tuple(range(5, 9)), default=5, help="Early stopping patience on validation loss.")
    parser.add_argument("--device", type=str, default="auto", help='Training device: "auto", "cpu", "cuda", or "cuda:0" (default: auto).')
    parser.add_argument("--require-cuda", action="store_true", help="Exit instead of falling back to CPU when CUDA is unavailable.")
    parser.add_argument("--d-model", type=int, default=64, help="Hidden size inside each Mamba branch.")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of Mamba residual blocks in each branch.")
    parser.add_argument("--d-state", type=int, default=16, help="Mamba state dimension.")
    parser.add_argument("--d-conv", type=int, default=4, help="Depthwise convolution kernel size in Mamba blocks.")
    parser.add_argument("--expand", type=int, default=2, help="Expansion ratio inside each Mamba block.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout used in both branches and the fusion head.")
    parser.add_argument("--fusion-hidden-dim", type=int, default=None, help="Hidden size of the dense fusion layer. Defaults to d-model.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count. Defaults to 0 on Windows and 2 on other platforms.")
    parser.add_argument("--cache-size", type=int, default=128, help="Number of paired files to keep in the dataset cache.")
    parser.add_argument("--max-train-windows-per-file", type=int, default=64, help="Optional cap on training windows per paired file.")
    parser.add_argument("--max-val-windows-per-file", type=int, default=None, help="Optional cap on validation windows per paired file.")
    parser.add_argument("--benchmark-warmup", type=int, default=20, help="Warmup iterations for single-window inference timing.")
    parser.add_argument("--benchmark-repeat", type=int, default=200, help="Measured iterations for single-window inference timing.")
    parser.add_argument("--cv-mode", choices=("loso", "kfold"), default="loso", help="Cross-validation mode: LOSO or subject-grouped K-fold.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of subject-grouped folds when --cv-mode kfold is used.")
    parser.add_argument("--fold-seed", type=int, default=42, help="Random seed for assigning subjects to K-fold groups.")
    parser.add_argument("--index-cache", type=str, default=None, help="Optional paired-file index cache path.")
    parser.add_argument("--rebuild-index-cache", action="store_true", help="Force rebuilding the paired-file index cache.")
    parser.add_argument("--output-prefix", type=str, default=None, help="Optional output prefix for checkpoints and plots.")
    parser.add_argument(
        "--start-subject",
        type=str,
        default=None,
        help="Optional LOSO validation subject to start from, e.g. AB20.",
    )
    parser.add_argument(
        "--end-subject",
        type=str,
        default=None,
        help="Optional LOSO validation subject to stop at, e.g. AB30.",
    )
    parser.add_argument(
        "--skip-existing-folds",
        action="store_true",
        help="Skip folds whose checkpoint, history, and confusion-matrix artifacts already exist.",
    )
    return parser.parse_args()


def count_windows(num_steps, window_size, stride):
    if num_steps < window_size:
        return 0
    return 1 + (num_steps - window_size) // stride


def build_pair_key(file_path, root_path, sensor_folder):
    rel_path = Path(file_path).resolve().relative_to(Path(root_path).resolve())
    parts = list(rel_path.parts)
    if len(parts) < 2 or parts[-2].lower() != sensor_folder.lower():
        raise ValueError(f"Unexpected paired file layout: {file_path}")
    return "/".join(parts[:-2] + [parts[-1]])


def build_sensor_file_map(root_path, sensor_folder):
    root_path = Path(root_path)
    mat_files = sorted(root_path.glob(f"**/{sensor_folder}/*.mat"))
    if not mat_files:
        raise ValueError(f"No MAT files found under '{root_path}' for sensor folder '{sensor_folder}'.")

    file_map = {}
    for file_path in mat_files:
        pair_key = build_pair_key(file_path, root_path, sensor_folder)
        file_map[pair_key] = str(file_path)
    return file_map


def build_paired_candidate_files(
    emg_root,
    imu_root,
    max_files_per_class=None,
    stair_up_token="l",
    ramp_up_token="l",
):
    print("Scanning paired EMG/IMU MAT files...")
    emg_file_map = build_sensor_file_map(emg_root, sensor_folder="emg")
    imu_file_map = build_sensor_file_map(imu_root, sensor_folder="imu")

    emg_keys = set(emg_file_map)
    imu_keys = set(imu_file_map)
    common_keys = sorted(emg_keys & imu_keys)
    emg_only_keys = sorted(emg_keys - imu_keys)
    imu_only_keys = sorted(imu_keys - emg_keys)

    if not common_keys:
        raise ValueError("No paired EMG/IMU files were found. Please confirm both processed dataset roots.")

    print(f"Found {len(emg_file_map)} EMG files")
    print(f"Found {len(imu_file_map)} IMU files")
    print(f"Paired files available: {len(common_keys)}")
    if emg_only_keys:
        print(f"EMG-only files skipped: {len(emg_only_keys)}")
    if imu_only_keys:
        print(f"IMU-only files skipped: {len(imu_only_keys)}")

    class_pairs = {label_name: [] for label_name in CLASS_TO_ID}
    unmatched_label_pairs = []
    for pair_key in common_keys:
        emg_path = emg_file_map[pair_key]
        imu_path = imu_file_map[pair_key]
        label_id, label_name = extract_label_from_path(
            emg_path,
            stair_up_token=stair_up_token,
            ramp_up_token=ramp_up_token,
        )
        if label_name is None:
            unmatched_label_pairs.append(pair_key)
            continue

        class_pairs[label_name].append(
            {
                "pair_key": pair_key,
                "emg_file_path": emg_path,
                "imu_file_path": imu_path,
                "label_id": label_id,
                "label_name": label_name,
            }
        )

    if unmatched_label_pairs:
        print(f"Pairs skipped because no class label was matched: {len(unmatched_label_pairs)}")

    selected_pairs = []
    print("\nPaired files per class:")
    for label_name in LABEL_NAMES:
        pairs = sorted(class_pairs[label_name], key=lambda item: item["pair_key"])
        print(f"  {label_name}: {len(pairs)} pairs")
        if max_files_per_class is not None:
            pairs = pairs[:max_files_per_class]
        selected_pairs.extend(pairs)

    return selected_pairs, emg_only_keys, imu_only_keys


def get_paired_file_signature(selected_pairs):
    signature = []
    for pair in selected_pairs:
        emg_path = Path(pair["emg_file_path"])
        imu_path = Path(pair["imu_file_path"])
        emg_stat = emg_path.stat()
        imu_stat = imu_path.stat()
        signature.append(
            {
                "pair_key": pair["pair_key"],
                "emg_file_path": str(emg_path),
                "emg_size": emg_stat.st_size,
                "emg_mtime_ns": emg_stat.st_mtime_ns,
                "imu_file_path": str(imu_path),
                "imu_size": imu_stat.st_size,
                "imu_mtime_ns": imu_stat.st_mtime_ns,
            }
        )
    return signature


def resolve_pair_index_cache_path(
    emg_root,
    imu_root,
    emg_window_size,
    emg_stride,
    imu_window_size,
    imu_stride,
    max_files_per_class,
    explicit_path=None,
):
    if explicit_path:
        return Path(explicit_path)

    cache_dir = Path.cwd() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    emg_tag = Path(emg_root).name or "emg"
    imu_tag = Path(imu_root).name or "imu"
    max_files_tag = "all" if max_files_per_class is None else f"mf{max_files_per_class}"
    file_name = (
        f"{emg_tag}_{imu_tag}_dual_ws{emg_window_size}_{imu_window_size}"
        f"_st{emg_stride}_{imu_stride}_{max_files_tag}_index.json"
    )
    return cache_dir / file_name


def rebuild_paired_file_records(
    selected_pairs,
    emg_window_size,
    emg_stride,
    imu_window_size,
    imu_stride,
    emg_expected_channels=None,
    imu_expected_channels=None,
):
    records = []
    load_errors = []
    emg_channel_names = None
    imu_channel_names = None
    subject_ids = set()

    print("\nInspecting paired files and building window metadata...")
    for pair in tqdm(selected_pairs, desc="Indexing paired files"):
        emg_path = pair["emg_file_path"]
        imu_path = pair["imu_file_path"]

        emg_signal, current_emg_channel_names, emg_err = load_mat_file(emg_path)
        imu_signal, current_imu_channel_names, imu_err = load_mat_file(imu_path)
        if emg_err is not None or imu_err is not None:
            error_parts = []
            if emg_err is not None:
                error_parts.append(f"EMG: {emg_err}")
            if imu_err is not None:
                error_parts.append(f"IMU: {imu_err}")
            load_errors.append((pair["pair_key"], " | ".join(error_parts)))
            continue

        if emg_channel_names is None:
            emg_channel_names = current_emg_channel_names
            print(
                f"Detected {len(emg_channel_names)} sEMG channels: "
                f"{', '.join(emg_channel_names[:6])}{' ...' if len(emg_channel_names) > 6 else ''}"
            )
        elif len(current_emg_channel_names) != len(emg_channel_names):
            load_errors.append(
                (
                    pair["pair_key"],
                    f"Inconsistent EMG channel count {len(current_emg_channel_names)} "
                    f"(expected {len(emg_channel_names)})",
                )
            )
            continue

        if imu_channel_names is None:
            imu_channel_names = current_imu_channel_names
            print(
                f"Detected {len(imu_channel_names)} IMU channels: "
                f"{', '.join(imu_channel_names[:6])}{' ...' if len(imu_channel_names) > 6 else ''}"
            )
        elif len(current_imu_channel_names) != len(imu_channel_names):
            load_errors.append(
                (
                    pair["pair_key"],
                    f"Inconsistent IMU channel count {len(current_imu_channel_names)} "
                    f"(expected {len(imu_channel_names)})",
                )
            )
            continue

        if emg_expected_channels is not None and emg_signal.shape[1] != emg_expected_channels:
            raise ValueError(
                f"Loaded EMG data has {emg_signal.shape[1]} channels, "
                f"but --emg-expected-channels={emg_expected_channels}."
            )
        if imu_expected_channels is not None and imu_signal.shape[1] != imu_expected_channels:
            raise ValueError(
                f"Loaded IMU data has {imu_signal.shape[1]} channels, "
                f"but --imu-expected-channels={imu_expected_channels}."
            )

        subject_id = extract_subject_id(emg_path)
        imu_subject_id = extract_subject_id(imu_path)
        if imu_subject_id != subject_id:
            load_errors.append(
                (
                    pair["pair_key"],
                    f"Subject mismatch between EMG ({subject_id}) and IMU ({imu_subject_id})",
                )
            )
            continue
        subject_ids.add(subject_id)

        emg_num_windows = count_windows(len(emg_signal), emg_window_size, emg_stride)
        imu_num_windows = count_windows(len(imu_signal), imu_window_size, imu_stride)
        paired_num_windows = min(emg_num_windows, imu_num_windows)
        if paired_num_windows == 0:
            load_errors.append(
                (
                    pair["pair_key"],
                    "Signal too short for at least one modality "
                    f"(emg_windows={emg_num_windows}, imu_windows={imu_num_windows})",
                )
            )
            continue

        records.append(
            PairedFileRecord(
                pair_key=pair["pair_key"],
                emg_file_path=emg_path,
                imu_file_path=imu_path,
                label_id=pair["label_id"],
                label_name=pair["label_name"],
                subject_id=subject_id,
                emg_num_steps=len(emg_signal),
                imu_num_steps=len(imu_signal),
                emg_num_windows=emg_num_windows,
                imu_num_windows=imu_num_windows,
                num_windows=paired_num_windows,
            )
        )

    return records, emg_channel_names, imu_channel_names, sorted(subject_ids), load_errors


def load_paired_dataset(
    emg_root,
    imu_root,
    emg_window_size,
    emg_stride,
    imu_window_size,
    imu_stride,
    max_files_per_class=None,
    emg_expected_channels=None,
    imu_expected_channels=None,
    cache_path=None,
    rebuild_index_cache=False,
    stair_up_token="l",
    ramp_up_token="l",
):
    selected_pairs, emg_only_keys, imu_only_keys = build_paired_candidate_files(
        emg_root,
        imu_root,
        max_files_per_class=max_files_per_class,
        stair_up_token=stair_up_token,
        ramp_up_token=ramp_up_token,
    )

    cache_path = resolve_pair_index_cache_path(
        emg_root=emg_root,
        imu_root=imu_root,
        emg_window_size=emg_window_size,
        emg_stride=emg_stride,
        imu_window_size=imu_window_size,
        imu_stride=imu_stride,
        max_files_per_class=max_files_per_class,
        explicit_path=cache_path,
    )
    file_signature = get_paired_file_signature(selected_pairs)
    cache_payload = None if rebuild_index_cache else load_index_cache(cache_path)

    if (
        cache_payload
        and cache_payload.get("version") == PAIR_INDEX_CACHE_VERSION
        and cache_payload.get("emg_root") == str(Path(emg_root).resolve())
        and cache_payload.get("imu_root") == str(Path(imu_root).resolve())
        and cache_payload.get("emg_window_size") == int(emg_window_size)
        and cache_payload.get("emg_stride") == int(emg_stride)
        and cache_payload.get("imu_window_size") == int(imu_window_size)
        and cache_payload.get("imu_stride") == int(imu_stride)
        and cache_payload.get("max_files_per_class") == max_files_per_class
        and cache_payload.get("emg_expected_channels") == emg_expected_channels
        and cache_payload.get("imu_expected_channels") == imu_expected_channels
        and cache_payload.get("stair_up_token") == stair_up_token
        and cache_payload.get("ramp_up_token") == ramp_up_token
        and cache_payload.get("file_signature") == file_signature
    ):
        print(f"\nLoading paired file index cache: {cache_path}")
        records = [PairedFileRecord(**record) for record in cache_payload["records"]]
        emg_channel_names = cache_payload["emg_channel_names"]
        imu_channel_names = cache_payload["imu_channel_names"]
        subject_ids = cache_payload["subject_ids"]
        load_errors = cache_payload.get("load_errors", [])
    else:
        if rebuild_index_cache:
            print(f"\nRebuilding paired file index cache: {cache_path}")
        else:
            print(f"\nCache miss or stale paired cache. Building file index: {cache_path}")

        (
            records,
            emg_channel_names,
            imu_channel_names,
            subject_ids,
            load_errors,
        ) = rebuild_paired_file_records(
            selected_pairs=selected_pairs,
            emg_window_size=emg_window_size,
            emg_stride=emg_stride,
            imu_window_size=imu_window_size,
            imu_stride=imu_stride,
            emg_expected_channels=emg_expected_channels,
            imu_expected_channels=imu_expected_channels,
        )
        save_index_cache(
            cache_path,
            {
                "version": PAIR_INDEX_CACHE_VERSION,
                "emg_root": str(Path(emg_root).resolve()),
                "imu_root": str(Path(imu_root).resolve()),
                "emg_window_size": int(emg_window_size),
                "emg_stride": int(emg_stride),
                "imu_window_size": int(imu_window_size),
                "imu_stride": int(imu_stride),
                "max_files_per_class": max_files_per_class,
                "emg_expected_channels": emg_expected_channels,
                "imu_expected_channels": imu_expected_channels,
                "stair_up_token": stair_up_token,
                "ramp_up_token": ramp_up_token,
                "file_signature": file_signature,
                "emg_channel_names": emg_channel_names,
                "imu_channel_names": imu_channel_names,
                "subject_ids": subject_ids,
                "records": [asdict(record) for record in records],
                "load_errors": load_errors,
                "emg_only_keys": emg_only_keys,
                "imu_only_keys": imu_only_keys,
            },
        )
        print(f"Saved paired file index cache: {cache_path}")

    if load_errors:
        print(f"\nSkipped paired files: {len(load_errors)}")
        for idx, (pair_key, reason) in enumerate(load_errors[:8], start=1):
            print(f"  {idx}. {pair_key} -> {reason}")
        if len(load_errors) > 8:
            print(f"  ... and {len(load_errors) - 8} more")

    if not records:
        raise ValueError(
            "No valid paired files remain after indexing. "
            "Please check paired EMG/IMU preprocessing outputs."
        )

    total_windows = sum(record.num_windows for record in records)
    print(f"\nTotal valid paired files: {len(records)}")
    print(f"Total paired windows available: {total_windows}")
    print(f"Subjects included: {len(subject_ids)} -> {', '.join(sorted(subject_ids))}")

    print("\nPaired windows per class:")
    for label_name in LABEL_NAMES:
        class_windows = sum(record.num_windows for record in records if record.label_name == label_name)
        class_files_count = sum(1 for record in records if record.label_name == label_name)
        print(f"  {label_name}: {class_windows} windows from {class_files_count} paired files")

    return records, emg_channel_names, imu_channel_names, subject_ids


class PairedSequenceWindowDataset(Dataset):
    """Stream paired EMG and IMU sliding windows from disk."""

    def __init__(
        self,
        file_records,
        emg_window_size,
        emg_stride,
        imu_window_size,
        imu_stride,
        emg_scaler=None,
        imu_scaler=None,
        fit_scaler=False,
        emg_feature_mask=None,
        imu_feature_mask=None,
        cache_size=32,
        max_windows_per_file=None,
        emg_temporal_pooling="none",
        emg_temporal_pool_size=1,
    ):
        self.file_records = list(file_records)
        self.emg_window_size = int(emg_window_size)
        self.emg_stride = int(emg_stride)
        self.imu_window_size = int(imu_window_size)
        self.imu_stride = int(imu_stride)
        (
            self.emg_temporal_pooling,
            self.emg_temporal_pool_size,
            self.emg_model_window_size,
        ) = resolve_temporal_pooling(
            emg_temporal_pooling,
            emg_temporal_pool_size,
            self.emg_window_size,
        )
        self.emg_feature_mask = emg_feature_mask
        self.imu_feature_mask = imu_feature_mask
        self.cache_size = max(1, int(cache_size))
        self.max_windows_per_file = (
            None if max_windows_per_file is None else max(1, int(max_windows_per_file))
        )
        self._pair_cache = OrderedDict()

        self.sampled_window_indices = [
            self._build_window_indices(record.num_windows) for record in self.file_records
        ]
        self.window_counts = [
            int(sampled.size) if sampled is not None else record.num_windows
            for record, sampled in zip(self.file_records, self.sampled_window_indices)
        ]
        self.cumulative_windows = np.cumsum(self.window_counts).tolist()
        self.total_windows = sum(self.window_counts)

        if fit_scaler:
            self.emg_scaler = emg_scaler if emg_scaler is not None else StandardScaler()
            self.imu_scaler = imu_scaler if imu_scaler is not None else StandardScaler()
            self._fit_scalers()
        else:
            self.emg_scaler = emg_scaler
            self.imu_scaler = imu_scaler

    def _build_window_indices(self, num_windows):
        if self.max_windows_per_file is None or num_windows <= self.max_windows_per_file:
            return None

        positions = (
            (np.arange(self.max_windows_per_file, dtype=np.float64) + 0.5)
            * num_windows
            / self.max_windows_per_file
        )
        return np.clip(positions.astype(np.int64), 0, num_windows - 1)

    def _fit_scalers(self):
        print("\nFitting modality-specific scalers on training files...")
        for record_idx, record in enumerate(tqdm(self.file_records, desc="Scaler")):
            emg_signal, imu_signal = self._load_pair(record, apply_scaler=False)
            if self.emg_temporal_pooling == "none":
                self.emg_scaler.partial_fit(emg_signal)
            else:
                window_indices = self.sampled_window_indices[record_idx]
                if window_indices is None:
                    window_indices = range(record.num_windows)
                pooled_windows = []
                for local_window_idx in window_indices:
                    start = int(local_window_idx) * self.emg_stride
                    end = start + self.emg_window_size
                    pooled_windows.append(self._apply_emg_temporal_pooling(emg_signal[start:end]))
                if pooled_windows:
                    self.emg_scaler.partial_fit(np.vstack(pooled_windows))
            self.imu_scaler.partial_fit(imu_signal)

    def _evict_if_needed(self):
        while len(self._pair_cache) > self.cache_size:
            self._pair_cache.popitem(last=False)

    def _load_pair(self, record, apply_scaler=True):
        emg_signal, _, emg_err = load_mat_file(record.emg_file_path)
        imu_signal, _, imu_err = load_mat_file(record.imu_file_path)
        if emg_err is not None or imu_err is not None:
            error_parts = []
            if emg_err is not None:
                error_parts.append(f"EMG: {emg_err}")
            if imu_err is not None:
                error_parts.append(f"IMU: {imu_err}")
            raise ValueError(
                f"Failed to reload paired file during dataset access: {record.pair_key} -> "
                + " | ".join(error_parts)
            )

        if self.emg_feature_mask is not None:
            emg_signal = apply_feature_mask(emg_signal, self.emg_feature_mask)
        if self.imu_feature_mask is not None:
            imu_signal = apply_feature_mask(imu_signal, self.imu_feature_mask)

        if (
            self.emg_temporal_pooling == "none"
            and apply_scaler
            and self.emg_scaler is not None
        ):
            emg_signal = self.emg_scaler.transform(emg_signal)
        if apply_scaler and self.imu_scaler is not None:
            imu_signal = self.imu_scaler.transform(imu_signal)

        return (
            emg_signal.astype(np.float32, copy=False),
            imu_signal.astype(np.float32, copy=False),
        )

    def _apply_emg_temporal_pooling(self, window):
        if self.emg_temporal_pooling == "none":
            return window

        grouped = window.reshape(
            self.emg_model_window_size,
            self.emg_temporal_pool_size,
            window.shape[1],
        )
        if self.emg_temporal_pooling == "mean":
            return grouped.mean(axis=1)
        if self.emg_temporal_pooling == "rms":
            return np.sqrt(np.mean(np.square(grouped), axis=1))
        raise ValueError(f"Unsupported sEMG temporal pooling mode: {self.emg_temporal_pooling}")

    def _get_cached_pair(self, record):
        if record.pair_key in self._pair_cache:
            pair = self._pair_cache.pop(record.pair_key)
            self._pair_cache[record.pair_key] = pair
            return pair

        pair = self._load_pair(record, apply_scaler=True)
        self._pair_cache[record.pair_key] = pair
        self._evict_if_needed()
        return pair

    def _locate_window(self, index):
        if index < 0 or index >= self.total_windows:
            raise IndexError(f"Window index {index} out of range [0, {self.total_windows})")

        record_idx = bisect_right(self.cumulative_windows, index)
        prev_end = 0 if record_idx == 0 else self.cumulative_windows[record_idx - 1]
        local_window_idx = index - prev_end
        return record_idx, local_window_idx

    def __len__(self):
        return self.total_windows

    def __getitem__(self, index):
        record_idx, local_window_idx = self._locate_window(index)
        record = self.file_records[record_idx]
        emg_signal, imu_signal = self._get_cached_pair(record)

        sampled_window_indices = self.sampled_window_indices[record_idx]
        if sampled_window_indices is not None:
            local_window_idx = int(sampled_window_indices[local_window_idx])

        emg_start = local_window_idx * self.emg_stride
        emg_end = emg_start + self.emg_window_size
        imu_start = local_window_idx * self.imu_stride
        imu_end = imu_start + self.imu_window_size

        emg_window = emg_signal[emg_start:emg_end]
        imu_window = imu_signal[imu_start:imu_end]
        emg_window = self._apply_emg_temporal_pooling(emg_window)
        if self.emg_temporal_pooling != "none" and self.emg_scaler is not None:
            emg_window = self.emg_scaler.transform(emg_window)

        return (
            torch.from_numpy(emg_window.astype(np.float32, copy=True)),
            torch.from_numpy(imu_window.copy()),
            record.label_id,
        )


def resolve_output_paths(explicit_prefix=None):
    prefix = explicit_prefix if explicit_prefix else "emg_imu_dual_loso"
    if prefix.endswith(".pth"):
        prefix = prefix[:-4]

    checkpoint_path = f"best_model_{prefix}.pth"
    history_path = f"training_history_{prefix}.png"
    confusion_matrix_path = f"confusion_matrix_{prefix}.png"
    return checkpoint_path, history_path, confusion_matrix_path


def select_loso_fold_specs(loso_folds, start_subject=None, end_subject=None):
    start_subject = None if start_subject is None else str(start_subject).upper()
    end_subject = None if end_subject is None else str(end_subject).upper()

    available_subjects = [subject_id for subject_id, _, _ in loso_folds]
    if start_subject is not None and start_subject not in available_subjects:
        raise ValueError(
            f"--start-subject={start_subject} is not in the LOSO subject list: {available_subjects}"
        )
    if end_subject is not None and end_subject not in available_subjects:
        raise ValueError(
            f"--end-subject={end_subject} is not in the LOSO subject list: {available_subjects}"
        )

    selected_specs = []
    collecting = start_subject is None
    for fold_index, (subject_id, train_records, val_records) in enumerate(loso_folds, start=1):
        if not collecting and subject_id.upper() == start_subject:
            collecting = True
        if not collecting:
            continue

        selected_specs.append((fold_index, subject_id, train_records, val_records))
        if end_subject is not None and subject_id.upper() == end_subject:
            break

    return selected_specs


def build_subject_group_kfold_folds(file_records, num_folds=5, random_state=42):
    """Build K folds by assigning whole subjects to validation folds."""
    subjects = sorted({record.subject_id for record in file_records})
    num_folds = int(num_folds)
    if num_folds < 2:
        raise ValueError("--num-folds must be at least 2 for subject-grouped K-fold CV.")
    if num_folds > len(subjects):
        raise ValueError(
            f"--num-folds={num_folds} is larger than the number of subjects ({len(subjects)})."
        )

    rng = np.random.default_rng(int(random_state))
    shuffled_subjects = np.array(subjects, dtype=object)
    rng.shuffle(shuffled_subjects)
    subject_groups = [list(group) for group in np.array_split(shuffled_subjects, num_folds)]

    folds = []
    for fold_index, val_subjects in enumerate(subject_groups, start=1):
        val_subject_set = set(val_subjects)
        train_records = [
            record for record in file_records if record.subject_id not in val_subject_set
        ]
        val_records = [
            record for record in file_records if record.subject_id in val_subject_set
        ]
        fold_id = "K" + f"{fold_index:02d}_" + "_".join(sorted(val_subject_set))
        folds.append((fold_id, train_records, val_records))

    return folds


def fold_artifacts_exist(fold_checkpoint_path, fold_history_path, fold_confusion_path):
    return (
        Path(fold_checkpoint_path).exists()
        and Path(fold_history_path).exists()
        and Path(fold_confusion_path).exists()
    )


def load_fold_result_from_checkpoint(checkpoint_path, fold_index=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = {
        "fold_index": fold_index,
        "subject_id": checkpoint.get("fold_subject_id"),
        "val_loss": float(checkpoint["val_loss"]),
        "val_acc": float(checkpoint["val_acc"]),
        "best_epoch": int(checkpoint["epoch"]),
        "checkpoint_path": str(checkpoint_path),
        "model_parameters": checkpoint.get("model_parameters"),
        "trainable_parameters": checkpoint.get("trainable_parameters"),
        "train_time_sec": checkpoint.get("train_time_sec"),
        "single_inference_ms": checkpoint.get("single_inference_ms"),
        "inference_benchmark": checkpoint.get("inference_benchmark"),
        "history": None,
    }
    return result


def collect_available_fold_results(base_checkpoint_path, loso_fold_specs):
    collected_results = []
    for fold_index, subject_id, _, _ in loso_fold_specs:
        fold_checkpoint_path = build_artifact_path(
            base_checkpoint_path,
            f"fold_{fold_index:02d}_{subject_id}",
        )
        if Path(fold_checkpoint_path).exists():
            collected_results.append(
                load_fold_result_from_checkpoint(fold_checkpoint_path, fold_index=fold_index)
            )
    return collected_results


def resolve_partial_output_paths(config, selected_fold_specs):
    if not selected_fold_specs:
        return config["checkpoint_path"], config["history_path"], config["confusion_matrix_path"]

    start_subject = selected_fold_specs[0][1]
    end_subject = selected_fold_specs[-1][1]
    suffix = f"partial_{start_subject}_{end_subject}"
    checkpoint_path = build_artifact_path(config["checkpoint_path"], suffix)
    history_path = build_artifact_path(config["history_path"], suffix)
    confusion_matrix_path = build_artifact_path(config["confusion_matrix_path"], suffix)
    return checkpoint_path, history_path, confusion_matrix_path


def count_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_single_window_inference(
    model,
    emg_window_size,
    emg_input_channels,
    imu_window_size,
    imu_input_channels,
    device,
    amp_enabled=False,
    warmup=20,
    repeat=200,
):
    warmup = max(0, int(warmup))
    repeat = max(1, int(repeat))
    model_was_training = model.training
    model.eval()

    emg_sample = torch.randn(
        1,
        int(emg_window_size),
        int(emg_input_channels),
        device=device,
    )
    imu_sample = torch.randn(
        1,
        int(imu_window_size),
        int(imu_input_channels),
        device=device,
    )

    with torch.inference_mode():
        for _ in range(warmup):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                model(emg_sample, imu_sample)

        synchronize_device(device)
        start_time = perf_counter()
        for _ in range(repeat):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                model(emg_sample, imu_sample)
        synchronize_device(device)
        elapsed_sec = perf_counter() - start_time

    if model_was_training:
        model.train()

    mean_ms = elapsed_sec * 1000.0 / repeat
    return {
        "single_inference_ms": mean_ms,
        "single_inference_sec": mean_ms / 1000.0,
        "benchmark_warmup": warmup,
        "benchmark_repeat": repeat,
        "benchmark_batch_size": 1,
    }


def resolve_runtime_metrics_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_runtime_metrics.csv")


def write_runtime_metrics_csv(fold_results, checkpoint_path):
    metrics_path = resolve_runtime_metrics_path(checkpoint_path)
    fieldnames = [
        "fold_index",
        "subject_id",
        "val_loss",
        "val_acc",
        "best_epoch",
        "model_parameters",
        "trainable_parameters",
        "train_time_sec",
        "single_inference_ms",
        "checkpoint_path",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in fold_results:
            row = {field: result.get(field) for field in fieldnames}
            if row["best_epoch"] is not None:
                row["best_epoch"] = int(row["best_epoch"]) + 1
            writer.writerow(row)
    return metrics_path


def summarize_runtime_results(fold_results):
    train_times = [
        float(result["train_time_sec"])
        for result in fold_results
        if result.get("train_time_sec") is not None
    ]
    inference_times = [
        float(result["single_inference_ms"])
        for result in fold_results
        if result.get("single_inference_ms") is not None
    ]
    parameter_counts = [
        int(result["model_parameters"])
        for result in fold_results
        if result.get("model_parameters") is not None
    ]

    if not train_times and not inference_times and not parameter_counts:
        return

    print("\nRuntime / size summary:")
    if parameter_counts:
        print(f"  model parameters: {parameter_counts[0]:,}")
    if train_times:
        print(f"  mean train time per fold: {np.mean(train_times):.2f} s +/- {np.std(train_times):.2f} s")
    if inference_times:
        print(
            "  mean single-window inference: "
            f"{np.mean(inference_times):.4f} ms +/- {np.std(inference_times):.4f} ms"
        )


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
            outputs = model(emg_sequences, imu_sequences)
            loss = criterion(outputs, labels)

        if amp_enabled and grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
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
                outputs = model(emg_sequences, imu_sequences)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), 100.0 * correct / total, all_preds, all_labels


def main():
    args = parse_args()
    emg_data_root = args.emg_data_root if args.emg_data_root else resolve_default_data_root("emg")
    imu_data_root = args.imu_data_root if args.imu_data_root else resolve_default_data_root("imu")
    emg_sample_rate_hz = (
        float(args.emg_sample_rate_hz)
        if args.emg_sample_rate_hz is not None
        else SENSOR_DEFAULT_SAMPLE_RATE["emg"]
    )
    imu_sample_rate_hz = (
        float(args.imu_sample_rate_hz)
        if args.imu_sample_rate_hz is not None
        else SENSOR_DEFAULT_SAMPLE_RATE["imu"]
    )
    emg_window_size = (
        int(args.emg_window_size)
        if args.emg_window_size is not None
        else ms_to_samples(args.window_ms, emg_sample_rate_hz)
    )
    imu_window_size = (
        int(args.imu_window_size)
        if args.imu_window_size is not None
        else ms_to_samples(args.window_ms, imu_sample_rate_hz)
    )
    emg_stride = (
        int(args.emg_stride)
        if args.emg_stride is not None
        else ms_to_samples(args.stride_ms, emg_sample_rate_hz)
    )
    imu_stride = (
        int(args.imu_stride)
        if args.imu_stride is not None
        else ms_to_samples(args.stride_ms, imu_sample_rate_hz)
    )
    emg_temporal_pooling, emg_temporal_pool_size, emg_model_window_size = resolve_temporal_pooling(
        args.emg_temporal_pooling,
        args.emg_temporal_pool_size,
        emg_window_size,
    )
    resolved_num_workers = resolve_num_workers(args.num_workers)
    checkpoint_path, history_path, confusion_matrix_path = resolve_output_paths(args.output_prefix)

    config = {
        "emg_data_root": emg_data_root,
        "imu_data_root": imu_data_root,
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "emg_sample_rate_hz": emg_sample_rate_hz,
        "imu_sample_rate_hz": imu_sample_rate_hz,
        "emg_window_size": emg_window_size,
        "imu_window_size": imu_window_size,
        "emg_stride": emg_stride,
        "imu_stride": imu_stride,
        "emg_temporal_pooling": emg_temporal_pooling,
        "emg_temporal_pool_size": emg_temporal_pool_size,
        "emg_model_window_size": emg_model_window_size,
        "num_classes": len(CLASS_TO_ID),
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
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
    print("Dual-branch Mamba training on paired sEMG + IMU data")
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
            f"EMG temporal pooling: {config['emg_temporal_pooling']} "
            f"x{config['emg_temporal_pool_size']} -> model sequence length "
            f"{config['emg_model_window_size']}"
        )
    else:
        print("EMG temporal pooling: none")
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

    is_partial_run = len(selected_fold_specs) != len(cv_folds)
    print(
        "Selected folds: "
        + ", ".join(
            f"{fold_index:02d}:{subject_id}" for fold_index, subject_id, _, _ in selected_fold_specs
        )
    )
    if is_partial_run:
        print("Running a partial CV range. Aggregate outputs will be written to partial artifact names.")

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
            if (
                best_overall_fold is None
                or existing_result["val_loss"] < best_overall_fold["val_loss"]
            ):
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

        model = DualBranchMambaSequenceClassifier(
            emg_input_channels=emg_input_channels,
            imu_input_channels=imu_input_channels,
            num_classes=config["num_classes"],
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            d_state=config["d_state"],
            d_conv=config["d_conv"],
            expand=config["expand"],
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
                    "emg_feature_mask": (
                        emg_feature_mask.tolist() if emg_feature_mask is not None else None
                    ),
                    "imu_feature_mask": (
                        imu_feature_mask.tolist() if imu_feature_mask is not None else None
                    ),
                    "selected_emg_feature_indices": selected_emg_feature_indices,
                    "selected_imu_feature_indices": selected_imu_feature_indices,
                    "original_emg_channels": original_emg_channels,
                    "original_imu_channels": original_imu_channels,
                    "emg_channel_names": detected_emg_channel_names,
                    "imu_channel_names": detected_imu_channel_names,
                    "label_names": LABEL_NAMES,
                    "label_display_names": LABEL_DISPLAY_NAMES,
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
                "model_family": "DualBranchMamba",
            }
        )
        torch.save(best_checkpoint, fold_checkpoint_path)
        fold_results.append(fold_result)
        all_cv_preds.extend(val_preds)
        all_cv_labels.extend(val_labels)

        print("\nFold result:")
        print(f"  subject: {val_subject_id}")
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
        print("\nCurrent-session LOSO classification report:")
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

        aggregate_checkpoint_path = config["checkpoint_path"]
        aggregate_history_path = config["history_path"]
        aggregate_confusion_matrix_path = config["confusion_matrix_path"]
        if is_partial_run:
            (
                aggregate_checkpoint_path,
                aggregate_history_path,
                aggregate_confusion_matrix_path,
            ) = resolve_partial_output_paths(config, selected_fold_specs)

        plot_confusion_matrix(
            all_cv_labels,
            all_cv_preds,
            LABEL_DISPLAY_NAMES,
            save_path=aggregate_confusion_matrix_path,
        )

        if best_overall_fold is not None and best_overall_fold["history"] is not None:
            best_overall_history = best_overall_fold["history"]
            plot_training_history(best_overall_history, save_path=aggregate_history_path)
            best_overall_checkpoint = torch.load(
                best_overall_fold["checkpoint_path"],
                weights_only=False,
            )
            torch.save(best_overall_checkpoint, aggregate_checkpoint_path)
    else:
        aggregate_checkpoint_path = config["checkpoint_path"]
        aggregate_history_path = config["history_path"]
        aggregate_confusion_matrix_path = config["confusion_matrix_path"]

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
            f"(subject {best_overall_fold['subject_id']})"
        )
        print(f"Best fold val loss: {best_overall_fold['val_loss']:.4f}")
        print(f"Best fold val acc: {best_overall_fold['val_acc']:.2f}%")
    if all_cv_labels and all_cv_preds:
        print(f"Aggregate confusion matrix saved to: {aggregate_confusion_matrix_path}")
        print(f"Best overall fold checkpoint copied to: {aggregate_checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
