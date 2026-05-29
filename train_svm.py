from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from train_mamba_emg_imu import (
    LABEL_DISPLAY_NAMES,
    LABEL_NAMES,
    apply_feature_mask,
    build_leave_one_subject_out_folds,
    load_dataset,
    load_mat_file,
    ms_to_samples,
    parse_feature_mask,
    resolve_default_data_root,
    resolve_sample_rate_hz,
    summarize_split,
)
from train_mamba_emg_imu_dual_branch import load_paired_dataset

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


warnings.filterwarnings("ignore")
RANDOM_SEED = 42
EMG_FEATURE_TYPES = ["mean", "std", "rms", "mav", "iemg", "wl", "zc", "ssc"]
IMU_FEATURE_TYPES = ["mean", "std", "min", "max", "range", "rms", "mav", "energy", "zc", "ssc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOSO SVM baseline on handcrafted window features.")
    parser.add_argument("--input-mode", choices=("single", "dual"), default="dual")
    parser.add_argument("--window-ms", type=float, default=500.0)
    parser.add_argument("--stride-ms", type=float, default=250.0)
    parser.add_argument("--max-files-per-class", type=int, default=None)
    parser.add_argument("--index-cache", type=str, default=None)
    parser.add_argument("--rebuild-index-cache", action="store_true")
    parser.add_argument("--max-train-windows-per-file", type=int, default=8)
    parser.add_argument("--max-test-windows-per-file", type=int, default=None)
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
    parser.add_argument("--emg-expected-channels", type=int, default=None)
    parser.add_argument("--imu-expected-channels", type=int, default=None)

    parser.add_argument("--stair-up-token", choices=("l", "r"), default="l")
    parser.add_argument("--ramp-up-token", choices=("l", "r"), default="l")
    parser.add_argument("--svm-kernel", choices=("linear", "rbf", "poly", "sigmoid"), default="linear")
    parser.add_argument("--svm-backend", choices=("auto", "libsvm", "liblinear"), default="auto")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--svm-degree", type=int, default=3)
    parser.add_argument("--svm-class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--svm-max-iter", type=int, default=5000)
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--export-feature-excel", action="store_true")
    parser.add_argument("--feature-excel-path", type=Path, default=None)
    parser.add_argument("--export-feature-max-windows-per-file", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def configure_style() -> None:
    if plt is None or sns is None:
        return
    sns.set_theme(style="whitegrid")


def format_ms_tag(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def get_selected_channel_names(channel_names: list[str], feature_mask: np.ndarray | None) -> list[str]:
    if feature_mask is None:
        return list(channel_names)
    return [name for name, keep in zip(channel_names, feature_mask.tolist()) if int(keep) == 1]


def resolve_feature_excel_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.feature_excel_path is not None:
        return args.feature_excel_path.resolve()
    return output_dir / "svm_extracted_features.xlsx"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    w = format_ms_tag(args.window_ms)
    s = format_ms_tag(args.stride_ms)
    if args.input_mode == "dual":
        return Path.cwd() / f"SVM{w}ms滑窗{s}ms步长sEMG和imu结果图"
    sensor_tag = "imu" if str(args.sensor_folder).lower() == "imu" else "sEMG"
    return Path.cwd() / f"SVM{w}ms滑窗{s}ms步长{sensor_tag}结果图"


def sample_window_indices(num_windows: int, max_windows_per_file: int | None) -> np.ndarray:
    if max_windows_per_file is None or num_windows <= max_windows_per_file:
        return np.arange(num_windows, dtype=np.int64)
    positions = ((np.arange(max_windows_per_file, dtype=np.float64) + 0.5) * num_windows / max_windows_per_file)
    return np.clip(positions.astype(np.int64), 0, num_windows - 1)


def build_svm_model(args: argparse.Namespace):
    class_weight = None if args.svm_class_weight == "none" else args.svm_class_weight
    backend = args.svm_backend

    if backend == "auto":
        backend = "liblinear" if args.svm_kernel == "linear" else "libsvm"

    if backend == "liblinear":
        if args.svm_kernel != "linear":
            raise ValueError("LinearSVC only supports --svm-kernel linear.")
        return LinearSVC(
            C=float(args.svm_c),
            class_weight=class_weight,
            max_iter=int(args.svm_max_iter),
            random_state=RANDOM_SEED,
            dual="auto",
        )

    return SVC(
        kernel=args.svm_kernel,
        C=float(args.svm_c),
        gamma=args.svm_gamma,
        degree=int(args.svm_degree),
        class_weight=class_weight,
        decision_function_shape="ovr",
        max_iter=int(args.svm_max_iter),
        random_state=RANDOM_SEED,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def estimate_fit_memory_gb(x: np.ndarray) -> float:
    return float(x.shape[0] * x.shape[1] * 8 / (1024 ** 3))


def build_single_feature_names(setup: dict[str, object]) -> list[str]:
    sensor_folder = str(setup["sensor_folder"]).lower()
    feature_types = IMU_FEATURE_TYPES if sensor_folder == "imu" else EMG_FEATURE_TYPES
    channel_names = list(setup["selected_channel_names"])
    return [f"{sensor_folder}:{channel}:{feature_type}" for channel in channel_names for feature_type in feature_types]


def build_dual_feature_names(setup: dict[str, object]) -> list[str]:
    emg_names = [
        f"emg:{channel}:{feature_type}"
        for channel in list(setup["selected_emg_channel_names"])
        for feature_type in EMG_FEATURE_TYPES
    ]
    imu_names = [
        f"imu:{channel}:{feature_type}"
        for channel in list(setup["selected_imu_channel_names"])
        for feature_type in IMU_FEATURE_TYPES
    ]
    return emg_names + imu_names


def get_feature_names(setup: dict[str, object]) -> list[str]:
    if str(setup["mode"]) == "dual":
        return build_dual_feature_names(setup)
    return build_single_feature_names(setup)


def extract_linear_feature_importance(model) -> np.ndarray | None:
    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    coef = np.asarray(coef, dtype=np.float64)
    if coef.ndim == 1:
        return np.abs(coef)
    return np.mean(np.abs(coef), axis=0)


def export_feature_excel(
    setup: dict[str, object],
    args: argparse.Namespace,
    feature_names: list[str],
    output_path: Path,
) -> None:
    max_windows_per_file = args.export_feature_max_windows_per_file
    if max_windows_per_file is None:
        max_windows_per_file = args.max_train_windows_per_file

    rows: list[dict[str, object]] = []

    if str(setup["mode"]) == "dual":
        for record in setup["records"]:
            indices = sample_window_indices(record.num_windows, max_windows_per_file)
            emg_x, imu_x, _ = load_dual_features(
                record,
                int(setup["emg_window_size"]),
                int(setup["emg_stride"]),
                int(setup["imu_window_size"]),
                int(setup["imu_stride"]),
                setup["emg_feature_mask"],
                setup["imu_feature_mask"],
                max_windows_per_file,
                args,
            )
            x = np.concatenate([emg_x, imu_x], axis=1)
            for row_idx, window_index in enumerate(indices.tolist()):
                row = {
                    "subject_id": record.subject_id,
                    "label_id": record.label_id,
                    "label_name": record.label_name,
                    "pair_key": record.pair_key,
                    "emg_file_path": record.emg_file_path,
                    "imu_file_path": record.imu_file_path,
                    "window_index": int(window_index),
                    "emg_start_sample": int(window_index * int(setup["emg_stride"])),
                    "imu_start_sample": int(window_index * int(setup["imu_stride"])),
                }
                for col_idx, feature_name in enumerate(feature_names):
                    row[feature_name] = float(x[row_idx, col_idx])
                rows.append(row)
    else:
        for record in setup["records"]:
            indices = sample_window_indices(record.num_windows, max_windows_per_file)
            x, _ = load_single_features(
                record,
                int(setup["window_size"]),
                int(setup["stride"]),
                setup["feature_mask"],
                max_windows_per_file,
                str(setup["sensor_folder"]),
                args,
            )
            for row_idx, window_index in enumerate(indices.tolist()):
                row = {
                    "subject_id": record.subject_id,
                    "label_id": record.label_id,
                    "label_name": record.label_name,
                    "file_path": record.file_path,
                    "window_index": int(window_index),
                    "start_sample": int(window_index * int(setup["stride"])),
                }
                for col_idx, feature_name in enumerate(feature_names):
                    row[feature_name] = float(x[row_idx, col_idx])
                rows.append(row)

    if not rows:
        raise ValueError("No extracted features are available to export.")

    feature_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        feature_df.to_excel(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl or xlsxwriter. "
            "Please install one of them in the project virtual environment."
        ) from exc


def _zero_crossings(window: np.ndarray, threshold: float) -> np.ndarray:
    prev = window[:-1]
    curr = window[1:]
    sign_change = (prev * curr) < 0
    amp_change = np.abs(curr - prev) >= threshold
    return np.sum(sign_change & amp_change, axis=0).astype(np.float32)


def _slope_sign_changes(window: np.ndarray, threshold: float) -> np.ndarray:
    diff1 = window[1:-1] - window[:-2]
    diff2 = window[1:-1] - window[2:]
    sign_change = (diff1 * diff2) > 0
    amp_change = (np.abs(diff1) >= threshold) | (np.abs(diff2) >= threshold)
    return np.sum(sign_change & amp_change, axis=0).astype(np.float32)


def extract_emg_channel_features(window: np.ndarray, zc_threshold: float, ssc_threshold: float) -> np.ndarray:
    abs_window = np.abs(window)
    diff = np.diff(window, axis=0)
    rms = np.sqrt(np.mean(window ** 2, axis=0))
    features = [
        np.mean(window, axis=0),
        np.std(window, axis=0),
        rms,
        np.mean(abs_window, axis=0),
        np.sum(abs_window, axis=0),
        np.sum(np.abs(diff), axis=0),
        _zero_crossings(window, zc_threshold),
        _slope_sign_changes(window, ssc_threshold),
    ]
    return np.concatenate([feature.astype(np.float32, copy=False) for feature in features], axis=0)


def extract_imu_channel_features(window: np.ndarray, zc_threshold: float, ssc_threshold: float) -> np.ndarray:
    abs_window = np.abs(window)
    diff = np.diff(window, axis=0)
    rms = np.sqrt(np.mean(window ** 2, axis=0))
    features = [
        np.mean(window, axis=0),
        np.std(window, axis=0),
        np.min(window, axis=0),
        np.max(window, axis=0),
        np.max(window, axis=0) - np.min(window, axis=0),
        rms,
        np.mean(abs_window, axis=0),
        np.sum(diff ** 2, axis=0),
        _zero_crossings(window, zc_threshold),
        _slope_sign_changes(window, ssc_threshold),
    ]
    return np.concatenate([feature.astype(np.float32, copy=False) for feature in features], axis=0)


def extract_single_window_features(
    window: np.ndarray,
    sensor_folder: str,
    args: argparse.Namespace,
) -> np.ndarray:
    if str(sensor_folder).lower() == "imu":
        return extract_imu_channel_features(
            window,
            zc_threshold=float(args.imu_zc_threshold),
            ssc_threshold=float(args.imu_ssc_threshold),
        )
    return extract_emg_channel_features(
        window,
        zc_threshold=float(args.emg_zc_threshold),
        ssc_threshold=float(args.emg_ssc_threshold),
    )


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    if plt is None or sns is None:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_DISPLAY_NAMES, yticklabels=LABEL_DISPLAY_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fold_accuracy(fold_df: pd.DataFrame, title: str, output_path: Path) -> None:
    if plt is None or sns is None or fold_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    sns.barplot(data=fold_df.sort_values(["fold_index", "subject_id"]), x="subject_id", y="accuracy_pct", color="#2F5C85", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Held-out Subject")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.grid(axis="x", visible=False)
    plt.xticks(rotation=45)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_feature_importance(feature_df: pd.DataFrame, output_path: Path, top_k: int = 20) -> None:
    if plt is None or sns is None or feature_df.empty:
        return
    plot_df = feature_df.nlargest(top_k, "importance").iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    sns.barplot(data=plot_df, x="importance", y="feature_name", color="#D97A2B", ax=ax)
    ax.set_title(f"Top {top_k} Feature Importances")
    ax.set_xlabel("Mean Absolute Linear Weight")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.grid(axis="y", visible=False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_group_importance(group_df: pd.DataFrame, output_path: Path) -> None:
    if plt is None or sns is None or group_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.barplot(data=group_df.sort_values("importance", ascending=False), x="group", y="importance", color="#2F5C85", ax=ax)
    ax.set_title("Feature Importance by Group")
    ax.set_xlabel("")
    ax.set_ylabel("Summed Mean Absolute Weight")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.grid(axis="x", visible=False)
    plt.xticks(rotation=45, ha="right")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def resolve_single_setup(args: argparse.Namespace) -> dict[str, object]:
    sensor_folder = str(args.sensor_folder).lower()
    data_root = args.data_root if args.data_root else resolve_default_data_root(sensor_folder)
    sample_rate_hz = resolve_sample_rate_hz(args.sample_rate_hz, sensor_folder)
    window_size = args.window_size if args.window_size is not None else ms_to_samples(args.window_ms, sample_rate_hz)
    stride = args.stride if args.stride is not None else ms_to_samples(args.stride_ms, sample_rate_hz)
    records, channel_names, subject_ids = load_dataset(
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        max_files_per_class=args.max_files_per_class,
        sensor_folder=sensor_folder,
        expected_channels=args.expected_channels,
        cache_path=args.index_cache,
        rebuild_index_cache=args.rebuild_index_cache,
        stair_up_token=args.stair_up_token,
        ramp_up_token=args.ramp_up_token,
    )
    feature_mask = parse_feature_mask(args.feature_mask, len(channel_names))
    return {
        "mode": "single",
        "data_root": data_root,
        "sensor_folder": sensor_folder,
        "sample_rate_hz": sample_rate_hz,
        "window_size": window_size,
        "stride": stride,
        "records": records,
        "subject_ids": subject_ids,
        "channel_names": channel_names,
        "feature_mask": feature_mask,
        "selected_channel_names": get_selected_channel_names(channel_names, feature_mask),
        "selected_channels": int(feature_mask.sum()) if feature_mask is not None else len(channel_names),
        "artifact_prefix": f"svm_{sensor_folder}_loso",
        "accuracy_title": "SVM LOSO Accuracy by Held-out Subject",
    }


def resolve_dual_setup(args: argparse.Namespace) -> dict[str, object]:
    emg_data_root = args.emg_data_root if args.emg_data_root else resolve_default_data_root("emg")
    imu_data_root = args.imu_data_root if args.imu_data_root else resolve_default_data_root("imu")
    emg_sr = resolve_sample_rate_hz(args.emg_sample_rate_hz, "emg")
    imu_sr = resolve_sample_rate_hz(args.imu_sample_rate_hz, "imu")
    emg_ws = args.emg_window_size if args.emg_window_size is not None else ms_to_samples(args.window_ms, emg_sr)
    imu_ws = args.imu_window_size if args.imu_window_size is not None else ms_to_samples(args.window_ms, imu_sr)
    emg_stride = args.emg_stride if args.emg_stride is not None else ms_to_samples(args.stride_ms, emg_sr)
    imu_stride = args.imu_stride if args.imu_stride is not None else ms_to_samples(args.stride_ms, imu_sr)
    records, emg_channels, imu_channels, subject_ids = load_paired_dataset(
        emg_root=emg_data_root,
        imu_root=imu_data_root,
        emg_window_size=emg_ws,
        emg_stride=emg_stride,
        imu_window_size=imu_ws,
        imu_stride=imu_stride,
        max_files_per_class=args.max_files_per_class,
        emg_expected_channels=args.emg_expected_channels,
        imu_expected_channels=args.imu_expected_channels,
        cache_path=args.index_cache,
        rebuild_index_cache=args.rebuild_index_cache,
        stair_up_token=args.stair_up_token,
        ramp_up_token=args.ramp_up_token,
    )
    emg_mask = parse_feature_mask(args.emg_feature_mask, len(emg_channels))
    imu_mask = parse_feature_mask(args.imu_feature_mask, len(imu_channels))
    return {
        "mode": "dual",
        "emg_data_root": emg_data_root,
        "imu_data_root": imu_data_root,
        "emg_sample_rate_hz": emg_sr,
        "imu_sample_rate_hz": imu_sr,
        "emg_window_size": emg_ws,
        "imu_window_size": imu_ws,
        "emg_stride": emg_stride,
        "imu_stride": imu_stride,
        "records": records,
        "subject_ids": subject_ids,
        "emg_channel_names": emg_channels,
        "imu_channel_names": imu_channels,
        "emg_feature_mask": emg_mask,
        "imu_feature_mask": imu_mask,
        "selected_emg_channel_names": get_selected_channel_names(emg_channels, emg_mask),
        "selected_imu_channel_names": get_selected_channel_names(imu_channels, imu_mask),
        "selected_emg_channels": int(emg_mask.sum()) if emg_mask is not None else len(emg_channels),
        "selected_imu_channels": int(imu_mask.sum()) if imu_mask is not None else len(imu_channels),
        "artifact_prefix": "svm_emg_imu_dual_loso",
        "accuracy_title": "SVM LOSO Accuracy by Held-out Subject (sEMG + IMU)",
    }


def load_single_features(
    record,
    window_size: int,
    stride: int,
    feature_mask: np.ndarray | None,
    max_windows_per_file: int | None,
    sensor_folder: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    signal, _, err = load_mat_file(record.file_path)
    if err is not None:
        raise ValueError(f"Failed to load file during SVM window extraction: {record.file_path} -> {err}")
    if feature_mask is not None:
        signal = apply_feature_mask(signal, feature_mask)
    starts = sample_window_indices(record.num_windows, max_windows_per_file) * stride
    x = np.stack(
        [
            extract_single_window_features(
                signal[start : start + window_size].astype(np.float32, copy=False),
                sensor_folder=sensor_folder,
                args=args,
            )
            for start in starts
        ],
        axis=0,
    ).astype(np.float32)
    y = np.full(x.shape[0], record.label_id, dtype=np.int64)
    return x, y


def load_dual_features(
    record,
    emg_window_size: int,
    emg_stride: int,
    imu_window_size: int,
    imu_stride: int,
    emg_feature_mask: np.ndarray | None,
    imu_feature_mask: np.ndarray | None,
    max_windows_per_file: int | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emg_signal, _, emg_err = load_mat_file(record.emg_file_path)
    imu_signal, _, imu_err = load_mat_file(record.imu_file_path)
    if emg_err is not None or imu_err is not None:
        parts = []
        if emg_err is not None:
            parts.append(f"EMG: {emg_err}")
        if imu_err is not None:
            parts.append(f"IMU: {imu_err}")
        raise ValueError(f"Failed to load paired file during SVM window extraction: {record.pair_key} -> " + " | ".join(parts))
    if emg_feature_mask is not None:
        emg_signal = apply_feature_mask(emg_signal, emg_feature_mask)
    if imu_feature_mask is not None:
        imu_signal = apply_feature_mask(imu_signal, imu_feature_mask)
    indices = sample_window_indices(record.num_windows, max_windows_per_file)
    emg_starts = indices * emg_stride
    imu_starts = indices * imu_stride
    emg_x = np.stack(
        [
            extract_emg_channel_features(
                emg_signal[start : start + emg_window_size].astype(np.float32, copy=False),
                zc_threshold=float(args.emg_zc_threshold),
                ssc_threshold=float(args.emg_ssc_threshold),
            )
            for start in emg_starts
        ],
        axis=0,
    ).astype(np.float32)
    imu_x = np.stack(
        [
            extract_imu_channel_features(
                imu_signal[start : start + imu_window_size].astype(np.float32, copy=False),
                zc_threshold=float(args.imu_zc_threshold),
                ssc_threshold=float(args.imu_ssc_threshold),
            )
            for start in imu_starts
        ],
        axis=0,
    ).astype(np.float32)
    y = np.full(emg_x.shape[0], record.label_id, dtype=np.int64)
    return emg_x, imu_x, y


def fit_single_scaler(records: list, setup: dict[str, object], max_windows_per_file: int | None, args: argparse.Namespace) -> StandardScaler:
    scaler = StandardScaler()
    for record in records:
        x, _ = load_single_features(
            record,
            int(setup["window_size"]),
            int(setup["stride"]),
            setup["feature_mask"],
            max_windows_per_file,
            str(setup["sensor_folder"]),
            args,
        )
        scaler.partial_fit(x)
    return scaler


def materialize_single(
    records: list,
    setup: dict[str, object],
    max_windows_per_file: int | None,
    scaler: StandardScaler,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for record in records:
        x, y = load_single_features(
            record,
            int(setup["window_size"]),
            int(setup["stride"]),
            setup["feature_mask"],
            max_windows_per_file,
            str(setup["sensor_folder"]),
            args,
        )
        xs.append(scaler.transform(x).astype(np.float32, copy=False))
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fit_dual_scalers(
    records: list,
    setup: dict[str, object],
    max_windows_per_file: int | None,
    args: argparse.Namespace,
) -> tuple[StandardScaler, StandardScaler]:
    emg_scaler = StandardScaler()
    imu_scaler = StandardScaler()
    for record in records:
        emg_x, imu_x, _ = load_dual_features(
            record,
            int(setup["emg_window_size"]),
            int(setup["emg_stride"]),
            int(setup["imu_window_size"]),
            int(setup["imu_stride"]),
            setup["emg_feature_mask"],
            setup["imu_feature_mask"],
            max_windows_per_file,
            args,
        )
        emg_scaler.partial_fit(emg_x)
        imu_scaler.partial_fit(imu_x)
    return emg_scaler, imu_scaler


def materialize_dual(
    records: list,
    setup: dict[str, object],
    max_windows_per_file: int | None,
    emg_scaler: StandardScaler,
    imu_scaler: StandardScaler,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for record in records:
        emg_x, imu_x, y = load_dual_features(
            record,
            int(setup["emg_window_size"]),
            int(setup["emg_stride"]),
            int(setup["imu_window_size"]),
            int(setup["imu_stride"]),
            setup["emg_feature_mask"],
            setup["imu_feature_mask"],
            max_windows_per_file,
            args,
        )
        emg_x = emg_scaler.transform(emg_x).astype(np.float32, copy=False)
        imu_x = imu_scaler.transform(imu_x).astype(np.float32, copy=False)
        xs.append(np.concatenate([emg_x, imu_x], axis=1))
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def write_report(output_path: Path, args: argparse.Namespace, setup: dict[str, object], summary: dict[str, object], fold_df: pd.DataFrame) -> None:
    lines = ["# SVM LOSO Experiment Summary", "", "## Protocol"]
    if args.input_mode == "dual":
        lines.append("- Input: paired sEMG and IMU windows.")
        lines.append("- Features: extract handcrafted features from each sEMG channel and each IMU channel, scale each modality on training folds only, then concatenate.")
    else:
        lines.append("- Input: single-modality windows transformed into handcrafted channel-wise features.")
    lines.append("- Evaluation: leave-one-subject-out cross-validation.")
    lines.append("- Leakage control: scaler(s) fit on training folds only.")
    lines.append("- Leakage control: SVM re-instantiated inside every fold.")
    lines.append("- Hyperparameters are fixed before LOSO and never tuned on the held-out subject.")
    lines.append("- StandardScaler is applied after handcrafted feature extraction and before SVM fitting.")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append(f"- Folds completed: {summary['folds_completed']}")
    lines.append(f"- Mean accuracy: {summary['mean_accuracy_pct']:.2f}% +/- {summary['std_accuracy_pct']:.2f}%")
    lines.append(f"- Mean weighted F1: {summary['mean_f1_weighted_pct']:.2f}% +/- {summary['std_f1_weighted_pct']:.2f}%")
    lines.append(f"- Mean macro F1: {summary['mean_f1_macro_pct']:.2f}% +/- {summary['std_f1_macro_pct']:.2f}%")
    lines.append(f"- Aggregate accuracy: {summary['aggregate_accuracy_pct']:.2f}%")
    if not fold_df.empty:
        best_row = fold_df.loc[fold_df["accuracy_pct"].idxmax()]
        worst_row = fold_df.loc[fold_df["accuracy_pct"].idxmin()]
        lines.append("")
        lines.append("## Subject-level Extremes")
        lines.append(f"- Best subject: {best_row['subject_id']} ({best_row['accuracy_pct']:.2f}%)")
        lines.append(f"- Worst subject: {worst_row['subject_id']} ({worst_row['accuracy_pct']:.2f}%)")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "svm_loso_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    setup = resolve_dual_setup(args) if args.input_mode == "dual" else resolve_single_setup(args)
    artifact_prefix = str(setup["artifact_prefix"])
    feature_names = get_feature_names(setup)

    print("=" * 80)
    print("SVM baseline on handcrafted window features with LOSO-CV")
    print("=" * 80)
    print(f"Input mode: {args.input_mode}")
    print(f"Output dir: {output_dir}")
    print("Important: each LOSO fold will fit scaler(s) on training only and create a brand-new SVM.")
    print(f"Subjects found: {len(setup['subject_ids'])} -> {', '.join(setup['subject_ids'])}")
    print(f"Classes: {', '.join(LABEL_DISPLAY_NAMES)}")

    if args.export_feature_excel:
        feature_excel_path = resolve_feature_excel_path(args, output_dir)
        print(f"Exporting extracted features to Excel: {feature_excel_path}")
        export_feature_excel(setup, args, feature_names, feature_excel_path)
        print(f"Feature Excel saved: {feature_excel_path}")

    if args.input_mode == "dual":
        print(f"sEMG root: {setup['emg_data_root']}")
        print(f"IMU root: {setup['imu_data_root']}")
        print(f"sEMG window/stride: {setup['emg_window_size']} / {setup['emg_stride']} samples")
        print(f"IMU window/stride: {setup['imu_window_size']} / {setup['imu_stride']} samples")
        print(f"sEMG channels: {setup['selected_emg_channels']} | IMU channels: {setup['selected_imu_channels']}")
    else:
        print(f"Data root: {setup['data_root']}")
        print(f"Sensor folder: {setup['sensor_folder']}")
        print(f"Window/stride: {setup['window_size']} / {setup['stride']} samples")
        print(f"Channels: {setup['selected_channels']}")

    loso_folds = build_leave_one_subject_out_folds(setup["records"])
    if not loso_folds:
        raise ValueError(
            "LOSO-CV requires at least two subjects. Increase --max-files-per-class if your subset keeps only one subject."
        )
    print(f"LOSO folds: {len(loso_folds)}")
    effective_backend = args.svm_backend
    if effective_backend == "auto":
        effective_backend = "liblinear" if args.svm_kernel == "linear" else "libsvm"
    print(f"SVM kernel/backend: {args.svm_kernel} / {effective_backend}")
    if effective_backend == "libsvm" and args.svm_kernel == "linear":
        print("Warning: libsvm linear SVC can be very slow on large flattened windows. Prefer --svm-backend liblinear.")
    if args.svm_kernel != "linear":
        print("Warning: non-linear SVM on flattened raw windows can be very slow and memory intensive.")

    fold_results: list[dict[str, object]] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    fold_importances: list[np.ndarray] = []

    for fold_index, (subject_id, train_records, test_records) in enumerate(loso_folds, start=1):
        print("\n" + "=" * 80)
        print(f"Fold {fold_index:02d}/{len(loso_folds)} - Held-out subject: {subject_id}")
        print("=" * 80)
        summarize_split("Train", train_records)
        summarize_split("Test", test_records)

        if args.input_mode == "dual":
            print("\nFitting modality-specific scalers on training windows only...")
            emg_scaler, imu_scaler = fit_dual_scalers(train_records, setup, args.max_train_windows_per_file, args)
            print("Materializing paired training features...")
            X_train, y_train = materialize_dual(train_records, setup, args.max_train_windows_per_file, emg_scaler, imu_scaler, args)
            print("Materializing paired test features...")
            X_test, y_test = materialize_dual(test_records, setup, args.max_test_windows_per_file, emg_scaler, imu_scaler, args)
            scaler_payload = {"emg_scaler": emg_scaler, "imu_scaler": imu_scaler}
            emg_feature_dim = int(setup["selected_emg_channels"]) * 8
            imu_feature_dim = int(setup["selected_imu_channels"]) * 10
            print(f"EMG feature dim: {emg_feature_dim}")
            print(f"IMU feature dim: {imu_feature_dim}")
            print(f"Total fused feature dim: {emg_feature_dim + imu_feature_dim}")
            print("Feature scaling: StandardScaler on EMG features and IMU features separately before concatenation.")
        else:
            print("\nFitting scaler on training windows only...")
            scaler = fit_single_scaler(train_records, setup, args.max_train_windows_per_file, args)
            print("Materializing training features...")
            X_train, y_train = materialize_single(train_records, setup, args.max_train_windows_per_file, scaler, args)
            print("Materializing test features...")
            X_test, y_test = materialize_single(test_records, setup, args.max_test_windows_per_file, scaler, args)
            scaler_payload = {"scaler": scaler}
            per_channel_dim = 10 if str(setup["sensor_folder"]).lower() == "imu" else 8
            print(f"Feature dim: {int(setup['selected_channels']) * per_channel_dim}")
            print("Feature scaling: StandardScaler on handcrafted features before SVM.")

        print(f"Training matrix: {X_train.shape}")
        print(f"Test matrix: {X_test.shape}")
        estimated_fit_memory_gb = estimate_fit_memory_gb(X_train)
        print(f"Estimated sklearn fit matrix memory (float64): {estimated_fit_memory_gb:.2f} GiB")
        if estimated_fit_memory_gb > float(args.max_fit_memory_gb):
            raise MemoryError(
                "The training matrix is too large for sklearn SVM on this machine. "
                f"Estimated float64 matrix memory is {estimated_fit_memory_gb:.2f} GiB, "
                f"which exceeds --max-fit-memory-gb={args.max_fit_memory_gb}. "
                "Try lowering --max-train-windows-per-file, adding feature masks, "
                "or using a smaller subset first."
            )

        model = build_svm_model(args)
        train_start = perf_counter()
        model.fit(X_train, y_train)
        train_time_sec = perf_counter() - train_start
        importance_vector = extract_linear_feature_importance(model)
        if importance_vector is not None:
            fold_importances.append(importance_vector)
        inference_start = perf_counter()
        y_pred = model.predict(X_test)
        inference_time_sec = perf_counter() - inference_start

        metrics = compute_metrics(y_test, y_pred)
        print(
            f"Fold accuracy: {metrics['accuracy'] * 100:.2f}% | "
            f"weighted F1: {metrics['f1_weighted'] * 100:.2f}% | "
            f"macro F1: {metrics['f1_macro'] * 100:.2f}%"
        )
        print(f"Train time: {train_time_sec:.2f} s | Inference time: {inference_time_sec:.4f} s")

        fold_results.append(
            {
                "fold_index": fold_index,
                "subject_id": subject_id,
                "train_files": len(train_records),
                "test_files": len(test_records),
                "train_windows_used": int(X_train.shape[0]),
                "test_windows_used": int(X_test.shape[0]),
                "feature_dim": int(X_train.shape[1]),
                "accuracy_pct": metrics["accuracy"] * 100.0,
                "f1_weighted_pct": metrics["f1_weighted"] * 100.0,
                "f1_macro_pct": metrics["f1_macro"] * 100.0,
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "train_time_sec": train_time_sec,
                "inference_time_sec": inference_time_sec,
            }
        )
        all_preds.extend(y_pred.tolist())
        all_labels.extend(y_test.tolist())

        plot_confusion(
            y_test,
            y_pred,
            f"SVM Confusion Matrix - Fold {fold_index:02d} ({subject_id})",
            output_dir / f"confusion_matrix_{artifact_prefix}_fold_{fold_index:02d}_{subject_id}.png",
        )

        if args.save_fold_models:
            model_dir = output_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": model,
                "fold_index": fold_index,
                "subject_id": subject_id,
                "input_mode": args.input_mode,
                "label_names": LABEL_NAMES,
                "label_display_names": LABEL_DISPLAY_NAMES,
                **scaler_payload,
            }
            joblib.dump(payload, model_dir / f"{artifact_prefix}_fold_{fold_index:02d}_{subject_id}.joblib")

    fold_df = pd.DataFrame(fold_results).sort_values(["fold_index", "subject_id"]).reset_index(drop=True)
    fold_df.to_csv(output_dir / "svm_loso_fold_results.csv", index=False, encoding="utf-8-sig")

    y_true_all = np.asarray(all_labels, dtype=np.int64)
    y_pred_all = np.asarray(all_preds, dtype=np.int64)
    agg = compute_metrics(y_true_all, y_pred_all)
    summary = {
        "input_mode": args.input_mode,
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
        "mean_train_time_sec": float(fold_df["train_time_sec"].mean()),
        "mean_inference_time_sec": float(fold_df["inference_time_sec"].mean()),
    }
    (output_dir / "svm_loso_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "svm_loso_classification_report.txt").write_text(
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

    if fold_importances:
        importance_matrix = np.vstack(fold_importances)
        importance_df = pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": importance_matrix.mean(axis=0),
                "importance_std": importance_matrix.std(axis=0, ddof=0),
            }
        )
        feature_parts = importance_df["feature_name"].str.split(":", expand=True)
        importance_df["modality"] = feature_parts[0]
        importance_df["channel"] = feature_parts[1]
        importance_df["feature_type"] = feature_parts[2]
        importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df.to_csv(output_dir / "svm_feature_importance.csv", index=False, encoding="utf-8-sig")

        group_df = (
            importance_df.groupby(["modality", "feature_type"], as_index=False)["importance"]
            .sum()
            .assign(group=lambda df: df["modality"] + ":" + df["feature_type"])
        )
        group_df.to_csv(output_dir / "svm_feature_group_importance.csv", index=False, encoding="utf-8-sig")
        plot_top_feature_importance(importance_df, output_dir / "svm_top_feature_importance.png")
        plot_feature_group_importance(group_df, output_dir / "svm_feature_group_importance.png")
        print("\nTop 10 features by mean absolute linear weight:")
        for _, row in importance_df.head(10).iterrows():
            print(f"  {row['feature_name']}: {row['importance']:.6f}")
    else:
        print("\nFeature importance was skipped because the selected SVM backend does not expose linear coefficients.")

    plot_confusion(y_true_all, y_pred_all, "SVM Aggregate Confusion Matrix - LOSO", output_dir / f"confusion_matrix_{artifact_prefix}.png")
    plot_fold_accuracy(fold_df, str(setup["accuracy_title"]), output_dir / f"{artifact_prefix}_fold_accuracy.png")
    write_report(output_dir / "svm_loso_report.md", args, setup, summary, fold_df)

    print("\n" + "=" * 80)
    print("SVM LOSO complete")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Mean accuracy: {summary['mean_accuracy_pct']:.2f}% +/- {summary['std_accuracy_pct']:.2f}%")
    print(f"Mean weighted F1: {summary['mean_f1_weighted_pct']:.2f}% +/- {summary['std_f1_weighted_pct']:.2f}%")
    print(f"Aggregate accuracy over all held-out windows: {summary['aggregate_accuracy_pct']:.2f}%")
    print("Leakage check: scaler(s) fit only on training folds, and SVM re-instantiated inside every fold.")


if __name__ == "__main__":
    main()
