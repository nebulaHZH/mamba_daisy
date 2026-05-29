"""
Train a sequence Mamba classifier on real sensor time-series data.

This version avoids loading all sliding windows into memory at once.
Instead, it:
- scans files and builds file-level window metadata
- splits by source file to avoid leakage
- fits the scaler by streaming through training files
- reads and slices windows on demand during training
"""

import argparse
import json
import os
import re
import warnings
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from mamba_models import MambaSequenceClassifier

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

warnings.filterwarnings("ignore")

STRUCT_EXCLUDED_FIELDS = {"Header"}
CLASS_DEFINITIONS = [
    ("levelground", "levelground"),
    ("treadmill", "treadmill"),
    ("stair_ascent", "stair_up"),
    ("stair_descent", "stair_down"),
    ("ramp_ascent", "ramp_up"),
    ("ramp_descent", "ramp_down"),
]
CLASS_TO_ID = {label_name: idx for idx, (label_name, _) in enumerate(CLASS_DEFINITIONS)}
LABEL_NAMES = [label_name for label_name, _ in CLASS_DEFINITIONS]
LABEL_DISPLAY_NAMES = [display_name for _, display_name in CLASS_DEFINITIONS]
INDEX_CACHE_VERSION = 3
SENSOR_DEFAULT_SAMPLE_RATE = {
    "emg": 1000.0,
    "imu": 200.0,
}


@dataclass(frozen=True)
class FileRecord:
    file_path: str
    label_id: int
    label_name: str
    subject_id: str
    num_steps: int
    num_windows: int


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


set_seed(42, deterministic=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sequence Mamba model with streaming sliding windows."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to Camargo dataset root. Example: camargo数据集_emg_processed",
    )
    parser.add_argument(
        "--sensor-folder",
        type=str,
        default="emg",
        help="Only load MAT files under this sensor folder name (default: emg).",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=500.0,
        help="Sliding window length in milliseconds (default: 500).",
    )
    parser.add_argument(
        "--stride-ms",
        type=float,
        default=250.0,
        help="Sliding stride in milliseconds (default: 250).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=None,
        help="Sampling rate used to convert milliseconds to samples. Defaults to 1000 for EMG and 200 for IMU.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Window size in samples. Overrides --window-ms if provided.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride in samples. Overrides --stride-ms if provided.",
    )
    parser.add_argument(
        "--feature-mask",
        type=str,
        default=None,
        help='Comma-separated 0/1 mask for channels, e.g. "1,0,1,1,0,0,1,0,1,0,0".',
    )
    parser.add_argument(
        "--temporal-pooling",
        choices=("none", "mean", "rms"),
        default="none",
        help="Optional temporal pooling applied to each window before scaling/model input.",
    )
    parser.add_argument(
        "--temporal-pool-size",
        type=int,
        default=1,
        help="Number of consecutive time points pooled into one step (default: 1).",
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=None,
        help="Optional expected sensor channel count. Raise an error if loaded data does not match.",
    )
    parser.add_argument(
        "--max-files-per-class",
        type=int,
        default=None,
        help="Limit files per class for quick experiments.",
    )
    parser.add_argument(
        "--stair-up-token",
        choices=("l", "r"),
        default="l",
        help=(
            "Map stair_<index>_<token>_* filenames to stair ascent. "
            "The opposite token is treated as stair descent. Default: l."
        ),
    )
    parser.add_argument(
        "--ramp-up-token",
        choices=("l", "r"),
        default="l",
        help=(
            "Map ramp_<index>_<token>_* filenames to ramp ascent. "
            "The opposite token is treated as ramp descent. Default: l."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training (default: 128).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        choices=tuple(range(5, 9)),
        default=5,
        help="Early stopping patience on validation loss (choices: 5-8, default: 5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Training device: "auto", "cpu", "cuda", or "cuda:0" (default: auto).',
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit instead of falling back to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count. Defaults to 0 on Windows and 2 on other platforms.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=128,
        help="Number of processed files to keep in the per-dataset cache (default: 128).",
    )
    parser.add_argument(
        "--max-train-windows-per-file",
        type=int,
        default=64,
        help="Cap training windows per file to keep each epoch tractable (default: 64).",
    )
    parser.add_argument(
        "--max-val-windows-per-file",
        type=int,
        default=64,
        help="Cap validation windows per file for faster early stopping (default: 64).",
    )
    parser.add_argument(
        "--cv-mode",
        choices=("loso", "kfold"),
        default="loso",
        help="Cross-validation mode: LOSO or subject-grouped K-fold.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of subject-grouped folds when --cv-mode kfold is used.",
    )
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=42,
        help="Random seed for assigning subjects to K-fold groups.",
    )
    parser.add_argument(
        "--index-cache",
        type=str,
        default=None,
        help="Optional path to file-index cache JSON. Defaults to a generated cache under .cache/.",
    )
    parser.add_argument(
        "--rebuild-index-cache",
        action="store_true",
        help="Force rebuilding the file-index cache instead of reusing an existing one.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional output prefix for checkpoint and plots. Defaults to sensor-specific names.",
    )
    return parser.parse_args()


def parse_feature_mask(mask_str, expected_dim):
    """Parse a comma-separated 0/1 mask into numpy array."""
    if mask_str is None:
        return None

    mask_values = [item.strip() for item in mask_str.split(",") if item.strip()]
    try:
        mask = np.array([int(v) for v in mask_values], dtype=np.int64)
    except ValueError as exc:
        raise ValueError(f"feature-mask contains non-integer value(s): {mask_str}") from exc

    if mask.size != expected_dim:
        raise ValueError(
            f"feature-mask length {mask.size} does not match channel count {expected_dim}."
        )
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("feature-mask must contain only 0 or 1.")
    if mask.sum() == 0:
        raise ValueError("feature-mask cannot be all zeros.")

    return mask


def resolve_temporal_pooling(pooling, pool_size, window_size):
    """Validate temporal pooling settings and return the output sequence length."""
    pooling = str(pooling or "none").lower()
    pool_size = max(1, int(pool_size))
    window_size = int(window_size)

    if pooling == "none":
        return pooling, 1, window_size
    if pool_size <= 1:
        raise ValueError("--temporal-pool-size must be greater than 1 when pooling is enabled.")
    if window_size % pool_size != 0:
        raise ValueError(
            f"window_size={window_size} must be divisible by temporal_pool_size={pool_size} "
            "so every pooled step covers the same duration."
        )
    return pooling, pool_size, window_size // pool_size


def ms_to_samples(duration_ms, sample_rate_hz):
    """Convert milliseconds to sample count."""
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be positive, got {duration_ms}")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    return max(1, int(round(duration_ms * sample_rate_hz / 1000.0)))


def count_windows(num_steps, window_size, stride):
    """Count how many fixed windows can be extracted from a signal."""
    if num_steps < window_size:
        return 0
    return 1 + (num_steps - window_size) // stride


def apply_feature_mask(signal, mask):
    """Apply channel mask to a 2D signal array shaped as (time, channels)."""
    return signal[:, mask.astype(bool)]


def get_channel_names_from_data_struct(data_struct):
    """Infer EMG channel names from MATLAB struct fields."""
    dtype_names = getattr(getattr(data_struct, "dtype", None), "names", None)
    if not dtype_names:
        raise ValueError("data_struct does not expose field names.")
    return [name for name in dtype_names if name not in STRUCT_EXCLUDED_FIELDS]


def extract_subject_id(file_path):
    """Extract subject identifier like AB06 from the file path."""
    for part in Path(file_path).parts:
        if part.upper().startswith("AB"):
            return part
    return "UNKNOWN"


def extract_label_from_path(file_path, stair_up_token="l", ramp_up_token="l"):
    """Extract 6-class motion label from the file path and file name."""
    normalized_path = str(file_path).replace("\\", "/").lower()
    file_name = Path(file_path).name.lower()

    if "/levelground/" in normalized_path or file_name.startswith("levelground"):
        return CLASS_TO_ID["levelground"], "levelground"

    if "/treadmill/" in normalized_path or file_name.startswith("treadmill"):
        return CLASS_TO_ID["treadmill"], "treadmill"

    stair_match = re.search(r"stair_\d+_([lr])_", file_name)
    if stair_match:
        stair_direction = stair_match.group(1)
        label_name = "stair_ascent" if stair_direction == stair_up_token.lower() else "stair_descent"
        return CLASS_TO_ID[label_name], label_name

    ramp_match = re.search(r"ramp_\d+_([lr])_", file_name)
    if ramp_match:
        ramp_direction = ramp_match.group(1)
        label_name = "ramp_ascent" if ramp_direction == ramp_up_token.lower() else "ramp_descent"
        return CLASS_TO_ID[label_name], label_name

    return None, None


def _extract_signal_from_data_struct(data_struct, channel_names):
    """Extract signal from classic MATLAB struct format."""
    time_steps = len(data_struct)
    num_channels = len(channel_names)
    signal = np.zeros((time_steps, num_channels), dtype=np.float32)

    for ch_idx, channel in enumerate(channel_names):
        if channel in data_struct.dtype.names:
            for t in range(time_steps):
                signal[t, ch_idx] = data_struct[channel][t, 0][0, 0]
    return signal


def _safe_string(cell):
    """Convert MATLAB cell/string metadata to a Python string."""
    array = np.asarray(cell)
    if array.size == 0:
        return ""
    value = array.squeeze()
    if isinstance(value, np.ndarray):
        return str(value.tolist())
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    return str(value)


def _extract_channel_names_from_metadata(mat_data):
    """Extract saved channel names from processed MAT metadata if available."""
    for key in ("channel_names", "col_names"):
        value = mat_data.get(key)
        if isinstance(value, np.ndarray) and value.size > 0:
            names = [_safe_string(item) for item in value.reshape(-1)]
            names = [name for name in names if name and name not in STRUCT_EXCLUDED_FIELDS]
            if names:
                return names
    return None


def _extract_named_numeric_matrix(mat_data):
    """Prefer processed sensor matrices over generic numeric metadata."""
    preferred_keys = (
        "normalized_emg",
        "normalized_imu",
        "envelope_emg",
        "filtered_imu",
        "rectified_emg",
        "raw_emg",
        "raw_imu",
    )
    for key in preferred_keys:
        value = mat_data.get(key)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            if value.ndim == 1:
                return value.reshape(-1, 1).astype(np.float32), key
            if value.ndim == 2:
                matrix = value.astype(np.float32)
                if matrix.shape[0] <= 64 and matrix.shape[1] > matrix.shape[0]:
                    matrix = matrix.T
                return matrix, key
    return None, None


def _extract_first_numeric_matrix(mat_data):
    """Fallback: extract first 2D numeric matrix variable from MAT."""
    for key, value in mat_data.items():
        if key.startswith("__") or key in {
            "sample_rate_hz",
            "lowpass_cutoff_hz",
            "lowpass_order",
        }:
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            if value.ndim == 2:
                matrix = value.astype(np.float32)
                if matrix.shape[0] <= 64 and matrix.shape[1] > matrix.shape[0]:
                    matrix = matrix.T
                return matrix
            if value.ndim == 1:
                return value.reshape(-1, 1).astype(np.float32)
    return None


def load_mat_file(file_path):
    """Load one MAT file and extract sensor time-series data."""
    try:
        mat_data = sio.loadmat(file_path)
        if "data_struct" in mat_data:
            channel_names = get_channel_names_from_data_struct(mat_data["data_struct"])
            signal = _extract_signal_from_data_struct(mat_data["data_struct"], channel_names)
            return signal, channel_names, None

        named_matrix, _ = _extract_named_numeric_matrix(mat_data)
        if named_matrix is not None:
            channel_names = _extract_channel_names_from_metadata(mat_data)
            if channel_names is None:
                channel_names = [f"channel_{idx + 1}" for idx in range(named_matrix.shape[1])]
            return named_matrix, channel_names, None

        numeric_matrix = _extract_first_numeric_matrix(mat_data)
        if numeric_matrix is not None:
            channel_names = _extract_channel_names_from_metadata(mat_data)
            if channel_names is None:
                channel_names = [f"channel_{idx + 1}" for idx in range(numeric_matrix.shape[1])]
            return numeric_matrix, channel_names, None

        visible_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        if "None" in visible_keys:
            return (
                None,
                None,
                (
                    "MATLAB table object detected (MatlabOpaque key 'None'). "
                    "scipy cannot decode this format directly. "
                    "Please export each table to numeric array/struct first."
                ),
            )
        return None, None, f"No supported numeric data found. Variables: {visible_keys}"

    except Exception as exc:
        return None, None, f"Failed to load file: {exc}"


def build_candidate_files(
    data_root,
    sensor_folder,
    max_files_per_class=None,
    stair_up_token="l",
    ramp_up_token="l",
):
    """Discover candidate MAT files grouped by class."""
    mat_pattern = os.path.join(data_root, "**", sensor_folder, "*.mat")
    mat_files = sorted(Path(data_root).glob(f"**/{sensor_folder}/*.mat"))
    if len(mat_files) == 0:
        raise ValueError(
            f"No MAT files found under: {mat_pattern}. "
            "Please check the dataset path."
        )

    class_files = {label_name: [] for label_name in CLASS_TO_ID}
    for file_path in mat_files:
        _, label_name = extract_label_from_path(
            file_path,
            stair_up_token=stair_up_token,
            ramp_up_token=ramp_up_token,
        )
        if label_name:
            class_files[label_name].append(str(file_path))

    selected_files = []
    for label_name, files in class_files.items():
        if max_files_per_class:
            files = files[:max_files_per_class]
        selected_files.extend(files)

    selected_files = sorted(selected_files)
    return mat_pattern, mat_files, class_files, selected_files


def get_cache_file_signature(file_paths):
    """Create a cheap file signature from path, size, and mtime."""
    signature = []
    for file_path in file_paths:
        stat = Path(file_path).stat()
        signature.append(
            {
                "file_path": file_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return signature


def resolve_index_cache_path(data_root, sensor_folder, window_size, stride, max_files_per_class, explicit_path=None):
    """Resolve the on-disk cache path for file-level window metadata."""
    if explicit_path:
        return Path(explicit_path)

    cache_dir = Path.cwd() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_files_tag = "all" if max_files_per_class is None else f"mf{max_files_per_class}"
    root_name = Path(data_root).name or "dataset"
    file_name = f"{root_name}_{sensor_folder}_ws{window_size}_st{stride}_{max_files_tag}_index.json"
    return cache_dir / file_name


def load_index_cache(cache_path):
    """Load cached file metadata if the cache file exists."""
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_index_cache(cache_path, payload):
    """Persist file metadata cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def rebuild_file_records(
    class_files,
    selected_files,
    window_size,
    stride,
    expected_channels=None,
):
    """Load selected MAT files and compute file-level window metadata."""
    records = []
    load_errors = []
    channel_names = None
    subject_ids = set()

    selected_set = set(selected_files)
    print("\nInspecting files and building window metadata...")
    for label_name, files in class_files.items():
        label_id = CLASS_TO_ID[label_name]
        files = [file_path for file_path in files if file_path in selected_set]

        for file_path in tqdm(files, desc=f"Indexing {label_name}"):
            subject_ids.add(extract_subject_id(file_path))

            signal, current_channel_names, err = load_mat_file(file_path)
            if err is not None:
                load_errors.append((file_path, err))
                continue

            if channel_names is None:
                channel_names = current_channel_names
                print(
                    f"Detected {len(channel_names)} sensor channels: "
                    f"{', '.join(channel_names[:6])}{' ...' if len(channel_names) > 6 else ''}"
                )
            elif signal.shape[1] != len(channel_names):
                load_errors.append(
                    (
                        file_path,
                        f"Inconsistent channel count {signal.shape[1]} (expected {len(channel_names)})",
                    )
                )
                continue

            if expected_channels is not None and signal.shape[1] != expected_channels:
                raise ValueError(
                    f"Loaded data has {signal.shape[1]} channels, but --expected-channels={expected_channels}. "
                    "Please confirm the dataset channel count."
                )

            num_windows = count_windows(len(signal), window_size, stride)
            if num_windows == 0:
                load_errors.append(
                    (file_path, f"Signal too short ({len(signal)}), window_size={window_size}")
                )
                continue

            records.append(
                FileRecord(
                    file_path=file_path,
                    label_id=label_id,
                    label_name=label_name,
                    subject_id=extract_subject_id(file_path),
                    num_steps=len(signal),
                    num_windows=num_windows,
                )
            )

    return records, channel_names, sorted(subject_ids), load_errors


def load_dataset(
    data_root,
    window_size,
    stride,
    max_files_per_class=None,
    sensor_folder="emg",
    expected_channels=None,
    cache_path=None,
    rebuild_index_cache=False,
    stair_up_token="l",
    ramp_up_token="l",
):
    """
    Scan dataset files and build file-level window metadata.

    This function does not materialize all windows in memory.
    """
    print("Scanning MAT files...")
    mat_pattern, mat_files, class_files, selected_files = build_candidate_files(
        data_root,
        sensor_folder,
        max_files_per_class=max_files_per_class,
        stair_up_token=stair_up_token,
        ramp_up_token=ramp_up_token,
    )
    print(f"Found {len(mat_files)} MAT files in '{sensor_folder}' folders")

    class_files = {label_name: [] for label_name in CLASS_TO_ID}
    for file_path in mat_files:
        _, label_name = extract_label_from_path(
            file_path,
            stair_up_token=stair_up_token,
            ramp_up_token=ramp_up_token,
        )
        if label_name:
            class_files[label_name].append(str(file_path))

    print("\nFiles per class:")
    for cls, files in class_files.items():
        print(f"  {cls}: {len(files)} files")

    cache_path = resolve_index_cache_path(
        data_root,
        sensor_folder,
        window_size,
        stride,
        max_files_per_class,
        explicit_path=cache_path,
    )
    file_signature = get_cache_file_signature(selected_files)
    cache_payload = None if rebuild_index_cache else load_index_cache(cache_path)

    if (
        cache_payload
        and cache_payload.get("version") == INDEX_CACHE_VERSION
        and cache_payload.get("data_root") == str(Path(data_root).resolve())
        and cache_payload.get("sensor_folder") == sensor_folder
        and cache_payload.get("window_size") == int(window_size)
        and cache_payload.get("stride") == int(stride)
        and cache_payload.get("max_files_per_class") == max_files_per_class
        and cache_payload.get("expected_channels") == expected_channels
        and cache_payload.get("stair_up_token") == stair_up_token
        and cache_payload.get("ramp_up_token") == ramp_up_token
        and cache_payload.get("file_signature") == file_signature
    ):
        print(f"\nLoading file index cache: {cache_path}")
        records = [FileRecord(**record) for record in cache_payload["records"]]
        channel_names = cache_payload["channel_names"]
        subject_ids = cache_payload["subject_ids"]
        load_errors = cache_payload.get("load_errors", [])
    else:
        if rebuild_index_cache:
            print(f"\nRebuilding file index cache: {cache_path}")
        else:
            print(f"\nCache miss or stale cache. Building file index: {cache_path}")
        records, channel_names, subject_ids, load_errors = rebuild_file_records(
            class_files,
            selected_files,
            window_size,
            stride,
            expected_channels=expected_channels,
        )
        save_index_cache(
            cache_path,
            {
                "version": INDEX_CACHE_VERSION,
                "data_root": str(Path(data_root).resolve()),
                "sensor_folder": sensor_folder,
                "window_size": int(window_size),
                "stride": int(stride),
                "max_files_per_class": max_files_per_class,
                "expected_channels": expected_channels,
                "stair_up_token": stair_up_token,
                "ramp_up_token": ramp_up_token,
                "file_signature": file_signature,
                "channel_names": channel_names,
                "subject_ids": subject_ids,
                "records": [asdict(record) for record in records],
                "load_errors": load_errors,
            },
        )
        print(f"Saved file index cache: {cache_path}")

    if load_errors:
        print(f"\nSkipped files: {len(load_errors)}")
        for idx, (path, reason) in enumerate(load_errors[:8], 1):
            print(f"  {idx}. {path} -> {reason}")
        if len(load_errors) > 8:
            print(f"  ... and {len(load_errors) - 8} more")

    if not records:
        raise ValueError(
            "No valid files remain after indexing. "
            "Possible reasons: signals are shorter than window_size, labels are not matched, "
            "or MAT format is unsupported."
        )

    total_windows = sum(record.num_windows for record in records)
    print(f"\nTotal valid files: {len(records)}")
    print(f"Total windows available: {total_windows}")
    print(f"Subjects included: {len(subject_ids)} -> {', '.join(sorted(subject_ids))}")

    print("\nWindows per class:")
    for label_name in LABEL_NAMES:
        class_windows = sum(record.num_windows for record in records if record.label_name == label_name)
        class_files_count = sum(1 for record in records if record.label_name == label_name)
        print(f"  {label_name}: {class_windows} windows from {class_files_count} files")

    return records, channel_names, sorted(subject_ids)


def resolve_default_data_root(sensor_folder="emg"):
    """Resolve local Camargo dataset root based on the requested sensor folder."""
    sensor_folder = str(sensor_folder).lower()
    if sensor_folder == "imu":
        candidates = [
            Path.cwd() / "camargo数据集_imu_processed",
            Path.cwd() / "camargo数据集",
            Path.cwd() / "camargo_dataset",
            Path.cwd() / "camargo数据集_imu_processed" / "AB10" / "10_28_2018",
            Path.cwd() / "camargo数据集" / "AB10" / "10_28_2018",
            Path.cwd() / "camargo_dataset" / "AB10" / "10_28_2018",
        ]
    else:
        candidates = [
            Path.cwd() / "camargo数据集_emg_processed",
            Path.cwd() / "camargo数据集",
            Path.cwd() / "camargo_dataset",
            Path.cwd() / "camargo数据集_emg_processed" / "AB10" / "10_28_2018",
            Path.cwd() / "camargo数据集" / "AB10" / "10_28_2018",
            Path.cwd() / "camargo_dataset" / "AB10" / "10_28_2018",
        ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return r"E:\0_yao\dataset\carmago_struct"


def resolve_sample_rate_hz(requested_sample_rate_hz, sensor_folder):
    """Resolve the sampling rate for a given sensor folder."""
    if requested_sample_rate_hz is not None:
        return float(requested_sample_rate_hz)
    return SENSOR_DEFAULT_SAMPLE_RATE.get(str(sensor_folder).lower(), 1000.0)


def resolve_output_paths(sensor_folder, explicit_prefix=None):
    """Resolve sensor-specific output file names."""
    if explicit_prefix:
        prefix = explicit_prefix
    elif str(sensor_folder).lower() == "imu":
        prefix = "imu_seq"
    else:
        prefix = "seq"

    if prefix.endswith(".pth"):
        prefix = prefix[:-4]

    checkpoint_path = f"best_model_{prefix}.pth"
    history_path = f"training_history_{prefix}.png"
    confusion_matrix_path = f"confusion_matrix_{prefix}.png"
    return checkpoint_path, history_path, confusion_matrix_path


def resolve_num_workers(requested_workers):
    """Pick a conservative DataLoader worker count when the user does not specify one."""
    if requested_workers is not None:
        return max(0, int(requested_workers))
    if os.name == "nt":
        return 0
    return 2


def resolve_training_device(requested_device="auto", require_cuda=False):
    """Resolve and validate the torch device used for training."""
    requested_device = str(requested_device or "auto").strip().lower()
    cuda_available = torch.cuda.is_available()

    if requested_device == "auto":
        device = torch.device("cuda:0" if cuda_available else "cpu")
    else:
        device = torch.device(requested_device)

    if device.type == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested, but this Python environment cannot use CUDA. "
                "Install a CUDA-enabled PyTorch build in the active environment."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
        torch.cuda.set_device(device)
    elif require_cuda:
        raise RuntimeError(
            "CUDA is required by --require-cuda, but the resolved training device is CPU. "
            "Use --device cuda:0 with a CUDA-enabled PyTorch environment."
        )

    return device


def describe_training_device(device):
    if device.type != "cuda":
        return "cpu"
    current_index = torch.cuda.current_device() if device.index is None else device.index
    device_name = torch.cuda.get_device_name(current_index)
    cuda_version = torch.version.cuda or "unknown"
    return f"cuda:{current_index} ({device_name}, CUDA {cuda_version})"


def split_file_records(records, test_size=0.2, val_size=0.1, random_state=42):
    """Split file records into train/val/test at file level with stratification."""
    labels = [record.label_id for record in records]
    unique_labels, counts = np.unique(labels, return_counts=True)
    stratify_labels = labels if np.all(counts >= 2) and len(unique_labels) > 1 else None
    train_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    train_labels = [record.label_id for record in train_records]
    train_unique_labels, train_counts = np.unique(train_labels, return_counts=True)
    train_stratify_labels = (
        train_labels if np.all(train_counts >= 2) and len(train_unique_labels) > 1 else None
    )
    val_ratio_within_train = val_size / (1.0 - test_size)
    train_records, val_records = train_test_split(
        train_records,
        test_size=val_ratio_within_train,
        random_state=random_state,
        stratify=train_stratify_labels,
    )
    return train_records, val_records, test_records


class SequenceWindowDataset(Dataset):
    """File-backed dataset that slices sliding windows on demand."""

    def __init__(
        self,
        file_records,
        window_size,
        stride,
        scaler=None,
        fit_scaler=False,
        feature_mask=None,
        cache_size=32,
        max_windows_per_file=None,
        temporal_pooling="none",
        temporal_pool_size=1,
    ):
        self.file_records = list(file_records)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.feature_mask = feature_mask
        self.cache_size = max(1, int(cache_size))
        (
            self.temporal_pooling,
            self.temporal_pool_size,
            self.output_window_size,
        ) = resolve_temporal_pooling(temporal_pooling, temporal_pool_size, self.window_size)
        self.max_windows_per_file = (
            None if max_windows_per_file is None else max(1, int(max_windows_per_file))
        )
        self._signal_cache = OrderedDict()

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
            self.scaler = scaler if scaler is not None else StandardScaler()
            self._fit_scaler()
        else:
            self.scaler = scaler

    def _fit_scaler(self):
        print("\nFitting scaler on training files...")
        for record_idx, record in enumerate(tqdm(self.file_records, desc="Scaler")):
            signal = self._load_signal(record.file_path, apply_scaler=False)
            if self.temporal_pooling == "none":
                self.scaler.partial_fit(signal)
                continue

            window_indices = self.sampled_window_indices[record_idx]
            if window_indices is None:
                window_indices = range(record.num_windows)

            pooled_windows = []
            for local_window_idx in window_indices:
                start = int(local_window_idx) * self.stride
                end = start + self.window_size
                pooled_windows.append(self._apply_temporal_pooling(signal[start:end]))
            if pooled_windows:
                self.scaler.partial_fit(np.vstack(pooled_windows))

    def _build_window_indices(self, num_windows):
        if self.max_windows_per_file is None or num_windows <= self.max_windows_per_file:
            return None

        positions = (
            (np.arange(self.max_windows_per_file, dtype=np.float64) + 0.5)
            * num_windows
            / self.max_windows_per_file
        )
        return np.clip(positions.astype(np.int64), 0, num_windows - 1)

    def _evict_if_needed(self):
        while len(self._signal_cache) > self.cache_size:
            self._signal_cache.popitem(last=False)

    def _load_signal(self, file_path, apply_scaler=True):
        signal, _, err = load_mat_file(file_path)
        if err is not None:
            raise ValueError(f"Failed to reload file during dataset access: {file_path} -> {err}")

        if self.feature_mask is not None:
            signal = apply_feature_mask(signal, self.feature_mask)

        if self.temporal_pooling == "none" and apply_scaler and self.scaler is not None:
            signal = self.scaler.transform(signal)

        return signal.astype(np.float32, copy=False)

    def _apply_temporal_pooling(self, window):
        if self.temporal_pooling == "none":
            return window

        pooled_shape = (
            self.output_window_size,
            self.temporal_pool_size,
            window.shape[1],
        )
        grouped = window.reshape(pooled_shape)
        if self.temporal_pooling == "mean":
            return grouped.mean(axis=1)
        if self.temporal_pooling == "rms":
            return np.sqrt(np.mean(np.square(grouped), axis=1))
        raise ValueError(f"Unsupported temporal pooling mode: {self.temporal_pooling}")

    def _get_cached_signal(self, file_path):
        if file_path in self._signal_cache:
            signal = self._signal_cache.pop(file_path)
            self._signal_cache[file_path] = signal
            return signal

        signal = self._load_signal(file_path, apply_scaler=True)
        self._signal_cache[file_path] = signal
        self._evict_if_needed()
        return signal

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
        signal = self._get_cached_signal(record.file_path)

        sampled_window_indices = self.sampled_window_indices[record_idx]
        if sampled_window_indices is not None:
            local_window_idx = int(sampled_window_indices[local_window_idx])

        start = local_window_idx * self.stride
        end = start + self.window_size
        window = signal[start:end]
        window = self._apply_temporal_pooling(window)
        if self.temporal_pooling != "none" and self.scaler is not None:
            window = self.scaler.transform(window)

        return torch.from_numpy(window.astype(np.float32, copy=True)), record.label_id


class FileAwareBatchSampler(Sampler):
    """Shuffle file order each epoch while keeping windows from the same file contiguous."""

    def __init__(self, dataset, batch_size, shuffle_files=True, drop_last=False):
        self.cumulative_windows = list(dataset.cumulative_windows)
        self.batch_size = max(1, int(batch_size))
        self.shuffle_files = shuffle_files
        self.drop_last = drop_last

    def __iter__(self):
        file_order = np.arange(len(self.cumulative_windows))
        if self.shuffle_files:
            np.random.shuffle(file_order)

        batch = []
        for record_idx in file_order:
            start = 0 if record_idx == 0 else self.cumulative_windows[record_idx - 1]
            end = self.cumulative_windows[record_idx]

            for index in range(start, end):
                batch.append(index)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total_windows = self.cumulative_windows[-1] if self.cumulative_windows else 0
        if self.drop_last:
            return total_windows // self.batch_size
        return (total_windows + self.batch_size - 1) // self.batch_size


def summarize_split(split_name, file_records):
    """Print file-level and window-level counts for a split."""
    total_files = len(file_records)
    total_windows = sum(record.num_windows for record in file_records)
    subjects = sorted({record.subject_id for record in file_records})

    print(f"\n{split_name} split:")
    print(f"  files: {total_files}")
    print(f"  windows: {total_windows}")
    print(f"  subjects: {len(subjects)}")
    for label_name in LABEL_NAMES:
        label_files = sum(1 for record in file_records if record.label_name == label_name)
        label_windows = sum(record.num_windows for record in file_records if record.label_name == label_name)
        print(f"  {label_name}: {label_files} files, {label_windows} windows")


def summarize_epoch_windows(split_name, dataset):
    """Print how many windows are actually used by the dataset each epoch."""
    original_windows = sum(record.num_windows for record in dataset.file_records)
    used_windows = len(dataset)

    if original_windows == 0:
        print(f"{split_name} windows used per epoch: 0")
        return

    if used_windows == original_windows:
        print(f"{split_name} windows used per epoch: {used_windows}")
        return

    ratio = 100.0 * used_windows / original_windows
    print(
        f"{split_name} windows used per epoch: {used_windows} / {original_windows} "
        f"({ratio:.2f}%)"
    )


class EarlyStopping:
    """Stop training when validation loss stops improving and keep the best weights."""

    def __init__(self, patience, checkpoint_path, min_delta=0.0):
        self.patience = max(1, int(patience))
        self.checkpoint_path = str(checkpoint_path)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.counter = 0

    def step(self, val_loss, epoch, payload):
        current_loss = float(val_loss)
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.best_epoch = int(epoch)
            self.counter = 0
            torch.save(payload, self.checkpoint_path)
            return True, False

        self.counter += 1
        return False, self.counter >= self.patience

    def restore_best_weights(self, model, map_location="cpu"):
        checkpoint = torch.load(self.checkpoint_path, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint


def build_artifact_path(base_path, suffix):
    """Create a sibling artifact path with a suffix before the extension."""
    path = Path(base_path)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def build_leave_one_subject_out_folds(file_records):
    """Build leave-one-subject-out folds."""
    subjects = sorted({record.subject_id for record in file_records})
    folds = []
    for subject_id in subjects:
        train_records = [record for record in file_records if record.subject_id != subject_id]
        val_records = [record for record in file_records if record.subject_id == subject_id]
        if train_records and val_records:
            folds.append((subject_id, train_records, val_records))
    return folds


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


def summarize_cv_results(fold_results):
    """Print aggregate cross-validation metrics."""
    if not fold_results:
        print("No completed folds.")
        return

    val_losses = [result["val_loss"] for result in fold_results]
    val_accs = [result["val_acc"] for result in fold_results]
    best_epochs = [result["best_epoch"] + 1 for result in fold_results]
    print("\nCross-validation summary:")
    print(f"  folds: {len(fold_results)}")
    print(f"  mean val loss: {np.mean(val_losses):.4f} +/- {np.std(val_losses):.4f}")
    print(f"  mean val acc: {np.mean(val_accs):.2f}% +/- {np.std(val_accs):.2f}%")
    print(f"  mean best epoch: {np.mean(best_epochs):.2f}")


def train_epoch(model, dataloader, criterion, optimizer, device, amp_enabled=False, grad_scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train")
    for sequences, labels in pbar:
        sequences = sequences.to(device, non_blocking=amp_enabled)
        labels = labels.to(device, non_blocking=amp_enabled)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(sequences)
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
        for sequences, labels in dataloader:
            sequences = sequences.to(device, non_blocking=amp_enabled)
            labels = labels.to(device, non_blocking=amp_enabled)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(sequences)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), 100.0 * correct / total, all_preds, all_labels


def plot_training_history(history, save_path="training_history_seq.png"):
    if plt is None:
        print("Skip plotting training history: matplotlib is not installed.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(history["val_acc"], label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training history saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path="confusion_matrix_seq.png"):
    if plt is None or sns is None:
        print("Skip plotting confusion matrix: matplotlib/seaborn is not installed.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Sequence Mamba")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def main():
    args = parse_args()
    resolved_sensor_folder = str(args.sensor_folder).lower()
    default_data_root = resolve_default_data_root(resolved_sensor_folder)
    sample_rate_hz = resolve_sample_rate_hz(args.sample_rate_hz, resolved_sensor_folder)
    window_size = args.window_size if args.window_size is not None else ms_to_samples(args.window_ms, sample_rate_hz)
    stride = args.stride if args.stride is not None else ms_to_samples(args.stride_ms, sample_rate_hz)
    temporal_pooling, temporal_pool_size, model_window_size = resolve_temporal_pooling(
        args.temporal_pooling,
        args.temporal_pool_size,
        window_size,
    )
    resolved_num_workers = resolve_num_workers(args.num_workers)
    checkpoint_path, history_path, confusion_matrix_path = resolve_output_paths(
        resolved_sensor_folder,
        explicit_prefix=args.output_prefix,
    )

    config = {
        "data_root": args.data_root if args.data_root else default_data_root,
        "sensor_folder": resolved_sensor_folder,
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "sample_rate_hz": sample_rate_hz,
        "window_size": window_size,
        "stride": stride,
        "temporal_pooling": temporal_pooling,
        "temporal_pool_size": temporal_pool_size,
        "model_window_size": model_window_size,
        "num_classes": len(CLASS_TO_ID),
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "d_model": 64,
        "n_layers": 2,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dropout": 0.2,
        "max_files_per_class": args.max_files_per_class,
        "max_train_windows_per_file": args.max_train_windows_per_file,
        "max_val_windows_per_file": args.max_val_windows_per_file,
        "cv_mode": args.cv_mode,
        "num_folds": args.num_folds,
        "fold_seed": args.fold_seed,
        "feature_mask": args.feature_mask,
        "expected_channels": args.expected_channels,
        "stair_up_token": args.stair_up_token,
        "ramp_up_token": args.ramp_up_token,
        "num_workers": resolved_num_workers,
        "cache_size": args.cache_size,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "confusion_matrix_path": confusion_matrix_path,
        "index_cache": str(
            resolve_index_cache_path(
                args.data_root if args.data_root else default_data_root,
                resolved_sensor_folder,
                window_size,
                stride,
                args.max_files_per_class,
                explicit_path=args.index_cache,
            )
        ),
    }

    print("=" * 60)
    print("Mamba training on real sequence data")
    print(f"Dataset: Camargo {config['sensor_folder'].upper()}")
    print("=" * 60)

    device = resolve_training_device(args.device, args.require_cuda)
    amp_enabled = device.type == "cuda"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    print(f"\nUsing device: {describe_training_device(device)}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"DataLoader workers: {config['num_workers']}")
    print(f"Data root: {config['data_root']}")
    print(
        f"Direction mapping: stair {config['stair_up_token']}=up, "
        f"ramp {config['ramp_up_token']}=up"
    )
    print(
        f"Window setting: {config['window_ms']} ms / {config['stride_ms']} ms "
        f"-> {config['window_size']} samples / {config['stride']} samples "
        f"(sample rate: {config['sample_rate_hz']} Hz)"
    )
    if config["temporal_pooling"] != "none":
        print(
            f"Temporal pooling: {config['temporal_pooling']} x{config['temporal_pool_size']} "
            f"-> model sequence length {config['model_window_size']}"
        )
    else:
        print("Temporal pooling: none")
    print(f"Early stopping patience: {config['patience']} (monitor: val_loss)")
    print(f"Checkpoint path: {config['checkpoint_path']}")

    file_records, detected_channel_names, subject_ids = load_dataset(
        config["data_root"],
        window_size=config["window_size"],
        stride=config["stride"],
        max_files_per_class=config["max_files_per_class"],
        sensor_folder=config["sensor_folder"],
        expected_channels=config["expected_channels"],
        cache_path=config["index_cache"],
        rebuild_index_cache=args.rebuild_index_cache,
        stair_up_token=config["stair_up_token"],
        ramp_up_token=config["ramp_up_token"],
    )

    original_channels = len(detected_channel_names)
    feature_mask = parse_feature_mask(config["feature_mask"], original_channels)
    selected_feature_indices = list(range(original_channels))
    if feature_mask is not None:
        selected_feature_indices = np.where(feature_mask == 1)[0].tolist()
        print(
            f"\nFeature mask applied: keep {len(selected_feature_indices)}/{original_channels} channels"
        )
        print(f"Selected channel indices: {selected_feature_indices}")
    else:
        print(f"\nNo feature mask specified, using all {original_channels} channels")

    print(f"Channel names: {', '.join(detected_channel_names)}")
    print(f"All detected subjects: {len(subject_ids)}")
    print("Classes: " + ", ".join(f"{idx}={name}" for idx, name in enumerate(LABEL_DISPLAY_NAMES)))

    loader_kwargs = {
        "num_workers": config["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    if config["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    input_channels = len(selected_feature_indices)
    num_classes = config["num_classes"]
    print(f"Input shape: (batch, {config['model_window_size']}, {input_channels})")

    if config["cv_mode"] == "loso":
        cv_folds = build_leave_one_subject_out_folds(file_records)
        cv_label = "LOSO"
        print(f"\nLeave-one-subject-out folds: {len(cv_folds)}")
        if not cv_folds:
            raise ValueError(
                "Leave-one-subject-out cross-validation requires records from at least two subjects. "
                "Please remove overly strict file limits such as --max-files-per-class."
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

    print("\n" + "=" * 60)
    print(f"Start {cv_label} cross-validation...")
    print("=" * 60)

    all_cv_preds = []
    all_cv_labels = []
    fold_results = []
    best_overall_fold = None

    for fold_index, (val_subject_id, train_records, val_records) in enumerate(cv_folds, start=1):
        print("\n" + "=" * 60)
        val_subjects = sorted({record.subject_id for record in val_records})
        print(f"Fold {fold_index}/{len(cv_folds)} - Validation group: {val_subject_id}")
        print(f"Validation subjects: {', '.join(val_subjects)}")
        print("=" * 60)

        summarize_split("Train", train_records)
        summarize_split("Val", val_records)

        train_dataset = SequenceWindowDataset(
            train_records,
            window_size=config["window_size"],
            stride=config["stride"],
            fit_scaler=True,
            feature_mask=feature_mask,
            cache_size=config["cache_size"],
            max_windows_per_file=config["max_train_windows_per_file"],
            temporal_pooling=config["temporal_pooling"],
            temporal_pool_size=config["temporal_pool_size"],
        )
        scaler = train_dataset.scaler
        val_dataset = SequenceWindowDataset(
            val_records,
            window_size=config["window_size"],
            stride=config["stride"],
            scaler=scaler,
            feature_mask=feature_mask,
            cache_size=config["cache_size"],
            max_windows_per_file=None,
            temporal_pooling=config["temporal_pooling"],
            temporal_pool_size=config["temporal_pool_size"],
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

        model = MambaSequenceClassifier(
            input_channels=input_channels,
            num_classes=num_classes,
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            d_state=config["d_state"],
            d_conv=config["d_conv"],
            expand=config["expand"],
            dropout=config["dropout"],
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        early_stopping = EarlyStopping(
            patience=config["patience"],
            checkpoint_path=fold_checkpoint_path,
        )

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
                    "scaler": scaler,
                    "feature_mask": feature_mask.tolist() if feature_mask is not None else None,
                    "selected_feature_indices": selected_feature_indices,
                    "original_channels": original_channels,
                    "channel_names": detected_channel_names,
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

        best_checkpoint = early_stopping.restore_best_weights(model, map_location=device)
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
            "history": history,
        }
        fold_results.append(fold_result)
        all_cv_preds.extend(val_preds)
        all_cv_labels.extend(val_labels)

        print("\nFold result:")
        print(f"  subject: {val_subject_id}")
        print(f"  best epoch: {fold_result['best_epoch'] + 1}")
        print(f"  best val loss: {best_val_loss:.4f}")
        print(f"  best val acc: {best_val_acc:.2f}%")
        print(f"  checkpoint: {fold_checkpoint_path}")

        if best_overall_fold is None or fold_result["val_loss"] < best_overall_fold["val_loss"]:
            best_overall_fold = fold_result

    summarize_cv_results(fold_results)

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

    if best_overall_fold is not None:
        best_overall_history = best_overall_fold["history"]
        plot_training_history(best_overall_history, save_path=config["history_path"])
        best_overall_checkpoint = torch.load(
            best_overall_fold["checkpoint_path"],
            weights_only=False,
        )
        torch.save(best_overall_checkpoint, config["checkpoint_path"])

    print("\n" + "=" * 60)
    print("Cross-validation complete!")
    if best_overall_fold is not None:
        print(
            f"Best fold: {best_overall_fold['fold_index']} "
            f"(subject {best_overall_fold['subject_id']})"
        )
        print(f"Best fold val loss: {best_overall_fold['val_loss']:.4f}")
        print(f"Best fold val acc: {best_overall_fold['val_acc']:.2f}%")
    print(f"Aggregate confusion matrix saved to: {config['confusion_matrix_path']}")
    print(f"Best overall fold checkpoint copied to: {config['checkpoint_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
