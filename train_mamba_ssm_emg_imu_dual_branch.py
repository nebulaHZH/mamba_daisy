"""
Train the pure-PyTorch selective-scan Mamba SSM model.

Use --modality to choose the data path:
- both: paired sEMG + IMU dual-branch training
- emg: single-modality sEMG training
- imu: single-modality IMU training

The heavy lifting is still done by the existing training scripts. This wrapper
only selects the modality and swaps their model class to the pure-PyTorch SSM
implementation in mamba_ssm_pytorch.py.
"""

import argparse
import sys
from pathlib import Path

import train_mamba_emg_imu as single_base
import train_mamba_emg_imu_dual_branch as dual_base
from mamba_ssm_pytorch import (
    DualBranchMambaSSMSequenceClassifier,
    MambaSSMSequenceClassifier,
)

_DUAL_RESOLVE_OUTPUT_PATHS = dual_base.resolve_output_paths
_SINGLE_RESOLVE_OUTPUT_PATHS = single_base.resolve_output_paths
OUTPUT_DIR = Path("mamba_ssm_results")


def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--modality",
        choices=("both", "emg", "imu"),
        default="both",
        help="Choose both for paired EMG+IMU, emg for sEMG only, or imu for IMU only.",
    )
    parsed, _ = parser.parse_known_args(argv[1:])
    return parsed


def remove_option(argv, option, takes_value=True):
    cleaned = [argv[0]]
    skip_next = False
    for index, arg in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            if takes_value and index + 1 < len(argv):
                skip_next = True
            continue
        if takes_value and arg.startswith(f"{option}="):
            continue
        cleaned.append(arg)
    return cleaned


def pop_option_value(argv, option):
    value = None
    cleaned = [argv[0]]
    skip_next = False
    for index, arg in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            if index + 1 < len(argv):
                value = argv[index + 1]
                skip_next = True
            continue
        if arg.startswith(f"{option}="):
            value = arg.split("=", 1)[1]
            continue
        cleaned.append(arg)
    return cleaned, value


def has_option(argv, option):
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv[1:])


def append_option_if_missing(argv, option, value):
    if has_option(argv, option):
        return argv
    return argv + [option, str(value)]


def prepare_dual_argv(argv):
    argv = remove_option(argv, "--modality", takes_value=True)
    argv = append_option_if_missing(argv, "--emg-temporal-pooling", "rms")
    argv = append_option_if_missing(argv, "--emg-temporal-pool-size", 5)
    return argv


def prepare_single_argv(argv, modality):
    argv = remove_option(argv, "--modality", takes_value=True)

    data_root_option = "--emg-data-root" if modality == "emg" else "--imu-data-root"
    other_root_option = "--imu-data-root" if modality == "emg" else "--emg-data-root"
    argv, selected_data_root = pop_option_value(argv, data_root_option)
    argv, _ = pop_option_value(argv, other_root_option)
    if selected_data_root is not None and not has_option(argv, "--data-root"):
        argv += ["--data-root", selected_data_root]

    sensor_sample_rate = "--emg-sample-rate-hz" if modality == "emg" else "--imu-sample-rate-hz"
    other_sample_rate = "--imu-sample-rate-hz" if modality == "emg" else "--emg-sample-rate-hz"
    argv, selected_sample_rate = pop_option_value(argv, sensor_sample_rate)
    argv, _ = pop_option_value(argv, other_sample_rate)
    if selected_sample_rate is not None and not has_option(argv, "--sample-rate-hz"):
        argv += ["--sample-rate-hz", selected_sample_rate]

    sensor_window = "--emg-window-size" if modality == "emg" else "--imu-window-size"
    other_window = "--imu-window-size" if modality == "emg" else "--emg-window-size"
    argv, selected_window = pop_option_value(argv, sensor_window)
    argv, _ = pop_option_value(argv, other_window)
    if selected_window is not None and not has_option(argv, "--window-size"):
        argv += ["--window-size", selected_window]

    sensor_stride = "--emg-stride" if modality == "emg" else "--imu-stride"
    other_stride = "--imu-stride" if modality == "emg" else "--emg-stride"
    argv, selected_stride = pop_option_value(argv, sensor_stride)
    argv, _ = pop_option_value(argv, other_stride)
    if selected_stride is not None and not has_option(argv, "--stride"):
        argv += ["--stride", selected_stride]

    sensor_mask = "--emg-feature-mask" if modality == "emg" else "--imu-feature-mask"
    other_mask = "--imu-feature-mask" if modality == "emg" else "--emg-feature-mask"
    argv, selected_mask = pop_option_value(argv, sensor_mask)
    argv, _ = pop_option_value(argv, other_mask)
    if selected_mask is not None and not has_option(argv, "--feature-mask"):
        argv += ["--feature-mask", selected_mask]

    sensor_expected = "--emg-expected-channels" if modality == "emg" else "--imu-expected-channels"
    other_expected = "--imu-expected-channels" if modality == "emg" else "--emg-expected-channels"
    argv, selected_expected = pop_option_value(argv, sensor_expected)
    argv, _ = pop_option_value(argv, other_expected)
    if selected_expected is not None and not has_option(argv, "--expected-channels"):
        argv += ["--expected-channels", selected_expected]

    for ignored_option in (
        "--start-subject",
        "--end-subject",
        "--benchmark-warmup",
        "--benchmark-repeat",
    ):
        argv, ignored_value = pop_option_value(argv, ignored_option)
        if ignored_value is not None:
            print(f"Ignoring {ignored_option} for single-modality mode; the base single script runs all LOSO folds.")

    if has_option(argv, "--skip-existing-folds"):
        argv = remove_option(argv, "--skip-existing-folds", takes_value=False)
        print("Ignoring --skip-existing-folds for single-modality mode.")

    argv = append_option_if_missing(argv, "--sensor-folder", modality)
    if modality == "emg":
        argv = append_option_if_missing(argv, "--temporal-pooling", "rms")
        argv = append_option_if_missing(argv, "--temporal-pool-size", 5)
    return argv


def resolve_dual_output_paths(explicit_prefix=None):
    prefix = explicit_prefix if explicit_prefix else "mamba_ssm_emg_imu_dual_loso"
    prefix = Path(prefix).stem
    if prefix.endswith(".pth"):
        prefix = prefix[:-4]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        str(OUTPUT_DIR / f"best_model_{prefix}.pth"),
        str(OUTPUT_DIR / f"training_history_{prefix}.png"),
        str(OUTPUT_DIR / f"confusion_matrix_{prefix}.png"),
    )


def resolve_single_output_paths(sensor_folder, explicit_prefix=None):
    prefix = explicit_prefix if explicit_prefix else f"mamba_ssm_{sensor_folder}_loso"
    prefix = Path(prefix).stem
    if prefix.endswith(".pth"):
        prefix = prefix[:-4]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        str(OUTPUT_DIR / f"best_model_{prefix}.pth"),
        str(OUTPUT_DIR / f"training_history_{prefix}.png"),
        str(OUTPUT_DIR / f"confusion_matrix_{prefix}.png"),
    )


def run_dual(argv):
    dual_base.DualBranchMambaSequenceClassifier = DualBranchMambaSSMSequenceClassifier
    dual_base.resolve_output_paths = resolve_dual_output_paths
    sys.argv = prepare_dual_argv(argv)

    if "--output-prefix" not in sys.argv:
        print("Output prefix: mamba_ssm_emg_imu_dual_loso")
    print("Modality: both (paired sEMG + IMU)")
    print(
        "Using DualBranchMambaSSMSequenceClassifier. "
        "This pure PyTorch selective scan is structurally closer to Mamba, "
        "but slower than fused CUDA mamba-ssm kernels."
    )
    dual_base.main()


def run_single(argv, modality):
    single_base.MambaSequenceClassifier = MambaSSMSequenceClassifier
    single_base.resolve_output_paths = resolve_single_output_paths
    sys.argv = prepare_single_argv(argv, modality)

    if "--output-prefix" not in sys.argv:
        print(f"Output prefix: mamba_ssm_{modality}_loso")
    print(f"Modality: {modality}")
    if modality == "emg" and has_option(sys.argv, "--temporal-pooling"):
        print(
            "Using MambaSSMSequenceClassifier with temporal pooling. "
            "The physical sEMG window stays unchanged while the model sequence is shorter."
        )
    else:
        print("Using MambaSSMSequenceClassifier.")
    single_base.main()


def main():
    wrapper_args = parse_wrapper_args(sys.argv)
    if wrapper_args.modality == "both":
        run_dual(sys.argv)
    else:
        run_single(sys.argv, wrapper_args.modality)


if __name__ == "__main__":
    main()
