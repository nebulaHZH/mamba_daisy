from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report

from train_mamba_emg_imu import (
    LABEL_DISPLAY_NAMES,
    LABEL_NAMES,
    build_leave_one_subject_out_folds,
    summarize_split,
)
from train_svm import (
    compute_metrics,
    configure_style,
    estimate_fit_memory_gb,
    export_feature_excel,
    fit_dual_scalers,
    fit_single_scaler,
    get_feature_names,
    materialize_dual,
    materialize_single,
    plot_confusion,
    plot_feature_group_importance,
    plot_fold_accuracy,
    plot_top_feature_importance,
    resolve_dual_setup,
    resolve_feature_excel_path,
    resolve_single_setup,
)


warnings.filterwarnings("ignore")
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOSO XGBoost baseline on handcrafted window features.")
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

    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-gamma", type=float, default=0.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-max-bin", type=int, default=256)
    parser.add_argument("--xgb-tree-method", choices=("auto", "hist"), default="hist")
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument(
        "--xgb-importance-type",
        choices=("weight", "gain", "cover", "total_gain", "total_cover"),
        default="gain",
    )

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
    window_tag = format_ms_tag(args.window_ms)
    stride_tag = format_ms_tag(args.stride_ms)
    if args.input_mode == "dual":
        return Path.cwd() / f"XGBoost{window_tag}ms滑窗{stride_tag}ms步长sEMG和imu结果图"
    sensor_tag = "imu" if str(args.sensor_folder).lower() == "imu" else "sEMG"
    return Path.cwd() / f"XGBoost{window_tag}ms滑窗{stride_tag}ms步长{sensor_tag}结果图"


def build_xgboost_model(args: argparse.Namespace) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABEL_NAMES),
        n_estimators=int(args.xgb_n_estimators),
        max_depth=int(args.xgb_max_depth),
        learning_rate=float(args.xgb_learning_rate),
        subsample=float(args.xgb_subsample),
        colsample_bytree=float(args.xgb_colsample_bytree),
        min_child_weight=float(args.xgb_min_child_weight),
        gamma=float(args.xgb_gamma),
        reg_alpha=float(args.xgb_reg_alpha),
        reg_lambda=float(args.xgb_reg_lambda),
        max_bin=int(args.xgb_max_bin),
        tree_method=args.xgb_tree_method,
        n_jobs=int(args.xgb_n_jobs),
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        importance_type=args.xgb_importance_type,
    )


def extract_xgboost_feature_importance(model: xgb.XGBClassifier) -> np.ndarray | None:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return None
    return np.asarray(importance, dtype=np.float64)


def plot_feature_importance_if_available(
    fold_importances: list[np.ndarray],
    feature_names: list[str],
    output_dir: Path,
) -> None:
    if not fold_importances:
        print("\nFeature importance was skipped because XGBoost did not expose importances.")
        return

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
    importance_df.to_csv(output_dir / "xgboost_feature_importance.csv", index=False, encoding="utf-8-sig")

    group_df = (
        importance_df.groupby(["modality", "feature_type"], as_index=False)["importance"]
        .sum()
        .assign(group=lambda df: df["modality"] + ":" + df["feature_type"])
    )
    group_df.to_csv(output_dir / "xgboost_feature_group_importance.csv", index=False, encoding="utf-8-sig")
    plot_top_feature_importance(importance_df, output_dir / "xgboost_top_feature_importance.png")
    plot_feature_group_importance(group_df, output_dir / "xgboost_feature_group_importance.png")

    print("\nTop 10 features by mean XGBoost importance:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature_name']}: {row['importance']:.6f}")


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    summary: dict[str, object],
    fold_df: pd.DataFrame,
) -> None:
    lines = ["# XGBoost LOSO Experiment Summary", "", "## Protocol"]
    if args.input_mode == "dual":
        lines.append("- Input: paired sEMG and IMU windows.")
        lines.append("- Features: handcrafted channel-wise features from sEMG and IMU, scaled per fold, then concatenated.")
    else:
        lines.append("- Input: single-modality windows transformed into handcrafted channel-wise features.")
    lines.append("- Evaluation: leave-one-subject-out cross-validation.")
    lines.append("- Leakage control: StandardScaler is fit inside each LOSO fold using training data only.")
    lines.append("- Leakage control: a brand-new XGBoost model is instantiated inside every fold.")
    lines.append("- Hyperparameters are fixed before LOSO and never tuned on the held-out subject.")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append(f"- Folds completed: {summary['folds_completed']}")
    lines.append(f"- Mean accuracy: {summary['mean_accuracy_pct']:.2f}% +/- {summary['std_accuracy_pct']:.2f}%")
    lines.append(f"- Mean weighted F1: {summary['mean_f1_weighted_pct']:.2f}% +/- {summary['std_f1_weighted_pct']:.2f}%")
    lines.append(f"- Mean macro F1: {summary['mean_f1_macro_pct']:.2f}% +/- {summary['std_f1_macro_pct']:.2f}%")
    lines.append(f"- Aggregate accuracy: {summary['aggregate_accuracy_pct']:.2f}%")
    lines.append(f"- Mean train time per fold: {summary['mean_train_time_sec']:.2f}s")
    lines.append(f"- Mean inference time per fold: {summary['mean_inference_time_sec']:.4f}s")
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
    (output_dir / "xgboost_loso_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    setup = resolve_dual_setup(args) if args.input_mode == "dual" else resolve_single_setup(args)
    artifact_prefix = "xgboost_emg_imu_dual_loso" if args.input_mode == "dual" else f"xgboost_{setup['sensor_folder']}_loso"
    feature_names = get_feature_names(setup)

    print("=" * 80)
    print("XGBoost baseline on handcrafted window features with LOSO-CV")
    print("=" * 80)
    print(f"Input mode: {args.input_mode}")
    print(f"Output dir: {output_dir}")
    print("Important: each LOSO fold will fit scaler(s) on training only and create a brand-new XGBoost model.")
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
    print(
        f"XGBoost tree_method: {args.xgb_tree_method} | "
        f"estimators: {args.xgb_n_estimators} | max_depth: {args.xgb_max_depth}"
    )

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
            X_train, y_train = materialize_dual(
                train_records,
                setup,
                args.max_train_windows_per_file,
                emg_scaler,
                imu_scaler,
                args,
            )
            print("Materializing paired test features...")
            X_test, y_test = materialize_dual(
                test_records,
                setup,
                args.max_test_windows_per_file,
                emg_scaler,
                imu_scaler,
                args,
            )
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
            print("Feature scaling: StandardScaler on handcrafted features before XGBoost.")

        print(f"Training matrix: {X_train.shape}")
        print(f"Test matrix: {X_test.shape}")
        estimated_fit_memory_gb = estimate_fit_memory_gb(X_train)
        print(f"Estimated matrix memory if materialized as float64: {estimated_fit_memory_gb:.2f} GiB")
        if estimated_fit_memory_gb > float(args.max_fit_memory_gb):
            raise MemoryError(
                "The training matrix is too large for this configuration. "
                f"Estimated float64 matrix memory is {estimated_fit_memory_gb:.2f} GiB, "
                f"which exceeds --max-fit-memory-gb={args.max_fit_memory_gb}. "
                "Try lowering --max-train-windows-per-file, adding feature masks, or using a smaller subset first."
            )

        model = build_xgboost_model(args)
        train_start = perf_counter()
        model.fit(X_train, y_train)
        train_time_sec = perf_counter() - train_start

        importance_vector = extract_xgboost_feature_importance(model)
        if importance_vector is not None and importance_vector.shape[0] == len(feature_names):
            fold_importances.append(importance_vector)

        inference_start = perf_counter()
        y_pred = model.predict(X_test)
        inference_time_sec = perf_counter() - inference_start
        y_pred = np.asarray(y_pred, dtype=np.int64)

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
        all_labels.extend(np.asarray(y_test, dtype=np.int64).tolist())

        plot_confusion(
            np.asarray(y_test, dtype=np.int64),
            y_pred,
            f"XGBoost Confusion Matrix - Fold {fold_index:02d} ({subject_id})",
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
    fold_df.to_csv(output_dir / "xgboost_loso_fold_results.csv", index=False, encoding="utf-8-sig")

    y_true_all = np.asarray(all_labels, dtype=np.int64)
    y_pred_all = np.asarray(all_preds, dtype=np.int64)
    aggregate_metrics = compute_metrics(y_true_all, y_pred_all)
    summary = {
        "input_mode": args.input_mode,
        "folds_completed": int(len(fold_df)),
        "mean_accuracy_pct": float(fold_df["accuracy_pct"].mean()),
        "std_accuracy_pct": float(fold_df["accuracy_pct"].std(ddof=0)),
        "mean_f1_weighted_pct": float(fold_df["f1_weighted_pct"].mean()),
        "std_f1_weighted_pct": float(fold_df["f1_weighted_pct"].std(ddof=0)),
        "mean_f1_macro_pct": float(fold_df["f1_macro_pct"].mean()),
        "std_f1_macro_pct": float(fold_df["f1_macro_pct"].std(ddof=0)),
        "aggregate_accuracy_pct": float(aggregate_metrics["accuracy"] * 100.0),
        "aggregate_f1_weighted_pct": float(aggregate_metrics["f1_weighted"] * 100.0),
        "aggregate_f1_macro_pct": float(aggregate_metrics["f1_macro"] * 100.0),
        "mean_train_time_sec": float(fold_df["train_time_sec"].mean()),
        "mean_inference_time_sec": float(fold_df["inference_time_sec"].mean()),
    }
    (output_dir / "xgboost_loso_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "xgboost_loso_classification_report.txt").write_text(
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

    plot_feature_importance_if_available(fold_importances, feature_names, output_dir)
    plot_confusion(
        y_true_all,
        y_pred_all,
        "XGBoost Aggregate Confusion Matrix - LOSO",
        output_dir / f"confusion_matrix_{artifact_prefix}.png",
    )
    plot_fold_accuracy(
        fold_df,
        f"XGBoost LOSO Accuracy by Held-out Subject ({args.input_mode})",
        output_dir / f"{artifact_prefix}_fold_accuracy.png",
    )
    write_report(output_dir / "xgboost_loso_report.md", args, summary, fold_df)

    print("\n" + "=" * 80)
    print("XGBoost LOSO complete")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Mean accuracy: {summary['mean_accuracy_pct']:.2f}% +/- {summary['std_accuracy_pct']:.2f}%")
    print(f"Mean weighted F1: {summary['mean_f1_weighted_pct']:.2f}% +/- {summary['std_f1_weighted_pct']:.2f}%")
    print(f"Aggregate accuracy over all held-out windows: {summary['aggregate_accuracy_pct']:.2f}%")
    print("Leakage check: scaler(s) fit only on training folds, and XGBoost re-instantiated inside every fold.")


if __name__ == "__main__":
    main()
