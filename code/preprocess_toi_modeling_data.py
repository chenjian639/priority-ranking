"""
Preprocess NASA TESS TOI data for the exoplanet candidate risk project.

Outputs:
- raw/NASA_TESS_TOI_2026-05-28_FULL.csv
- processed/selected_columns_all_toi.csv
- processed/train_modeling_table_cp_kp_vs_fp_fa.csv
- processed/candidate_pc_apc_for_risk_prediction.csv
- processed/excluded_other_labels.csv
- processed/preprocessing_summary.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


ID_COLUMNS = ["tid", "toi", "toidisplay"]
LABEL_COLUMN = "tfopwg_disp"
FEATURE_COLUMNS = ["pl_orbper", "pl_rade", "st_tmag", "st_teff", "st_rad"]
METADATA_COLUMNS = ["rowupdate", "release_date"]

POSITIVE_LABELS = {"CP", "KP"}
NEGATIVE_LABELS = {"FP", "FA"}
CANDIDATE_LABELS = {"PC", "APC"}


def parse_args() -> argparse.Namespace:
    default_input = Path(r"C:\Users\35512\Desktop\NASA_TESS_TOI_2026-05-28_FULL.csv")
    default_output = Path(r"C:\Users\35512\Desktop\TESS_TOI_preprocessing_package")
    parser = argparse.ArgumentParser(
        description="Create modeling tables from NASA TESS TOI CSV data."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Path to raw TOI CSV.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Output package directory.")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_label(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA})


def add_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df[ID_COLUMNS + [LABEL_COLUMN] + FEATURE_COLUMNS + METADATA_COLUMNS].copy()
    out[LABEL_COLUMN] = normalize_label(out[LABEL_COLUMN])

    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["missing_feature_count"] = out[FEATURE_COLUMNS].isna().sum(axis=1)
    out["missing_feature_names"] = out[FEATURE_COLUMNS].isna().apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1
    )

    def group_label(label: object) -> str:
        if pd.isna(label):
            return "unlabeled_or_other"
        label = str(label)
        if label in POSITIVE_LABELS:
            return "confirmed_or_known_planet"
        if label in NEGATIVE_LABELS:
            return "false_positive_or_alarm"
        if label in CANDIDATE_LABELS:
            return "candidate_for_prediction"
        return "unlabeled_or_other"

    out["disposition_group"] = out[LABEL_COLUMN].map(group_label)
    return out


def build_outputs(raw_path: Path, output_dir: Path) -> dict:
    raw_path = raw_path.resolve()
    output_dir = output_dir.resolve()
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    required = ID_COLUMNS + [LABEL_COLUMN] + FEATURE_COLUMNS + METADATA_COLUMNS
    require_columns(df, required)

    raw_copy = raw_dir / raw_path.name
    if raw_copy != raw_path:
        shutil.copy2(raw_path, raw_copy)

    selected = add_modeling_columns(df)

    train_mask = selected[LABEL_COLUMN].isin(POSITIVE_LABELS | NEGATIVE_LABELS)
    candidate_mask = selected[LABEL_COLUMN].isin(CANDIDATE_LABELS)
    known_mask = train_mask | candidate_mask

    train = selected.loc[train_mask].copy()
    train["target_confirmed_planet"] = train[LABEL_COLUMN].isin(POSITIVE_LABELS).astype(int)
    train["target_label_name"] = train["target_confirmed_planet"].map(
        {1: "positive_CP_KP", 0: "negative_FP_FA"}
    )

    candidate = selected.loc[candidate_mask].copy()
    candidate["target_confirmed_planet"] = pd.NA
    candidate["target_label_name"] = "to_be_predicted_after_model_training"

    excluded = selected.loc[~known_mask].copy()

    ordered_columns = (
        ID_COLUMNS
        + [LABEL_COLUMN, "disposition_group", "target_confirmed_planet", "target_label_name"]
        + FEATURE_COLUMNS
        + ["missing_feature_count", "missing_feature_names"]
        + METADATA_COLUMNS
    )
    selected_columns = (
        ID_COLUMNS
        + [LABEL_COLUMN, "disposition_group"]
        + FEATURE_COLUMNS
        + ["missing_feature_count", "missing_feature_names"]
        + METADATA_COLUMNS
    )

    selected_path = processed_dir / "selected_columns_all_toi.csv"
    train_path = processed_dir / "train_modeling_table_cp_kp_vs_fp_fa.csv"
    candidate_path = processed_dir / "candidate_pc_apc_for_risk_prediction.csv"
    excluded_path = processed_dir / "excluded_other_labels.csv"
    summary_path = processed_dir / "preprocessing_summary.json"

    selected.to_csv(selected_path, index=False, encoding="utf-8-sig", columns=selected_columns)
    train.to_csv(train_path, index=False, encoding="utf-8-sig", columns=ordered_columns)
    candidate.to_csv(candidate_path, index=False, encoding="utf-8-sig", columns=ordered_columns)
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig", columns=selected_columns)

    summary = {
        "input_file": str(raw_path),
        "raw_copy": str(raw_copy),
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "selected_columns": selected_columns,
        "feature_columns_used_for_missing_count": FEATURE_COLUMNS,
        "missing_feature_count_definition": "Number of missing values across FEATURE_COLUMNS only; label and metadata fields are not counted.",
        "label_counts_original": {
            str(k) if pd.notna(k) else "NA": int(v)
            for k, v in selected[LABEL_COLUMN].value_counts(dropna=False).items()
        },
        "train_table_rows": int(len(train)),
        "train_positive_cp_kp_rows": int(train["target_confirmed_planet"].sum()),
        "train_negative_fp_fa_rows": int((train["target_confirmed_planet"] == 0).sum()),
        "candidate_pc_apc_rows": int(len(candidate)),
        "excluded_other_rows": int(len(excluded)),
        "feature_missing_counts_in_selected_all": {
            col: int(selected[col].isna().sum()) for col in FEATURE_COLUMNS
        },
        "outputs": {
            "selected_all_toi": str(selected_path),
            "train_modeling_table": str(train_path),
            "candidate_pc_apc_for_prediction": str(candidate_path),
            "excluded_other_labels": str(excluded_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_outputs(args.input, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
