import os
import sys
import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE5_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PHASE5_DIR.parent

if str(PHASE5_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PHASE 5 MODULES
# ============================================================

from phase5_utils import (
    load_config,
    cfg_path,
    ensure_dir,
    save_csv,
    save_json,
    standardize_metadata_df,
    validate_metadata_df,
    metadata_summary,
    build_sequence_manifest,
    validate_sequence_manifest,
    sequence_manifest_summary,
    save_sequence_manifest,
    print_dict,
    print_dataframe_summary,
)

from external_label_mapping import (
    build_all_external_metadata,
    save_metadata_outputs,
    metadata_quality_report,
)


# ============================================================
# HELPERS
# ============================================================

def save_dataset_split_metadata(config, all_df):
    """
    Save per-dataset metadata files.

    Outputs:
        le2i_metadata.csv
        mulcamfall_metadata.csv
        all_external_metadata.csv
    """
    outputs = save_metadata_outputs(config, all_df)
    return outputs


def save_dataset_split_sequences(config, sequence_df):
    """
    Save sequence manifest for:
        - all external datasets
        - each dataset separately

    Important:
        This is only the raw sequence manifest.
        The final fair common manifest will be created later after checking
        2D, 3D, and quality feature availability.
    """
    output_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    ensure_dir(output_dir)

    outputs = {}

    all_seq_path = output_dir / "all_external_sequences.csv"
    save_csv(sequence_df, all_seq_path)
    outputs["all_external_sequences"] = str(all_seq_path)

    if not sequence_df.empty and "dataset" in sequence_df.columns:
        for dataset_name, df_dataset in sequence_df.groupby("dataset"):
            safe_name = str(dataset_name).lower().replace(" ", "_")
            path = output_dir / f"{safe_name}_sequences.csv"
            save_csv(df_dataset.reset_index(drop=True), path)
            outputs[f"{safe_name}_sequences"] = str(path)

    return outputs


def build_preparation_report(config, metadata_df, sequence_df, metadata_outputs, sequence_outputs):
    """
    Build a JSON report for Phase 5 preparation.
    """
    report = {
        "phase": "Phase 5 - External Dataset Generalization",
        "step": "01_prepare_external_dataset",
        "metadata": metadata_summary(metadata_df),
        "metadata_quality_report": metadata_quality_report(metadata_df),
        "sequences": sequence_manifest_summary(sequence_df),
        "outputs": {
            "metadata_outputs": metadata_outputs,
            "sequence_outputs": sequence_outputs,
        },
        "fairness_rule": {
            "important": True,
            "message": (
                "This step creates the common external sequence manifest. "
                "All models must later evaluate on the same fair common manifest. "
                "After feature extraction, script 05/06 must filter this manifest by "
                "shared availability of 2D, 3D, and quality features before comparing models."
            ),
        },
    }

    return report


def print_label_warning(metadata_df, sequence_df):
    """
    Warn if one dataset contains only one class.
    This does not stop the pipeline, but it matters for interpretation.
    """
    print("\nDataset label check")
    print("=" * 100)

    if metadata_df.empty:
        print("Metadata is empty.")
        print("=" * 100)
        return

    for dataset_name, df_dataset in metadata_df.groupby("dataset"):
        label_counts = df_dataset["label_name"].value_counts(dropna=False).to_dict()
        print(f"{dataset_name}: {label_counts}")

        valid_labels = set(df_dataset["label"].dropna().astype(int).tolist())

        if valid_labels == {1}:
            print(
                f"WARNING: {dataset_name} only has Fall samples. "
                "Use it mainly for Fall Recall, not standalone Macro F1."
            )

        elif valid_labels == {0}:
            print(
                f"WARNING: {dataset_name} only has Not_Fall samples. "
                "Use it mainly for false positive analysis, not standalone Macro F1."
            )

    print("=" * 100)

    print("\nSequence label check")
    print("=" * 100)

    if sequence_df.empty:
        print("Sequence manifest is empty.")
        print("=" * 100)
        return

    for dataset_name, df_dataset in sequence_df.groupby("dataset"):
        label_counts = df_dataset["label_name"].value_counts(dropna=False).to_dict()
        print(f"{dataset_name}: {label_counts}")

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nPhase 5 - Step 01: Prepare External Dataset")
    print("=" * 100)

    config = load_config()

    preparation_dir = cfg_path(config, config["outputs"]["preparation_dir"])
    ensure_dir(preparation_dir)

    external_sequences_dir = cfg_path(config, config["outputs"]["external_sequences_dir"])
    ensure_dir(external_sequences_dir)

    # ------------------------------------------------------------
    # 1. Build metadata from external datasets
    # ------------------------------------------------------------
    print("\n[1/5] Building external metadata...")
    all_df = build_all_external_metadata(config)

    if all_df.empty:
        raise RuntimeError(
            "No external metadata was created. "
            "Check dataset paths and annotation files."
        )

    all_df = standardize_metadata_df(all_df)

    # ------------------------------------------------------------
    # 2. Validate metadata
    # ------------------------------------------------------------
    print("\n[2/5] Validating metadata...")
    metadata_validation = validate_metadata_df(all_df, strict=True)

    print_dict("Metadata validation", metadata_validation)
    print_dataframe_summary("Metadata preview", all_df, max_rows=12)

    # ------------------------------------------------------------
    # 3. Save metadata files
    # ------------------------------------------------------------
    print("\n[3/5] Saving metadata CSV files...")
    metadata_outputs = save_dataset_split_metadata(config, all_df)

    print_dict("Metadata outputs", metadata_outputs)

    # ------------------------------------------------------------
    # 4. Build common external sequence manifest
    # ------------------------------------------------------------
    print("\n[4/5] Building external sequence manifest...")

    sequence_df = build_sequence_manifest(
        metadata_df=all_df,
        config=config,
        dataset_filter=None,
    )

    if sequence_df.empty:
        raise RuntimeError(
            "External sequence manifest is empty. "
            "Check segment_start_frame, segment_end_frame, and video frame counts."
        )

    sequence_validation = validate_sequence_manifest(sequence_df, strict=True)

    print_dict("Sequence validation", sequence_validation)
    print_dataframe_summary("Sequence manifest preview", sequence_df, max_rows=12)

    sequence_outputs = save_dataset_split_sequences(config, sequence_df)

    print_dict("Sequence outputs", sequence_outputs)

    # ------------------------------------------------------------
    # 5. Save preparation report
    # ------------------------------------------------------------
    print("\n[5/5] Saving preparation report...")

    report = build_preparation_report(
        config=config,
        metadata_df=all_df,
        sequence_df=sequence_df,
        metadata_outputs=metadata_outputs,
        sequence_outputs=sequence_outputs,
    )

    report_path = preparation_dir / "01_prepare_external_dataset_report.json"
    save_json(report, report_path)

    print_dict("Preparation report", report)

    print_label_warning(all_df, sequence_df)

    print("\nDONE: Phase 5 external dataset preparation completed.")
    print("=" * 100)
    print(f"Metadata saved to : {metadata_outputs.get('all_external_metadata')}")
    print(f"Sequences saved to: {sequence_outputs.get('all_external_sequences')}")
    print(f"Report saved to   : {report_path}")
    print("=" * 100)

    print(
        "\nIMPORTANT FAIRNESS NOTE:\n"
        "This file created the common external sequence manifest.\n"
        "Later, after extracting 2D, 3D, and quality features, all models must be evaluated on\n"
        "the SAME fair common manifest, not on separate per-model sample sets.\n"
    )


if __name__ == "__main__":
    main()