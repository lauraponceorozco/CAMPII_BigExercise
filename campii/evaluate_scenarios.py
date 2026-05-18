import os
import nibabel as nib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DATA_DIR = os.path.join(SCRIPT_DIR, "data_test")
BASELINE_RESULTS_DIR = os.path.join(SCRIPT_DIR, "baseline_results")
CORRUPTED_RESULTS_DIR = os.path.join(SCRIPT_DIR, "corrupted_results")
AUGMENTED_RESULTS_DIR = os.path.join(SCRIPT_DIR, "augmented_results")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "evaluation_comparison.csv")

PATIENT_FOLDERS = sorted([f for f in os.listdir(BASELINE_DATA_DIR) if f.startswith("BraTS20_Training")])[:20]


def calculate_metrics(pred_path, gt_path):
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)

    pred = pred_img.get_fdata() > 0
    gt = gt_img.get_fdata() > 0

    voxel_dims = pred_img.header.get_zooms()
    voxel_vol = np.prod(voxel_dims)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    fp_vol = fp * voxel_vol
    fn_vol = fn * voxel_vol

    return dice, fp_vol, fn_vol


def available_predictions(pattern_dir, suffix):
    return os.path.exists(pattern_dir) and any(
        f.endswith(suffix) for f in os.listdir(pattern_dir)
    )


def main():
    rows = []

    for patient_id in PATIENT_FOLDERS:
        gt_path = os.path.join(BASELINE_DATA_DIR, patient_id, f"{patient_id}_seg.nii.gz")
        baseline_pred = os.path.join(BASELINE_RESULTS_DIR, f"{patient_id}_baseline_seg.nii.gz")
        corrupted_pred = os.path.join(CORRUPTED_RESULTS_DIR, f"{patient_id}_corrupted_seg.nii.gz")
        augmented_pred = os.path.join(AUGMENTED_RESULTS_DIR, f"{patient_id}_augmented_seg.nii.gz")

        row = {
            "Patient_ID": patient_id,
            "GroundTruth": os.path.exists(gt_path),
            "Baseline_Prediction": os.path.exists(baseline_pred),
            "Corrupted_Prediction": os.path.exists(corrupted_pred),
            "Augmented_Prediction": os.path.exists(augmented_pred),
            "Baseline_Dice": None,
            "Baseline_FPvol_mm3": None,
            "Baseline_FNvol_mm3": None,
            "Corrupted_Dice": None,
            "Corrupted_FPvol_mm3": None,
            "Corrupted_FNvol_mm3": None,
            "Augmented_Dice": None,
            "Augmented_FPvol_mm3": None,
            "Augmented_FNvol_mm3": None,
        }

        if row["GroundTruth"]:
            if row["Baseline_Prediction"]:
                row["Baseline_Dice"], row["Baseline_FPvol_mm3"], row["Baseline_FNvol_mm3"] = calculate_metrics(baseline_pred, gt_path)
            if row["Corrupted_Prediction"]:
                row["Corrupted_Dice"], row["Corrupted_FPvol_mm3"], row["Corrupted_FNvol_mm3"] = calculate_metrics(corrupted_pred, gt_path)
            if row["Augmented_Prediction"]:
                row["Augmented_Dice"], row["Augmented_FPvol_mm3"], row["Augmented_FNvol_mm3"] = calculate_metrics(augmented_pred, gt_path)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Evaluation completed.")
    print(f"Saved comparison results to: {OUTPUT_CSV}")

    if not available_predictions(CORRUPTED_RESULTS_DIR, "_corrupted_seg.nii.gz"):
        print("Warning: No corrupted predictions found. Run the corruption pipeline or generate corrupted model outputs.")
    if not available_predictions(AUGMENTED_RESULTS_DIR, "_augmented_seg.nii.gz"):
        print("Warning: No augmented predictions found. Evaluate the model on augmented inputs or add augmented predictions to augmented_results/.")

    print("\nNext actions:")
    print("  1. If you have model predictions for corrupted inputs, place them in campii/corrupted_results/ with names like <patient>_corrupted_seg.nii.gz")
    print("  2. If you have augmented model predictions, place them in campii/augmented_results/ with names like <patient>_augmented_seg.nii.gz")
    print("  3. Re-run this script to generate a full baseline/corrupted/augmented comparison.")
    print("  4. For visual inspection, open the files listed in campii/VISUALIZATION_GUIDE.py with NiiVue.")


if __name__ == "__main__":
    main()
