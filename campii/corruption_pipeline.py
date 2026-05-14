import os
import sys
import types
import numpy as np
import nibabel as nib
import pandas as pd

# FIX: Mock 'pwd' for Windows compatibility
if sys.platform == "win32":
    mock_pwd = types.ModuleType("pwd")
    def getpwuid(uid):
        return types.SimpleNamespace(pw_name='laura') 
    mock_pwd.getpwuid = getpwuid
    sys.modules["pwd"] = mock_pwd

from brats import AdultGliomaPreTreatmentSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms

# --- CONFIGURATION ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "data_test")
CORRUPTED_DATA_DIR = os.path.join(SCRIPT_DIR, "data_corrupted")
CORRUPTED_RESULTS_DIR = os.path.join(SCRIPT_DIR, "corrupted_results")
BASELINE_RESULTS_DIR = os.path.join(SCRIPT_DIR, "baseline_results")
OUTPUT_COMPARISON_CSV = os.path.join(SCRIPT_DIR, "corruption_comparison_metrics.csv")

os.makedirs(CORRUPTED_DATA_DIR, exist_ok=True)
os.makedirs(CORRUPTED_RESULTS_DIR, exist_ok=True)

# --- STEP 1: CREATE CORRUPTED DATA (T1 <-> T2 SWAP) ---
print("=" * 60)
print("STEP 1: Creating corrupted data (T1 <-> T2 swap)")
print("=" * 60)

patient_folders = sorted([f for f in os.listdir(BASE_DATA_DIR) if f.startswith("BraTS20_Training")])[:20]

for patient_id in patient_folders:
    print(f"\nProcessing: {patient_id}")
    
    patient_path = os.path.join(BASE_DATA_DIR, patient_id)
    corrupted_patient_path = os.path.join(CORRUPTED_DATA_DIR, patient_id)
    os.makedirs(corrupted_patient_path, exist_ok=True)
    
    # Define file paths
    t1n_path = os.path.join(patient_path, f"{patient_id}_t1.nii.gz")
    t1c_path = os.path.join(patient_path, f"{patient_id}_t1ce.nii.gz")
    t2w_path = os.path.join(patient_path, f"{patient_id}_t2.nii.gz")
    t2f_path = os.path.join(patient_path, f"{patient_id}_flair.nii.gz")
    
    try:
        # Load all modalities
        img_t1n = nib.load(t1n_path)
        img_t1c = nib.load(t1c_path)
        img_t2w = nib.load(t2w_path)
        img_t2f = nib.load(t2f_path)
        
        # Get data
        data_t1n = img_t1n.get_fdata()
        data_t1c = img_t1c.get_fdata()
        data_t2w = img_t2w.get_fdata()
        data_t2f = img_t2f.get_fdata()
        
        # CORRUPTION: Swap T1 (index 0) with T2 (index 2)
        # This simulates the model receiving:
        # Index 0: T2 data (instead of T1)
        # Index 1: T1ce data (unchanged)
        # Index 2: T1 data (instead of T2)
        # Index 3: FLAIR data (unchanged)
        
        data_t1n_corrupted = data_t2w  # T2 data saved as T1
        data_t2w_corrupted = data_t1n  # T1 data saved as T2
        
        # Save corrupted files (keep same naming convention)
        corrupted_t1n_path = os.path.join(corrupted_patient_path, f"{patient_id}_t1.nii.gz")
        corrupted_t1c_path = os.path.join(corrupted_patient_path, f"{patient_id}_t1ce.nii.gz")
        corrupted_t2w_path = os.path.join(corrupted_patient_path, f"{patient_id}_t2.nii.gz")
        corrupted_t2f_path = os.path.join(corrupted_patient_path, f"{patient_id}_flair.nii.gz")
        
        nib.save(nib.Nifti1Image(data_t1n_corrupted, img_t1n.affine, img_t1n.header), corrupted_t1n_path)
        nib.save(nib.Nifti1Image(data_t1c, img_t1c.affine, img_t1c.header), corrupted_t1c_path)
        nib.save(nib.Nifti1Image(data_t2w_corrupted, img_t2w.affine, img_t2w.header), corrupted_t2w_path)
        nib.save(nib.Nifti1Image(data_t2f, img_t2f.affine, img_t2f.header), corrupted_t2f_path)
        
        print(f"  ✓ Corrupted files saved to {corrupted_patient_path}")
        
    except Exception as e:
        print(f"  ✗ Error processing {patient_id}: {e}")

print("\n✓ Corrupted data creation complete!")

# --- STEP 2: RUN SEGMENTATION ON CORRUPTED DATA ---
print("\n" + "=" * 60)
print("STEP 2: Running segmentation on corrupted data")
print("=" * 60)

segmenter = AdultGliomaPreTreatmentSegmenter(
    algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_1
)

for patient_id in patient_folders:
    print(f"\nSegmenting: {patient_id}")
    
    corrupted_patient_path = os.path.join(CORRUPTED_DATA_DIR, patient_id)
    
    t1n_path = os.path.join(corrupted_patient_path, f"{patient_id}_t1.nii.gz")
    t1c_path = os.path.join(corrupted_patient_path, f"{patient_id}_t1ce.nii.gz")
    t2w_path = os.path.join(corrupted_patient_path, f"{patient_id}_t2.nii.gz")
    t2f_path = os.path.join(corrupted_patient_path, f"{patient_id}_flair.nii.gz")
    
    output_file = os.path.join(CORRUPTED_RESULTS_DIR, f"{patient_id}_corrupted_seg.nii.gz")
    
    try:
        segmenter.infer_single(
            t1n=t1n_path,
            t1c=t1c_path,
            t2w=t2w_path,
            t2f=t2f_path,
            output_file=output_file
        )
        print(f"  ✓ Saved to {output_file}")
    except Exception as e:
        print(f"  ✗ Error segmenting {patient_id}: {e}")

print("\n✓ Segmentation on corrupted data complete!")

# --- STEP 3: COMPUTE METRICS AND COMPARE ---
print("\n" + "=" * 60)
print("STEP 3: Computing metrics and comparing baseline vs corrupted")
print("=" * 60)

def calculate_metrics(pred_path, gt_path):
    """Calculate Dice, FPvol, FNvol"""
    try:
        pred_img = nib.load(pred_path)
        gt_img = nib.load(gt_path)
        
        pred = pred_img.get_fdata() > 0
        gt = gt_img.get_fdata() > 0
        
        voxel_dims = pred_img.header.get_zooms()
        voxel_vol = np.prod(voxel_dims)
        
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, np.logical_not(gt)).sum()
        fn = np.logical_and(np.logical_not(pred), gt).sum()
        
        dice = (2. * tp) / (2. * tp + fp + fn + 1e-8)
        fp_vol = fp * voxel_vol
        fn_vol = fn * voxel_vol
        
        return dice, fp_vol, fn_vol
    except Exception as e:
        print(f"  Error calculating metrics: {e}")
        return None, None, None

results_list = []

for patient_id in patient_folders:
    # Paths
    baseline_pred = os.path.join(BASELINE_RESULTS_DIR, f"{patient_id}_baseline_seg.nii.gz")
    corrupted_pred = os.path.join(CORRUPTED_RESULTS_DIR, f"{patient_id}_corrupted_seg.nii.gz")
    gt_path = os.path.join(BASE_DATA_DIR, patient_id, f"{patient_id}_seg.nii.gz")
    
    if os.path.exists(baseline_pred) and os.path.exists(corrupted_pred) and os.path.exists(gt_path):
        # Baseline metrics
        baseline_dice, baseline_fp, baseline_fn = calculate_metrics(baseline_pred, gt_path)
        
        # Corrupted metrics
        corrupted_dice, corrupted_fp, corrupted_fn = calculate_metrics(corrupted_pred, gt_path)
        
        # Calculate deltas
        dice_delta = baseline_dice - corrupted_dice if baseline_dice and corrupted_dice else None
        fp_delta = corrupted_fp - baseline_fp if baseline_fp and corrupted_fp else None
        fn_delta = corrupted_fn - baseline_fn if baseline_fn and corrupted_fn else None
        
        results_list.append({
            'Patient_ID': patient_id,
            'Baseline_Dice': baseline_dice,
            'Corrupted_Dice': corrupted_dice,
            'Dice_Delta': dice_delta,
            'Baseline_FPvol_mm3': baseline_fp,
            'Corrupted_FPvol_mm3': corrupted_fp,
            'FPvol_Delta_mm3': fp_delta,
            'Baseline_FNvol_mm3': baseline_fn,
            'Corrupted_FNvol_mm3': corrupted_fn,
            'FNvol_Delta_mm3': fn_delta
        })
        
        print(f"\n{patient_id}:")
        print(f"  Baseline Dice: {baseline_dice:.4f} | Corrupted Dice: {corrupted_dice:.4f} (Δ {dice_delta:+.4f})")
        print(f"  Baseline FPvol: {baseline_fp:.0f} | Corrupted FPvol: {corrupted_fp:.0f} (Δ {fp_delta:+.0f})")
        print(f"  Baseline FNvol: {baseline_fn:.0f} | Corrupted FNvol: {corrupted_fn:.0f} (Δ {fn_delta:+.0f})")

# Save comparison results
df = pd.DataFrame(results_list)
df.to_csv(OUTPUT_COMPARISON_CSV, index=False)

print("\n" + "=" * 60)
print("COMPARISON SUMMARY STATISTICS")
print("=" * 60)

print("\nBaseline Metrics:")
print(f"  Mean Dice: {df['Baseline_Dice'].mean():.4f} ± {df['Baseline_Dice'].std():.4f}")
print(f"  Mean FPvol: {df['Baseline_FPvol_mm3'].mean():.0f} ± {df['Baseline_FPvol_mm3'].std():.0f} mm³")
print(f"  Mean FNvol: {df['Baseline_FNvol_mm3'].mean():.0f} ± {df['Baseline_FNvol_mm3'].std():.0f} mm³")

print("\nCorrupted Metrics:")
print(f"  Mean Dice: {df['Corrupted_Dice'].mean():.4f} ± {df['Corrupted_Dice'].std():.4f}")
print(f"  Mean FPvol: {df['Corrupted_FPvol_mm3'].mean():.0f} ± {df['Corrupted_FPvol_mm3'].std():.0f} mm³")
print(f"  Mean FNvol: {df['Corrupted_FNvol_mm3'].mean():.0f} ± {df['Corrupted_FNvol_mm3'].std():.0f} mm³")

print("\nImpact of Corruption (Corrupted - Baseline):")
print(f"  Δ Dice: {df['Dice_Delta'].mean():+.4f} ± {df['Dice_Delta'].std():.4f}")
print(f"  Δ FPvol: {df['FPvol_Delta_mm3'].mean():+.0f} ± {df['FPvol_Delta_mm3'].std():.0f} mm³")
print(f"  Δ FNvol: {df['FNvol_Delta_mm3'].mean():+.0f} ± {df['FNvol_Delta_mm3'].std():.0f} mm³")

print(f"\n✓ Comparison results saved to {OUTPUT_COMPARISON_CSV}")
print("\n" + "=" * 60)
print("✓ CORRUPTION PIPELINE COMPLETE")
print("=" * 60)
