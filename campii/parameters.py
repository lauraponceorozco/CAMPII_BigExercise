import os
import nibabel as nib
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
PRED_DIR = 'baseline_results'
GT_PARENT_DIR = 'data_test'
OUTPUT_CSV = 'segmentation_metrics.csv'

def calculate_metrics(pred_path, gt_path):
    # Load images
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)
    
    # Get data and convert to binary (Whole Tumor)
    pred = pred_img.get_fdata() > 0
    gt = gt_img.get_fdata() > 0
    
    # Get voxel volume for FPvol/FNvol
    voxel_dims = pred_img.header.get_zooms()
    voxel_vol = np.prod(voxel_dims) # mm^3
    
    # Calculate TP, FP, FN
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    
    # 1. Dice Coefficient
    dice = (2. * tp) / (2. * tp + fp + fn + 1e-8)
    
    # 2. Volumes (mm^3)
    fp_vol = fp * voxel_vol
    fn_vol = fn * voxel_vol
    
    return dice, fp_vol, fn_vol

# --- MAIN LOOP ---
results_list = []

# Filter for the .nii.gz prediction files
pred_files = [f for f in os.listdir(PRED_DIR) if f.endswith('_seg.nii.gz')]

print(f"Found {len(pred_files)} patients. Starting evaluation...")

for f in pred_files:
    # Extract Patient ID (e.g., BraTS20_Training_001)
    patient_id = f.replace('_baseline_seg.nii.gz', '')
    
    # Define paths
    p_path = os.path.join(PRED_DIR, f)
    # Assumes GT is at: data_test/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz
    gt_path = os.path.join(GT_PARENT_DIR, patient_id, f"{patient_id}_seg.nii.gz")
    
    if os.path.exists(gt_path):
        d, fp, fn = calculate_metrics(p_path, gt_path)
        results_list.append({
            'Patient_ID': patient_id,
            'Dice': d,
            'FP_vol_mm3': fp,
            'FN_vol_mm3': fn
        })
    else:
        print(f"Warning: Ground Truth not found for {patient_id}")

# --- STATISTICS ---
df = pd.DataFrame(results_list)

# Calculate global stats
stats = df[['Dice', 'FP_vol_mm3', 'FN_vol_mm3']].agg(['mean', 'std', 'median']).transpose()

print("\n--- INDIVIDUAL RESULTS ---")
print(df.to_string(index=False))

print("\n--- SUMMARY STATISTICS ---")
print(stats)

# Save to CSV so you can put it in your report
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to {OUTPUT_CSV}")