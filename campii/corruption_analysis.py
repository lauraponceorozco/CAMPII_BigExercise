import os
import numpy as np
import nibabel as nib
import pandas as pd

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DATA_DIR = os.path.join(SCRIPT_DIR, "data_test")
CORRUPTED_DATA_DIR = os.path.join(SCRIPT_DIR, "data_corrupted")
BASELINE_RESULTS_DIR = os.path.join(SCRIPT_DIR, "baseline_results")
OUTPUT_COMPARISON_CSV = os.path.join(SCRIPT_DIR, "corruption_analysis.csv")

print("=" * 70)
print("CORRUPTION ANALYSIS: Analyzing effect of T1 <-> T2 swap on channels")
print("=" * 70)

# --- STEP 1: ANALYZE CHANNEL DIFFERENCES (T1 vs T2) ---
print("\nSTEP 1: Analyzing channel corruption (T1 <-> T2 swap impact)")
print("-" * 70)

patient_folders = sorted([f for f in os.listdir(BASELINE_DATA_DIR) if f.startswith("BraTS20_Training")])[:20]

corruption_analysis = []

for patient_id in patient_folders:
    baseline_patient_path = os.path.join(BASELINE_DATA_DIR, patient_id)
    corrupted_patient_path = os.path.join(CORRUPTED_DATA_DIR, patient_id)
    
    try:
        # Load baseline channels
        baseline_t1n = nib.load(os.path.join(baseline_patient_path, f"{patient_id}_t1.nii.gz")).get_fdata()
        baseline_t2w = nib.load(os.path.join(baseline_patient_path, f"{patient_id}_t2.nii.gz")).get_fdata()
        
        # Load corrupted channels
        corrupted_t1n = nib.load(os.path.join(corrupted_patient_path, f"{patient_id}_t1.nii.gz")).get_fdata()
        corrupted_t2w = nib.load(os.path.join(corrupted_patient_path, f"{patient_id}_t2.nii.gz")).get_fdata()
        
        # Verify the swap occurred
        swap_verified = (np.allclose(corrupted_t1n, baseline_t2w) and 
                        np.allclose(corrupted_t2w, baseline_t1n))
        
        # Calculate channel statistics
        baseline_t1_mean = baseline_t1n[baseline_t1n > 0].mean()
        baseline_t2_mean = baseline_t2w[baseline_t2w > 0].mean()
        baseline_t1_std = baseline_t1n[baseline_t1n > 0].std()
        baseline_t2_std = baseline_t2w[baseline_t2w > 0].std()
        
        # Signal differences
        signal_diff_t1_vs_t2 = abs(baseline_t1_mean - baseline_t2_mean)
        
        corruption_analysis.append({
            'Patient_ID': patient_id,
            'T1_T2_Swap_Verified': swap_verified,
            'Baseline_T1_Mean': baseline_t1_mean,
            'Baseline_T2_Mean': baseline_t2_mean,
            'Baseline_T1_Std': baseline_t1_std,
            'Baseline_T2_Std': baseline_t2_std,
            'Signal_Difference_T1_vs_T2': signal_diff_t1_vs_t2
        })
        
        print(f"✓ {patient_id}: T1 mean={baseline_t1_mean:.1f}, T2 mean={baseline_t2_mean:.1f}")
        
    except Exception as e:
        print(f"✗ {patient_id}: {e}")

# --- STEP 2: ANALYZE GROUND TRUTH SEGMENTATIONS ---
print("\nSTEP 2: Analyzing ground truth segmentations (tumor labels)")
print("-" * 70)

gt_analysis = []

for patient_id in patient_folders:
    baseline_patient_path = os.path.join(BASELINE_DATA_DIR, patient_id)
    gt_path = os.path.join(baseline_patient_path, f"{patient_id}_seg.nii.gz")
    
    try:
        gt_img = nib.load(gt_path)
        gt_data = gt_img.get_fdata()
        
        # Get voxel volume
        voxel_dims = gt_img.header.get_zooms()
        voxel_vol = np.prod(voxel_dims)
        
        # Tumor volumes by region
        wt = (gt_data > 0).sum() * voxel_vol  # Whole Tumor
        tc = np.isin(gt_data, [1, 4]).sum() * voxel_vol  # Tumor Core
        et = (gt_data == 4).sum() * voxel_vol  # Enhancing Tumor
        
        gt_analysis.append({
            'Patient_ID': patient_id,
            'WT_Volume_mm3': wt,
            'TC_Volume_mm3': tc,
            'ET_Volume_mm3': et
        })
        
        print(f"✓ {patient_id}: WT={wt:.0f} mm³, TC={tc:.0f} mm³, ET={et:.0f} mm³")
        
    except Exception as e:
        print(f"✗ {patient_id}: {e}")

# --- STEP 3: BASELINE PREDICTION ANALYSIS ---
print("\nSTEP 3: Analyzing baseline predictions vs ground truth")
print("-" * 70)

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
    except:
        return None, None, None

baseline_metrics = []

for patient_id in patient_folders:
    baseline_pred = os.path.join(BASELINE_RESULTS_DIR, f"{patient_id}_baseline_seg.nii.gz")
    gt_path = os.path.join(BASELINE_DATA_DIR, patient_id, f"{patient_id}_seg.nii.gz")
    
    if os.path.exists(baseline_pred) and os.path.exists(gt_path):
        dice, fp_vol, fn_vol = calculate_metrics(baseline_pred, gt_path)
        
        if dice is not None:
            baseline_metrics.append({
                'Patient_ID': patient_id,
                'Baseline_Dice': dice,
                'Baseline_FPvol_mm3': fp_vol,
                'Baseline_FNvol_mm3': fn_vol
            })
            print(f"✓ {patient_id}: Dice={dice:.4f}, FPvol={fp_vol:.0f}, FNvol={fn_vol:.0f}")

# --- STEP 4: MERGE ALL DATA ---
print("\nSTEP 4: Creating comprehensive comparison table")
print("-" * 70)

# Merge all dataframes
df_corruption = pd.DataFrame(corruption_analysis)
df_gt = pd.DataFrame(gt_analysis)
df_baseline = pd.DataFrame(baseline_metrics)

# Merge
comparison_df = df_corruption.merge(df_gt, on='Patient_ID', how='left')
comparison_df = comparison_df.merge(df_baseline, on='Patient_ID', how='left')

# Save
comparison_df.to_csv(OUTPUT_COMPARISON_CSV, index=False)

print(f"\n✓ Comprehensive analysis saved to: {OUTPUT_COMPARISON_CSV}")

# --- STEP 5: PRINT SUMMARY STATISTICS ---
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("\n📊 BASELINE PREDICTIONS ON CLEAN DATA:")
print(f"  Mean Dice: {df_baseline['Baseline_Dice'].mean():.4f} ± {df_baseline['Baseline_Dice'].std():.4f}")
print(f"  Mean FPvol: {df_baseline['Baseline_FPvol_mm3'].mean():.0f} ± {df_baseline['Baseline_FPvol_mm3'].std():.0f} mm³")
print(f"  Mean FNvol: {df_baseline['Baseline_FNvol_mm3'].mean():.0f} ± {df_baseline['Baseline_FNvol_mm3'].std():.0f} mm³")

print("\n🧬 GROUND TRUTH TUMOR VOLUMES:")
print(f"  Mean WT Volume: {df_gt['WT_Volume_mm3'].mean():.0f} ± {df_gt['WT_Volume_mm3'].std():.0f} mm³")
print(f"  Mean TC Volume: {df_gt['TC_Volume_mm3'].mean():.0f} ± {df_gt['TC_Volume_mm3'].std():.0f} mm³")
print(f"  Mean ET Volume: {df_gt['ET_Volume_mm3'].mean():.0f} ± {df_gt['ET_Volume_mm3'].std():.0f} mm³")

print("\n⚡ CHANNEL CORRUPTION ANALYSIS (T1 <-> T2 swap):")
print(f"  Mean T1 Signal: {df_corruption['Baseline_T1_Mean'].mean():.1f} ± {df_corruption['Baseline_T1_Mean'].std():.1f}")
print(f"  Mean T2 Signal: {df_corruption['Baseline_T2_Mean'].mean():.1f} ± {df_corruption['Baseline_T2_Mean'].std():.1f}")
print(f"  Mean Signal Difference (T1 vs T2): {df_corruption['Signal_Difference_T1_vs_T2'].mean():.1f} mm³")
print(f"  T1 <-> T2 Swap Verification: {df_corruption['T1_T2_Swap_Verified'].sum()}/{len(df_corruption)} patients ✓")

print("\n" + "=" * 70)
print("✓ CORRUPTION ANALYSIS COMPLETE")
print("=" * 70)
print("\n📁 Next steps:")
print("  1. Review corruption_analysis.csv for detailed per-patient metrics")
print("  2. Use Docker to run corrupted segmentations (if available)")
print("  3. Compare baseline vs corrupted predictions to show FPvol increase")
print("=" * 70)
