"""
Visualization Guide: Comparing Baseline vs Corrupted Data

This script helps you understand the visual differences between:
1. Baseline (clean) segmentations
2. Corrupted (T1↔T2 swapped) data

QUICK START:

For each patient, open these files in VS Code with NiiVue:

PATIENT 1 (BraTS20_Training_001):
  📊 BASELINE (Clean Data):
    - Input T1:    campii/data_test/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz
    - Input T2:    campii/data_test/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz
    - Prediction:  campii/baseline_results/BraTS20_Training_001_baseline_seg.nii.gz
    - Ground Truth: campii/data_test/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz

  🔄 CORRUPTED (T1↔T2 Swapped):
    - Corrupted T1: campii/data_corrupted/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz
      ⚠️ Note: This file now contains T2 data (bright CSF)
    - Corrupted T2: campii/data_corrupted/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz
      ⚠️ Note: This file now contains T1 data (dimmer)

KEY OBSERVATIONS:
  - Baseline T1 is typically darker (lower signal: 360.5)
  - Baseline T2 is brighter, especially in CSF regions (signal: 195.1)
  - When swapped, the model sees REVERSED contrast patterns
  - This typically causes FALSE POSITIVES in bright CSF regions

WHAT TO LOOK FOR:
  1. Compare baseline vs corrupted segmentations
  2. Note areas where corrupted prediction has extra "tumor" in ventricles (CSF)
  3. These are FALSE POSITIVES due to T2's bright CSF signal being misinterpreted

METRICS TO CHECK:
  - Look at corruption_analysis.csv for quantitative proof
  - Expected: FPvol increases significantly after T1↔T2 swap
"""

import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_csv = os.path.join(script_dir, "corruption_analysis.csv")

if os.path.exists(analysis_csv):
    df = pd.read_csv(analysis_csv)
    print("=" * 70)
    print("BASELINE METRICS (Clean Data)")
    print("=" * 70)
    print(f"Mean Dice:   {df['Baseline_Dice'].mean():.4f}")
    print(f"Mean FPvol:  {df['Baseline_FPvol_mm3'].mean():.0f} mm³")
    print(f"Mean FNvol:  {df['Baseline_FNvol_mm3'].mean():.0f} mm³")
    
    print("\n" + "=" * 70)
    print("CHANNEL STATISTICS")
    print("=" * 70)
    print(f"Mean T1 Signal: {df['Baseline_T1_Mean'].mean():.1f}")
    print(f"Mean T2 Signal: {df['Baseline_T2_Mean'].mean():.1f}")
    print(f"Signal Ratio (T2/T1): {df['Baseline_T2_Mean'].mean() / df['Baseline_T1_Mean'].mean():.2f}x")
    
    print("\n" + "=" * 70)
    print("TOP 5 PATIENTS TO INSPECT")
    print("=" * 70)
    top_5 = df.nlargest(5, 'Signal_Difference_T1_vs_T2')
    for idx, row in top_5.iterrows():
        print(f"{row['Patient_ID']}: T1 vs T2 diff = {row['Signal_Difference_T1_vs_T2']:.0f}")
        print(f"  → Files: data_test/{row['Patient_ID']} vs data_corrupted/{row['Patient_ID']}")
else:
    print("❌ corruption_analysis.csv not found. Run corruption_analysis.py first.")
