

import sys
import types

# 1. FIX: Mock 'pwd' for Windows compatibility
if sys.platform == "win32":
    mock_pwd = types.ModuleType("pwd")
    def getpwuid(uid):
        # This provides a dummy user object so the library doesn't crash
        return types.SimpleNamespace(pw_name='laura') 
    mock_pwd.getpwuid = getpwuid
    sys.modules["pwd"] = mock_pwd



import os
from brats import AdultGliomaPreTreatmentSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms

# 1. Configuration based on your folder structure
BASE_DATA_DIR = "data_test"
OUTPUT_DIR = "baseline_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Initialize the segmenter (using the 2023 winning algorithm)
segmenter = AdultGliomaPreTreatmentSegmenter(
    algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_1
)

# 3. Get the list of patient folders
patient_folders = [f for f in os.listdir(BASE_DATA_DIR) if f.startswith("BraTS20_Training")]
# Start with a small subset (e.g., first 5) for Week 1
subset = sorted(patient_folders)[:20]

print(f"Found {len(subset)} patients for baseline testing.")

for patient_id in subset:
    print(f"\n--- Processing {patient_id} ---")
    
    # Construct full paths to the NIfTI files
    patient_path = os.path.join(BASE_DATA_DIR, patient_id)
    
    t1n_path = os.path.join(patient_path, f"{patient_id}_t1.nii.gz")
    t1c_path = os.path.join(patient_path, f"{patient_id}_t1ce.nii.gz")
    t2w_path = os.path.join(patient_path, f"{patient_id}_t2.nii.gz")
    t2f_path = os.path.join(patient_path, f"{patient_id}_flair.nii.gz")
    
    output_file = os.path.join(OUTPUT_DIR, f"{patient_id}_baseline_seg.nii.gz")

    try:
        # Run the orchestrator inference
        segmenter.infer_single(
            t1n=t1n_path, # Native T1
            t1c=t1c_path, # T1ce (contrast)
            t2w=t2w_path, # T2
            t2f=t2f_path, # FLAIR
            output_file=output_file
        )
        print(f"Success: Baseline saved to {output_file}")
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")

print("\nWeek 1 Baseline Inference Complete.")