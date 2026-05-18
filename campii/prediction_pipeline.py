import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
from brats import AdultGliomaPreTreatmentSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms, Backends

SCRIPT_DIR = Path(__file__).resolve().parent
CORRUPTED_DATA_DIR = SCRIPT_DIR / "data_corrupted"
AUGMENTED_DATA_DIR = SCRIPT_DIR / "data_augmented_shuffle"
CORRUPTED_RESULTS_DIR = SCRIPT_DIR / "corrupted_results"
AUGMENTED_RESULTS_DIR = SCRIPT_DIR / "augmented_results"
LOG_DIR = SCRIPT_DIR / "prediction_logs"

# Make output directories
CORRUPTED_RESULTS_DIR.mkdir(exist_ok=True)
AUGMENTED_RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def check_docker_available() -> bool:
    docker_path = shutil.which("docker")
    if docker_path is None:
        return False
    return True


def get_patient_ids(base_dir: Path):
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS20_Training")])


def infer_patient(input_files: dict[str, Path], output_file: Path, log_file: Path):
    segmenter = AdultGliomaPreTreatmentSegmenter(
        algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_1,
        force_cpu=False,
    )
    segmenter.infer_single(
        t1n=str(input_files["t1"]),
        t1c=str(input_files["t1ce"]),
        t2w=str(input_files["t2"]),
        t2f=str(input_files["flair"]),
        output_file=str(output_file),
        log_file=str(log_file),
        backend=Backends.DOCKER,
    )


def run_corrupted_predictions():
    print("Running corrupted predictions...")
    patient_ids = get_patient_ids(CORRUPTED_DATA_DIR)
    for patient_id in patient_ids:
        patient_dir = CORRUPTED_DATA_DIR / patient_id
        output_file = CORRUPTED_RESULTS_DIR / f"{patient_id}_corrupted_seg.nii.gz"
        log_file = LOG_DIR / f"{patient_id}_corrupted.log"

        inputs = {
            "t1": patient_dir / f"{patient_id}_t1.nii.gz",
            "t1ce": patient_dir / f"{patient_id}_t1ce.nii.gz",
            "t2": patient_dir / f"{patient_id}_t2.nii.gz",
            "flair": patient_dir / f"{patient_id}_flair.nii.gz",
        }

        if not all(p.exists() for p in inputs.values()):
            print(f"Skipping {patient_id}: missing corrupted input files")
            continue

        try:
            infer_patient(inputs, output_file, log_file)
            print(f"Saved corrupted prediction for {patient_id}")
        except Exception as exc:
            print(f"Failed prediction for {patient_id}: {exc}")


def run_augmented_predictions(augmentation_index: int = 1):
    print(f"Running augmented predictions for aug{augmentation_index}...")
    patient_ids = get_patient_ids(AUGMENTED_DATA_DIR)
    for patient_id in patient_ids:
        patient_dir = AUGMENTED_DATA_DIR / patient_id
        output_file = AUGMENTED_RESULTS_DIR / f"{patient_id}_aug{augmentation_index}_seg.nii.gz"
        log_file = LOG_DIR / f"{patient_id}_aug{augmentation_index}.log"

        inputs = {
            "t1": patient_dir / f"{patient_id}_aug{augmentation_index}_t1.nii.gz",
            "t1ce": patient_dir / f"{patient_id}_aug{augmentation_index}_t1ce.nii.gz",
            "t2": patient_dir / f"{patient_id}_aug{augmentation_index}_t2.nii.gz",
            "flair": patient_dir / f"{patient_id}_aug{augmentation_index}_flair.nii.gz",
        }

        if not all(p.exists() for p in inputs.values()):
            print(f"Skipping {patient_id}: missing augmented input files for aug{augmentation_index}")
            continue

        try:
            infer_patient(inputs, output_file, log_file)
            print(f"Saved augmented prediction for {patient_id} aug{augmentation_index}")
        except Exception as exc:
            print(f"Failed prediction for {patient_id} aug{augmentation_index}: {exc}")


def main():
    if not check_docker_available():
        print("Docker is not available on this system.")
        print("Install Docker Desktop or Docker Engine, then rerun this script.")
        sys.exit(1)

    run_corrupted_predictions()
    run_augmented_predictions(augmentation_index=1)


if __name__ == "__main__":
    main()
