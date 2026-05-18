import os
import json
import random
import nibabel as nib
from itertools import permutations

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, "data_test")
AUGMENTED_DATA_DIR = os.path.join(SCRIPT_DIR, "data_augmented_shuffle")
AUGMENTATION_RECORD = os.path.join(SCRIPT_DIR, "channel_shuffle_log.json")
NUM_AUGMENTATIONS_PER_PATIENT = 5

os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)

# Define the four BraTS modalities in canonical order
CANONICAL_MODALITIES = ["t1", "t1ce", "t2", "flair"]

# Use a reproducible seed so the augmentation set is stable
random.seed(42)


def load_modalities(patient_dir, patient_id):
    """Load the four BraTS modalities for a single patient."""
    paths = {}
    for modality in CANONICAL_MODALITIES:
        filename = f"{patient_id}_{modality}.nii.gz"
        full_path = os.path.join(patient_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing modality file: {full_path}")
        paths[modality] = nib.load(full_path)
    return paths


def save_augmentation(patient_aug_dir, patient_id, aug_index, mapping, loaded_images):
    """Save a single augmented patient with a given channel mapping."""
    os.makedirs(patient_aug_dir, exist_ok=True)
    record = {
        "patient_id": patient_id,
        "augmentation_index": aug_index,
        "channel_mapping": mapping,
    }

    for target_modality, source_modality in mapping.items():
        source_img = loaded_images[source_modality]
        output_filename = f"{patient_id}_aug{aug_index}_{target_modality}.nii.gz"
        output_path = os.path.join(patient_aug_dir, output_filename)
        nib.save(nib.Nifti1Image(source_img.get_fdata(), source_img.affine, source_img.header), output_path)

    return record


def create_channel_augmentations():
    """Create shuffled-channel augmentations for the BraTS patients."""
    patient_folders = sorted([f for f in os.listdir(BASE_DATA_DIR) if f.startswith("BraTS20_Training")])
    augmentation_log = []

    for patient_id in patient_folders:
        patient_dir = os.path.join(BASE_DATA_DIR, patient_id)
        patient_aug_dir = os.path.join(AUGMENTED_DATA_DIR, patient_id)
        loaded_images = load_modalities(patient_dir, patient_id)

        # Generate a stable set of random permutations for each patient
        all_permutations = list(permutations(CANONICAL_MODALITIES))
        random.shuffle(all_permutations)

        # Keep the identity mapping to preserve the clean baseline as a control case
        identity_mapping = {modality: modality for modality in CANONICAL_MODALITIES}
        augmentation_log.append(save_augmentation(patient_aug_dir, patient_id, 0, identity_mapping, loaded_images))

        # Add a small number of shuffled augmentations while avoiding duplicates and identity
        used_mappings = {tuple(identity_mapping[mod] for mod in CANONICAL_MODALITIES)}
        aug_count = 0
        aug_index = 1

        for perm in all_permutations:
            if aug_count >= NUM_AUGMENTATIONS_PER_PATIENT:
                break
            if tuple(perm) in used_mappings:
                continue

            mapping = {target: source for target, source in zip(CANONICAL_MODALITIES, perm)}
            record = save_augmentation(patient_aug_dir, patient_id, aug_index, mapping, loaded_images)
            augmentation_log.append(record)
            used_mappings.add(tuple(perm))
            aug_count += 1
            aug_index += 1

        print(f"Generated {aug_count + 1} augmentations for {patient_id}")

    with open(AUGMENTATION_RECORD, "w") as f:
        json.dump(augmentation_log, f, indent=2)

    print("\nChannel-shuffling augmentation complete.")
    print(f"Augmented data saved to: {AUGMENTED_DATA_DIR}")
    print(f"Augmentation log saved to: {AUGMENTATION_RECORD}")


if __name__ == "__main__":
    create_channel_augmentations()
