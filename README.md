BraTS sequence


1. Establish the Baseline Pipeline: prove the model works as intended on clean data. Compute Baseline Metrics, Dice Similarity Coefficient (DSC). Visual check.
2. Implementing the corruption. (e.g., swapping Index 0 [T1] with Index 2 [T2])
3. Documentation. Identify False Positives. Failure documentation: Show Baseline Image and corrupted image, quantitative proof. e.g FPvol (False Positive Volume) increasing significantly after the swap.e.g show areas where the model confused T2's bright fluid (CSF) with tumor tissue.
4. Implement fix: Channel-Shuffling Augmentation. reevaluate. We want to see the False Positive Volume (FPvol) return to Baseline levels.

Summary table for the baseline, corrupted and fixed cases, parameters:
1.Dice Similarity Coefficient (DSC): It measures the overlap between the model's predicted tumor mask and the ground-truth mask provided in the dataset. Report this for three overlapping regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET). A high-quality baseline for BraTS usually shows a DSC between 0.65 and 0.90.

2.False Positive Volume (FPvol):  This is the total volume (measured in mL or mm3) of voxels the model incorrectly labeled as tumor when they were actually healthy tissue. In the clean baseline, this should be low, as the model should not see tumors in healthy brain fluid (CSF).

3.False Negative Volume (FNvol): This is the total volume of actual tumor tissue that the model missed or ignored. We do not want the model to stop detecting real tumors.
