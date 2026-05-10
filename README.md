BraTS sequence


1. Establish the Baseline Pipeline: prove the model works as intended on clean data. Compute Baseline Metrics, Dice Similarity Coefficient (DSC). Visual check.
2. Implementing the corruption. (e.g., swapping Index 0 [T1] with Index 2 [T2])
3. Documentation. Identify False Positives. Failure documentation: Show Baseline Image and corrupted image, quantitative proof. e.g FPvol (False Positive Volume) increasing significantly after the swap.e.g show areas where the model confused T2's bright fluid (CSF) with tumor tissue.
4. Implement fix: Channel-Shuffling Augmentation. reevaluate. We want to see the False Positive Volume (FPvol) return to Baseline levels.
