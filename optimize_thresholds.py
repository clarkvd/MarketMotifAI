import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def optimize_thresholds(y_true, y_proba, target_matrix=None, base_threshold=None, plot=True):
    """
    Grid search to find optimal T1 and T2 thresholds that produce a normalized
    confusion matrix closest to the target matrix using weighted MSE.

    Parameters:
    - y_true (np.ndarray): Ground truth binary labels (0 or 1).
    - y_proba (np.ndarray): Model output probabilities for class 1.
    - target_matrix (np.ndarray): 2x3 matrix with desired prediction distributions per class.
    - base_threshold (float): Baseline threshold to define T1/T2 search space (e.g., ROC-optimal).
    - plot (bool): If True, display heatmap of the best normalized confusion matrix.

    Returns:
    - best_T1 (float)
    - best_T2 (float)
    - best_conf (np.ndarray): Best normalized confusion matrix.
    - best_mse (float): Weighted MSE between best_conf and target_matrix.
    """
    if target_matrix is None:
        target_matrix = np.array([
            [0.7, 0.2, 0.1],  # Desired for true label 0
            [0.1, 0.2, 0.7]   # Desired for true label 1
        ])
    if base_threshold is None:
        base_threshold = 0.5  # default fallback

    # Class weighting
    count_0 = np.sum(y_true == 0)
    count_1 = np.sum(y_true == 1)
    weight_0 = 1.0 / count_0
    weight_1 = 1.0 / count_1
    total = weight_0 + weight_1
    weight_0 /= total
    weight_1 /= total

    best_mse = float('inf')
    best_T1, best_T2 = None, None
    best_conf = None

    T1_range = np.linspace(base_threshold / 10, base_threshold, 100)
    T2_range = np.linspace(base_threshold, base_threshold * 10, 100)

    for T1 in T1_range:
        for T2 in T2_range:
            if T2 <= T1:
                continue

            y_pred = np.select(
                [y_proba < T1, y_proba < T2],
                [0, 1],
                default=2
            )

            cm = np.zeros((2, 3), dtype=int)
            for yt, yp in zip(y_true, y_pred):
                cm[yt, yp] += 1

            cm_normalized = cm.astype(float)
            for i in range(2):
                row_sum = cm_normalized[i].sum()
                if row_sum > 0:
                    cm_normalized[i] /= row_sum

            mse = (
                weight_0 * np.mean((cm_normalized[0] - target_matrix[0]) ** 2) +
                weight_1 * np.mean((cm_normalized[1] - target_matrix[1]) ** 2)
            )

            if mse < best_mse:
                best_mse = mse
                best_T1, best_T2 = T1, T2
                best_conf = cm_normalized.copy()

    '''if best_T1 is not None:
        print(f"Best thresholds:")
        print(f"  T1 = {best_T1:.4f}")
        print(f"  T2 = {best_T2:.4f}")
        print(f"  Weighted MSE to target matrix = {best_mse:.6f}")
        print("\n--- Normalized Confusion Matrix ---")
        print(pd.DataFrame(best_conf, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1", "Pred 2"]))

        if plot:
            sns.heatmap(best_conf, annot=True, cmap='Blues', fmt='.2f',
                        xticklabels=['Pred 0', 'Pred 1', 'Pred 2'],
                        yticklabels=['True 0', 'True 1'])
            plt.title("Normalized Confusion Matrix (Weighted MSE Optimized)")
            plt.tight_layout()
            plt.show()
    else:
        print("No valid thresholds found.")'''

    return best_T1, best_T2, best_conf, best_mse
