import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def optimize_thresholds(y_true, y_proba, target_matrix=None, base_threshold=None, plot=True):
    """
    Performs a grid search over threshold pairs (T1, T2) to find the best decision boundaries
    for converting probabilities into 3 prediction categories (0, 1, 2) such that the resulting
    normalized 2x3 confusion matrix closely matches a desired target matrix.

    Categories:
        - Predict 0 if prob < T1
        - Predict 1 if T1 <= prob < T2
        - Predict 2 if prob >= T2

    Parameters:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_proba (np.ndarray): Predicted probabilities for the positive class (class 1).
        target_matrix (np.ndarray): Desired 2x3 normalized confusion matrix. If None, uses default.
        base_threshold (float): Threshold around which T1 and T2 are searched. Default is 0.5.
        plot (bool): Whether to plot the best normalized confusion matrix.

    Returns:
        best_T1 (float): Lower decision threshold.
        best_T2 (float): Upper decision threshold.
        best_conf (np.ndarray): Best normalized 2x3 confusion matrix found.
        best_mse (float): Weighted mean squared error between best_conf and target_matrix.
    """

    # Use default target matrix if none provided
    if target_matrix is None:
        target_matrix = np.array([
            [0.7, 0.2, 0.1],  # Desired distribution for true label 0
            [0.1, 0.2, 0.7]   # Desired distribution for true label 1
        ])

    # Default base threshold if not provided
    if base_threshold is None:
        base_threshold = 0.5

    # Calculate class weights to balance MSE contribution
    count_0 = np.sum(y_true == 0)
    count_1 = np.sum(y_true == 1)
    weight_0 = 1.0 / count_0
    weight_1 = 1.0 / count_1
    total = weight_0 + weight_1
    weight_0 /= total
    weight_1 /= total

    # Initialize tracking variables
    best_mse = float('inf')
    best_T1, best_T2 = None, None
    best_conf = None

    # Define search grid around base threshold
    T1_range = np.linspace(base_threshold / 10, base_threshold, 100)
    T2_range = np.linspace(base_threshold, base_threshold * 10, 100)

    # Grid search over (T1, T2)
    for T1 in T1_range:
        for T2 in T2_range:
            if T2 <= T1:
                continue  # Enforce T2 > T1

            # Classify based on 3-bin logic
            y_pred = np.select(
                [y_proba < T1, y_proba < T2],
                [0, 1],
                default=2
            )

            # Build raw 2x3 confusion matrix
            cm = np.zeros((2, 3), dtype=int)
            for yt, yp in zip(y_true, y_pred):
                cm[yt, yp] += 1

            # Normalize each row of the confusion matrix
            cm_normalized = cm.astype(float)
            for i in range(2):
                row_sum = cm_normalized[i].sum()
                if row_sum > 0:
                    cm_normalized[i] /= row_sum

            # Weighted mean squared error vs. target matrix
            mse = (
                weight_0 * np.mean((cm_normalized[0] - target_matrix[0]) ** 2) +
                weight_1 * np.mean((cm_normalized[1] - target_matrix[1]) ** 2)
            )

            # Update best if this is the lowest error so far
            if mse < best_mse:
                best_mse = mse
                best_T1, best_T2 = T1, T2
                best_conf = cm_normalized.copy()

    # Optional plot
    if plot and best_conf is not None:
        plt.figure(figsize=(6, 4))
        sns.heatmap(best_conf, annot=True, fmt=".2f", cmap="coolwarm", cbar=False,
                    xticklabels=['Pred 0', 'Pred 1', 'Pred 2'],
                    yticklabels=['True 0', 'True 1'])
        plt.title("Best Normalized Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.show()

    return best_T1, best_T2, best_conf, best_mse
