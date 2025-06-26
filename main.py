# main.py

from call_peaks import call_peak_events
from get_sequences import build_feature_matrix, extract_context_sequences
from motif_analysis import cluster_and_extract_motifs, add_similarity_features
from train_model import train_optuna_xgboost_model
from optimize_thresholds import optimize_thresholds
from visualization import (
    plot_residual_peaks,
    #plot_roc_curve,
    #plot_confusion_matrix,
    #plot_fragility_prediction
)

import numpy as np
import pandas as pd

# --- Step 1: Call Peaks ---
ticker = "AAPL"  # Replace or make dynamic
df_resid, df_merged_peaks, df_summits_filtered = call_peak_events(ticker)

# --- Step 2: Extract Sequence Features ---
df_features = build_feature_matrix(ticker)
df_sequences_all, df_scaled_features, df_summits = extract_context_sequences(
    df_features, df_summits_filtered
)

# --- Step 3: Motif Clustering + Similarity Features ---
df_enrichment, df_peak_clusters, ctrl_labels, centroid_motifs, fdr_adj = cluster_and_extract_motifs(
    df_sequences_all
)
df_model = add_similarity_features(
    df_features=df_features,
    df_scaled_features=df_scaled_features,
    centroid_motifs=centroid_motifs,
    fdr_adj=fdr_adj
)

# --- Step 4: Model Training ---
significant_clusters = df_enrichment[df_enrichment['Significant']]['Cluster_ID'].tolist()
results, df_test = train_optuna_xgboost_model(
    df_model=df_model,
    df_summits=df_summits_filtered,
    significant_clusters=significant_clusters,
    n_trials=1000
)

# --- Step 5: Threshold Optimization ---
y_true = np.array(results['y_test'])
y_proba = np.array(results['y_proba'])
target_matrix = np.array([
    [0.7, 0.2, 0.1],  # True 0
    [0.1, 0.2, 0.7]   # True 1
])

best_T1, best_T2, best_conf, best_mse = optimize_thresholds(y_true, y_proba, target_matrix)

# --- Step 6: Output Results ---
print(f"Best thresholds:")
print(f"  T1 = {best_T1:.4f}")
print(f"  T2 = {best_T2:.4f}")
print(f"  Weighted MSE to target matrix = {best_mse:.6f}")
print("\n--- Normalized Confusion Matrix ---")
print(pd.DataFrame(best_conf, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1", "Pred 2"]))

# --- Step 7: Visualizations ---

# 1. Residual volatility and peaks (1 year view)
plot_residual_peaks(df_resid, df_summits_filtered, df_merged_peaks, window='1y')

# 2. ROC Curve
plot_roc_curve(y_true, y_proba, best_thresh=results['best_threshold'])

# 3. Confusion Matrix
plot_confusion_matrix(best_conf)

# 4. Fragility score prediction vs ground truth
plot_fragility_prediction(
    df_test=df_test,
    y_proba=y_proba,
    y_true=y_true,
    df_merged_peaks=df_merged_peaks,
    best_T1=best_T1,
    best_T2=best_T2,
    window_days=90
)
