"""
Full pipeline for time-series volatility fragility modeling across multiple tickers.

Steps:
0. Download macroeconomic indicators (sentiment, CPI, FOMC).
1. Detect volatility spike ("fragility") peaks for each ticker.
2. Extract contextual sequences and engineer features.
3. Cluster peak patterns into motif-like shapes using PCA + KMeans.
4. Assess motif enrichment against control windows (Fisher's exact test).
5. Train an XGBoost classifier with Optuna hyperparameter tuning.
6. Optimize post-hoc decision thresholds to shape output distributions.
7. Calculate SHAP values for explainability.
8. Summarize top enriched motifs and risk prediction for current day.
"""

from call_peaks import call_peak_events
from get_sequences import build_feature_matrix, extract_context_sequences
from motif_analysis import cluster_and_extract_motifs, add_similarity_features
from train_model import train_optuna_xgboost_model
from optimize_thresholds import optimize_thresholds

import numpy as np
import pandas as pd
import subprocess
import joblib
import shap
import matplotlib.pyplot as plt

# --- Step 0: Ensure today's macro data is available ---
subprocess.run(["python3", "download_todays_data.py"], check=True)

# --- Optional human-readable label map for motifs ---
feature_map = {
    'RSI': 'RSI (momentum)',
    'MACD_Diff': 'MACD divergence',
    'BBW': 'Bollinger Band Width',
    'Residual_Mean_5': '5-day mean residual',
    'Residual_Momentum_5': '5-day momentum residual',
    'Sentiment_Mean_3': '3-day average sentiment',
    'VIX_Mean_5': '5-day VIX average',
    'Volume_Mean_5': '5-day average volume'
}

def describe_day(day: int) -> str:
    """Returns a human-readable description for relative day offsets."""
    if day == 0:
        return "on the summit day"
    elif day > 0:
        return f"{day} day{'s' if day > 1 else ''} after the summit"
    else:
        return f"{abs(day)} day{'s' if abs(day) > 1 else ''} before the summit"

def summarize_centroid_readably(centroid_df, top_n=4):
    """
    Summarizes a centroid motif in human-readable language by identifying the
    most extreme features and when they occur.
    """
    summary = []
    for feature in centroid_df.columns:
        values = centroid_df[feature]
        idx_max = values.abs().idxmax()
        raw_val = values.loc[idx_max]
        description = (
            f"{feature_map.get(feature, feature)} is "
            f"{'elevated' if raw_val > 0 else 'depressed'} "
            f"{describe_day(idx_max)}"
        )
        summary.append((abs(raw_val), description))

    summary_sorted = sorted(summary, key=lambda x: x[0], reverse=True)
    return [desc for _, desc in summary_sorted[:top_n]]

# --- List of tickers to process ---
tickers = ["AAPL", "MSFT", "TSLA", "NVDA"]

# --- Store results across tickers ---
bundle_all = {}

# === PIPELINE LOOP ===
for ticker in tickers:
    print(f"\n🔍 Processing ticker: {ticker}")

    # Step 1: Call fragility peaks based on residual volatility
    df_resid, df_merged_peaks, df_summits_filtered = call_peak_events(ticker)

    # Step 2: Build feature matrix and extract context sequences
    df_features = build_feature_matrix(ticker)
    df_sequences_all, df_scaled_features, df_summits = extract_context_sequences(
        df_features, df_summits_filtered
    )

    # Step 3: Cluster motifs, extract enrichment, and add similarity features
    df_enrichment, df_peak_clusters, ctrl_labels, centroid_motifs, fdr_adj = cluster_and_extract_motifs(
        df_sequences_all
    )
    df_model = add_similarity_features(
        df_features=df_features,
        df_scaled_features=df_scaled_features,
        centroid_motifs=centroid_motifs,
        fdr_adj=fdr_adj
    )

    # Step 4: Train model using Optuna tuning
    significant_clusters = df_enrichment[df_enrichment['Significant']]['Cluster_ID'].tolist()
    print(df_enrichment[df_enrichment['Significant']])
    results, df_test = train_optuna_xgboost_model(
        df_model=df_model,
        df_summits=df_summits_filtered,
        significant_clusters=significant_clusters,
        n_trials=100
    )

    # Step 5: Optimize classification thresholds to meet target structure
    y_true = np.array(results['y_test'])
    y_proba = np.array(results['y_proba'])
    best_threshold = results['best_threshold']
    roc_auc = results['roc_auc']
    fpr = results['fpr']
    tpr = results['tpr']

    target_matrix = np.array([
        [0.7, 0.2, 0.1],  # True 0
        [0.1, 0.2, 0.7]   # True 1
    ])
    best_T1, best_T2, best_conf, best_mse = optimize_thresholds(
        y_true, y_proba, target_matrix, best_threshold
    )

    # Step 6: SHAP explainability
    model = results['model']
    X_test = results['X_test']
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    results['shap_values'] = shap_values.values.tolist()
    results['shap_feature_names'] = X_test.columns.tolist()

    # Step 7: Generate prediction for most recent day
    today_index = df_test.index[-1]
    today_proba = y_proba[df_test.index.get_loc(today_index)]
    today_date = str(df_test.loc[today_index, 'Date'])

    if today_proba < best_T1:
        today_risk = 'low'
    elif today_proba < best_T2:
        today_risk = 'moderate'
    else:
        today_risk = 'high'

    distance_to_boundary = min(abs(today_proba - best_T1), abs(today_proba - best_T2))
    confidence = round(100 - (distance_to_boundary * 100), 1)
    confidence = max(0, min(confidence, 100))

    today_prediction = {
        'date': today_date,
        'probability': float(today_proba),
        'risk_level': today_risk,
        'confidence': confidence
    }

    # Step 8: Summarize enriched motif content
    significant_info = []
    readable_clusters = []
    for cid in significant_clusters:
        row = df_enrichment[df_enrichment['Cluster_ID'] == cid].iloc[0]
        centroid_df = centroid_motifs[cid]
        readable = summarize_centroid_readably(centroid_df)

        significant_info.append({
            'Cluster_ID': cid,
            'FDR_Adjusted_P': row['FDR_Adjusted_P'],
            'Peak_Count': row['Peak_Count'],
            'Control_Count': row['Control_Count'],
            'Odds_Ratio': row['Odds_Ratio'],
            'Centroid_Motif_Matrix': centroid_df
        })

        readable_clusters.append({
            'Cluster_ID': cid,
            'FDR_Adjusted_P': row['FDR_Adjusted_P'],
            'Highlights': readable
        })

    # --- Save all output for this ticker ---
    bundle_all[ticker] = {
        'df_resid': df_resid,
        'df_merged_peaks': df_merged_peaks,
        'df_summits_filtered': df_summits_filtered,
        'results': results,
        'df_test': df_test,
        'best_T1': best_T1,
        'best_T2': best_T2,
        'best_conf': best_conf,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'df_enrichment': df_enrichment,
        'significant_clusters': significant_clusters,
        'significant_cluster_info': significant_info,
        'significant_cluster_summaries': readable_clusters,
        'today_prediction': today_prediction
    }

# === Final Save ===
joblib.dump(bundle_all, 'bundle.pkl')
