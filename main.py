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

# --- Step 0: Download macroeconomic data ---
subprocess.run(["python3", "download_todays_data.py"], check=True)

# --- Feature name mappings ---
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
    if day == 0:
        return "on the summit day"
    elif day > 0:
        return f"{day} day{'s' if day > 1 else ''} after the summit"
    else:
        return f"{abs(day)} day{'s' if abs(day) > 1 else ''} before the summit"

def summarize_centroid_readably(centroid_df, top_n=4):
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

# --- Ticker list ---
tickers = ["AAPL", "MSFT"]  # Modify as needed

# --- Dictionary to store results for all tickers ---
bundle_all = {}

for ticker in tickers:
    print(f"\nProcessing ticker: {ticker}")

    # Step 1: Call Peaks
    df_resid, df_merged_peaks, df_summits_filtered = call_peak_events(ticker)

    # Step 2: Extract Sequence Features
    df_features = build_feature_matrix(ticker)
    df_sequences_all, df_scaled_features, df_summits = extract_context_sequences(
        df_features, df_summits_filtered
    )

    # Step 3: Motif Clustering + Similarity Features
    df_enrichment, df_peak_clusters, ctrl_labels, centroid_motifs, fdr_adj = cluster_and_extract_motifs(
        df_sequences_all
    )
    df_model = add_similarity_features(
        df_features=df_features,
        df_scaled_features=df_scaled_features,
        centroid_motifs=centroid_motifs,
        fdr_adj=fdr_adj
    )

    # Step 4: Model Training
    significant_clusters = df_enrichment[df_enrichment['Significant']]['Cluster_ID'].tolist()
    print(df_enrichment[df_enrichment['Significant']])
    results, df_test = train_optuna_xgboost_model(
        df_model=df_model,
        df_summits=df_summits_filtered,
        significant_clusters=significant_clusters,
        n_trials=100
    )

    # Step 5: Threshold Optimization
    y_true = np.array(results['y_test'])
    y_proba = np.array(results['y_proba'])
    best_threshold = results['best_threshold']
    roc_auc = results['roc_auc']

    target_matrix = np.array([
        [0.7, 0.2, 0.1],  # True 0
        [0.1, 0.2, 0.7]   # True 1
    ])
    best_T1, best_T2, best_conf, best_mse = optimize_thresholds(y_true, y_proba, target_matrix, best_threshold)

    # Step 6: SHAP value calculation
    model = results['model']
    X_test = results['X_test']
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    results['shap_values'] = shap_values.values.tolist()
    results['shap_feature_names'] = X_test.columns.tolist()

    # Step 7: Summarize significant motifs
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

    # Save everything related to this ticker
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
        'df_enrichment': df_enrichment,
        'significant_clusters': significant_clusters,
        'significant_cluster_info': significant_info,
        'significant_cluster_summaries': readable_clusters
    }

# Save one combined bundle
joblib.dump(bundle_all, 'bundle2.pkl')
