import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

def cluster_and_extract_motifs(
    df_sequences_all: pd.DataFrame,
    context_window: int = 10,
    selected_cols: list = None
):
    """
    Performs motif discovery using PCA and KMeans clustering on time-windowed peak sequences.
    Evaluates cluster enrichment via Fisher's exact test to identify statistically significant patterns.

    Parameters:
        df_sequences_all (pd.DataFrame): Combined time-aligned peak and control sequences.
        context_window (int): Number of days before and after summit to include (default = 10).
        selected_cols (list): List of feature columns to use (default: key indicators).

    Returns:
        df_enrichment (pd.DataFrame): Enrichment stats for each cluster including FDR p-values.
        df_peak_clusters (pd.DataFrame): Mapping from peak summit dates to cluster IDs.
        ctrl_labels (np.array): Cluster labels assigned to control sequences.
        centroid_motifs (dict): Dictionary mapping cluster IDs to 2D motif matrices.
        fdr_adj (list): FDR-adjusted p-values for all clusters.
    """
    if selected_cols is None:
        selected_cols = [
            'Residual_Mean_5', 'Residual_Momentum_5', 'RSI',
            'MACD_Diff', 'BBW', 'Sentiment_Mean_3',
            'VIX_Mean_5', 'Volume_Mean_5'
        ]

    window_length = 2 * context_window + 1

    def build_flattened_matrix(df_seq: pd.DataFrame, seq_type: str):
        """Creates flattened (1D) matrix for each sequence window."""
        df_t = df_seq[df_seq['Type'] == seq_type]
        summit_dates = np.sort(df_t['Summit_Date'].unique())
        n_windows = len(summit_dates)
        n_features = len(selected_cols)
        X = np.zeros((n_windows, window_length * n_features), dtype=float)

        for idx, summit in enumerate(summit_dates):
            sub = df_t[df_t['Summit_Date'] == summit]
            pivot = sub.pivot_table(index='Days_From_Summit', values=selected_cols, aggfunc='first')
            pivot = pivot.reindex(list(range(-context_window, context_window + 1)), fill_value=0.0)
            X[idx, :] = pivot[selected_cols].values.flatten(order='C')

        return X, summit_dates

    # Step 1: Flatten windows for peaks and controls
    X_peak, peak_summits = build_flattened_matrix(df_sequences_all, 'Peak')
    X_ctrl, ctrl_summits = build_flattened_matrix(df_sequences_all, 'Control')

    # Step 2: Reduce dimensionality using PCA
    n_components = min(10, X_peak.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_peak_pca = pca.fit_transform(X_peak)
    X_ctrl_pca = pca.transform(X_ctrl)

    # Step 3: Try multiple cluster sizes, test enrichment of peaks vs controls
    results = []
    for n_clusters in range(2, 16):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        peak_labels = kmeans.fit_predict(X_peak_pca)
        ctrl_labels = kmeans.predict(X_ctrl_pca)

        # Count members in each cluster
        peak_counts = np.bincount(peak_labels, minlength=n_clusters)
        ctrl_counts = np.bincount(ctrl_labels, minlength=n_clusters)

        # Fisher’s exact test for enrichment
        enrichment_p = []
        for cluster_id in range(n_clusters):
            a, b = peak_counts[cluster_id], len(peak_labels) - peak_counts[cluster_id]
            c, d = ctrl_counts[cluster_id], len(ctrl_labels) - ctrl_counts[cluster_id]
            _, p_val = fisher_exact([[a, b], [c, d]], alternative='greater')
            enrichment_p.append(p_val)

        # FDR correction
        _, fdr_adj, _, _ = multipletests(enrichment_p, alpha=0.05, method='fdr_bh')
        significant_clusters = [i for i, p in enumerate(fdr_adj) if p < 0.05]

        results.append({
            'n_clusters': n_clusters,
            'num_significant_clusters': len(significant_clusters),
            'total_peak_in_significant': sum(peak_counts[i] for i in significant_clusters)
        })

    df_cluster_eval = pd.DataFrame(results)

    # Step 4: Choose best cluster count (max signal in peaks)
    best_n_clusters = df_cluster_eval.sort_values('total_peak_in_significant', ascending=False).iloc[0]['n_clusters']
    kmeans = KMeans(n_clusters=int(best_n_clusters), random_state=42, n_init=10)
    peak_labels = kmeans.fit_predict(X_peak_pca)
    ctrl_labels = kmeans.predict(X_ctrl_pca)

    # Step 5: Final enrichment test for selected clustering
    peak_counts = np.bincount(peak_labels, minlength=int(best_n_clusters))
    ctrl_counts = np.bincount(ctrl_labels, minlength=int(best_n_clusters))
    enrichment_data = []
    for cluster_id in range(int(best_n_clusters)):
        a, b = peak_counts[cluster_id], len(peak_labels) - peak_counts[cluster_id]
        c, d = ctrl_counts[cluster_id], len(ctrl_labels) - ctrl_counts[cluster_id]
        oddsratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
        enrichment_data.append({
            'Cluster_ID': cluster_id,
            'Peak_Count': a,
            'Control_Count': c,
            'Odds_Ratio': oddsratio,
            'P_Value': p_value
        })

    df_enrichment = pd.DataFrame(enrichment_data)
    _, fdr_adj, _, _ = multipletests(df_enrichment['P_Value'], alpha=0.05, method='fdr_bh')
    df_enrichment['FDR_Adjusted_P'] = fdr_adj
    df_enrichment['Significant'] = df_enrichment['FDR_Adjusted_P'] < 0.05

    # Step 6: Reconstruct centroids back into motif form
    cluster_centroids_flat = pca.inverse_transform(kmeans.cluster_centers_)
    centroid_motifs = {
        cid: pd.DataFrame(
            cluster_centroids_flat[cid].reshape(window_length, len(selected_cols)),
            index=range(-context_window, context_window + 1),
            columns=selected_cols
        )
        for cid in range(int(best_n_clusters))
    }

    df_peak_clusters = pd.DataFrame({
        'Summit_Date': peak_summits,
        'Cluster_ID': peak_labels
    })

    return df_enrichment, df_peak_clusters, ctrl_labels, centroid_motifs, fdr_adj

def add_similarity_features(
    df_features: pd.DataFrame,
    df_scaled_features: pd.DataFrame,
    centroid_motifs: dict,
    fdr_adj: list,
    context_window: int = 10,
    selected_cols: list = None
) -> pd.DataFrame:
    """
    Calculates similarity between each date's contextual window and enriched cluster motifs.
    Merges similarity scores back into df_features for downstream modeling.

    Parameters:
        df_features (pd.DataFrame): Original feature table with 'Date' column.
        df_scaled_features (pd.DataFrame): Z-score normalized feature table by date.
        centroid_motifs (dict): Dictionary of cluster_id → motif matrix (21xN).
        fdr_adj (list): FDR-corrected p-values from motif enrichment analysis.
        context_window (int): Number of days before/after date to define the sequence.
        selected_cols (list): Which features to use in similarity comparison.

    Returns:
        df_model (pd.DataFrame): Merged DataFrame with added similarity columns (e.g. Similarity_Motif_2).
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr

    if selected_cols is None:
        selected_cols = [
            'Residual_Mean_5', 'Residual_Momentum_5', 'RSI',
            'MACD_Diff', 'BBW', 'Sentiment_Mean_3',
            'VIX_Mean_5', 'Volume_Mean_5'
        ]

    window_length = 2 * context_window + 1

    # Step A: Identify significant clusters
    significant_clusters = [cid for cid in centroid_motifs if fdr_adj[cid] < 0.05]

    # Step B: Flatten each centroid motif into 1D vectors
    flat_motifs = {
        cid: centroid_motifs[cid].loc[-context_window:context_window, selected_cols].values.flatten(order='C')
        for cid in significant_clusters
    }

    # Step C: Compute Pearson correlation between daily context and each motif
    sim_data = []
    df_scaled_features = df_scaled_features.sort_values('Date').reset_index(drop=True)
    for current_date in df_scaled_features['Date']:
        start = current_date - pd.Timedelta(days=context_window)
        end = current_date + pd.Timedelta(days=context_window)

        window_df = df_scaled_features[
            (df_scaled_features['Date'] >= start) &
            (df_scaled_features['Date'] <= end)
        ].copy()
        window_df['Days_From_Center'] = (window_df['Date'] - current_date).dt.days

        pivot = window_df.pivot_table(index='Days_From_Center', values=selected_cols, aggfunc='first')
        pivot = pivot.reindex(range(-context_window, context_window + 1), fill_value=0.0)
        flat_window = pivot[selected_cols].values.flatten(order='C')

        sim_row = {'Date': current_date}
        for cid, motif_vec in flat_motifs.items():
            sim = 0.0 if np.all(flat_window == 0) else pearsonr(flat_window, motif_vec)[0]
            sim_row[f'Similarity_Motif_{cid}'] = 0.0 if np.isnan(sim) else sim
        sim_data.append(sim_row)

    df_similarity = pd.DataFrame(sim_data)

    # Step D: Merge similarity scores back to main feature table
    df_features['Date'] = pd.to_datetime(df_features['Date']).dt.tz_localize(None).dt.normalize()
    df_similarity['Date'] = pd.to_datetime(df_similarity['Date']).dt.tz_localize(None).dt.normalize()
    df_model = pd.merge(df_features, df_similarity, on='Date', how='inner')

    return df_model
