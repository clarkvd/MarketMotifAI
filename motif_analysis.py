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
    if selected_cols is None:
        selected_cols = [
            'Residual_Mean_5', 'Residual_Momentum_5', 'RSI',
            'MACD_Diff', 'BBW', 'Sentiment_Mean_3',
            'VIX_Mean_5', 'Volume_Mean_5'
        ]

    window_length = 2 * context_window + 1

    def build_flattened_matrix(df_seq: pd.DataFrame, seq_type: str):
        df_t = df_seq[df_seq['Type'] == seq_type]
        summit_dates = np.sort(df_t['Summit_Date'].unique())
        n_windows = len(summit_dates)
        n_features = len(selected_cols)
        X = np.zeros((n_windows, window_length * n_features), dtype=float)

        for idx, summit in enumerate(summit_dates):
            sub = df_t[df_t['Summit_Date'] == summit]
            pivot = sub.pivot_table(index='Days_From_Summit', values=selected_cols, aggfunc='first')
            full_index = list(range(-context_window, context_window + 1))
            pivot = pivot.reindex(full_index, fill_value=0.0)
            flattened = pivot[selected_cols].values.flatten(order='C')
            X[idx, :] = flattened

        return X, summit_dates

    # Step 1: Build matrices
    X_peak, peak_summits = build_flattened_matrix(df_sequences_all, seq_type='Peak')
    X_ctrl, ctrl_summits = build_flattened_matrix(df_sequences_all, seq_type='Control')

    # Step 2: PCA
    n_components = min(10, X_peak.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_peak_pca = pca.fit_transform(X_peak)
    X_ctrl_pca = pca.transform(X_ctrl)

    # Step 3: Determine best cluster count
    results = []
    for n_clusters in range(2, 16):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        peak_labels = kmeans.fit_predict(X_peak_pca)
        ctrl_labels = kmeans.predict(X_ctrl_pca)
    
        peak_counts = np.bincount(peak_labels, minlength=n_clusters)
        ctrl_counts = np.bincount(ctrl_labels, minlength=n_clusters)
    
        enrichment_p = []
        for cluster_id in range(n_clusters):
            a, b = peak_counts[cluster_id], len(peak_labels) - peak_counts[cluster_id]
            c, d = ctrl_counts[cluster_id], len(ctrl_labels) - ctrl_counts[cluster_id]
            table = np.array([[a, b], [c, d]])
            _, p_val = fisher_exact(table, alternative='greater')
            enrichment_p.append(p_val)
    
        _, fdr_adj, _, _ = multipletests(enrichment_p, alpha=0.05, method='fdr_bh')
        significant_clusters = [i for i, p in enumerate(fdr_adj) if p < 0.05]
        num_sig = len(significant_clusters)
        total_peak_in_sig = sum(peak_counts[i] for i in significant_clusters)
    
        results.append({
            'n_clusters': n_clusters,
            'num_significant_clusters': num_sig,
            'total_peak_in_significant': total_peak_in_sig
        })
    
    df_cluster_eval = pd.DataFrame(results)
    
    # Find max number of significant clusters
    max_sig = df_cluster_eval['num_significant_clusters'].max()
    candidates = df_cluster_eval[df_cluster_eval['num_significant_clusters'] == max_sig]
    
    # Choose the one with highest total peak count in significant clusters
    best_row = candidates.sort_values('total_peak_in_significant', ascending=False).iloc[0]
    best_n_clusters = best_row['n_clusters']

    # Step 4: Final clustering
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    peak_labels = kmeans.fit_predict(X_peak_pca)
    ctrl_labels = kmeans.predict(X_ctrl_pca)

    # Step 5: Enrichment stats
    peak_counts = np.bincount(peak_labels, minlength=best_n_clusters)
    ctrl_counts = np.bincount(ctrl_labels, minlength=best_n_clusters)
    n_peak_total = len(peak_labels)
    n_ctrl_total = len(ctrl_labels)

    enrichment_data = []
    for cluster_id in range(best_n_clusters):
        a, b = peak_counts[cluster_id], n_peak_total - peak_counts[cluster_id]
        c, d = ctrl_counts[cluster_id], n_ctrl_total - ctrl_counts[cluster_id]
        table = np.array([[a, b], [c, d]])
        oddsratio, p_value = fisher_exact(table, alternative='greater')
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

    # Step 6: Centroid motifs
    cluster_centroids_flat = pca.inverse_transform(kmeans.cluster_centers_)
    centroid_motifs = {}
    for cluster_id in range(best_n_clusters):
        flat_vec = cluster_centroids_flat[cluster_id]
        motif_matrix = flat_vec.reshape(window_length, len(selected_cols))
        centroid_motifs[cluster_id] = pd.DataFrame(
            motif_matrix,
            index=range(-context_window, context_window + 1),
            columns=selected_cols
        )

    # Optional: match cluster labels to peak dates
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
    Adds similarity-to-motif features to df_features based on contextual windows
    and returns a merged DataFrame (df_model).
    
    Parameters:
        df_features (pd.DataFrame): Original feature DataFrame with 'Date' column.
        df_scaled_features (pd.DataFrame): Time-sorted DataFrame with scaled features.
        centroid_motifs (dict): Dictionary of cluster ID to motif DataFrame (21x8).
        fdr_adj (list): List of FDR-adjusted p-values, indexed by cluster ID.
        context_window (int): Days before/after to include in the context window.
        selected_cols (list): Columns to extract for comparison. Default: preset list.
    
    Returns:
        df_model (pd.DataFrame): df_features merged with similarity features.
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

    # Step B: Build flattened motifs
    flat_motifs = {
        cid: centroid_motifs[cid].loc[-context_window:context_window, selected_cols].values.flatten(order='C')
        for cid in significant_clusters
    }

    # Step C: Compute similarity for each date
    df_scaled_features = df_scaled_features.sort_values('Date').reset_index(drop=True)
    all_dates = df_scaled_features['Date'].tolist()
    sim_data = []

    for current_date in all_dates:
        start = current_date - pd.Timedelta(days=context_window)
        end = current_date + pd.Timedelta(days=context_window)

        window_df = df_scaled_features[
            (df_scaled_features['Date'] >= start) &
            (df_scaled_features['Date'] <= end)
        ].copy()
        window_df['Days_From_Center'] = (window_df['Date'] - current_date).dt.days

        pivot = window_df.pivot_table(index='Days_From_Center', values=selected_cols, aggfunc='first')
        pivot = pivot.reindex(list(range(-context_window, context_window + 1)), fill_value=0.0)
        flat_window = pivot[selected_cols].values.flatten(order='C')

        sim_row = {'Date': current_date}
        for cid, motif_vec in flat_motifs.items():
            sim = 0.0 if np.all(flat_window == 0) else pearsonr(flat_window, motif_vec)[0]
            sim_row[f'Similarity_Motif_{cid}'] = 0.0 if np.isnan(sim) else sim
        sim_data.append(sim_row)

    df_similarity = pd.DataFrame(sim_data)

    # Step D: Merge similarity features into df_features
    df_features['Date'] = pd.to_datetime(df_features['Date']).dt.tz_localize(None).dt.normalize()
    df_similarity['Date'] = pd.to_datetime(df_similarity['Date']).dt.tz_localize(None).dt.normalize()
    df_model = pd.merge(df_features, df_similarity, on='Date', how='inner')

    return df_model
