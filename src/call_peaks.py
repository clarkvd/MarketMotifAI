import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression

def call_peak_events(ticker: str, benchmark: str = "^GSPC", lookback: int = 60, window_size: int = 5):
    """
    Detects statistically significant volatility spikes ("peaks") in a stock's price action 
    relative to a benchmark index using rolling regression residuals and Z-score testing.

    Parameters:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        benchmark (str): The benchmark index ticker (default: '^GSPC' for S&P 500).
        lookback (int): The number of days to use for background volatility modeling.
        window_size (int): The window size (in days) for testing local volatility events.

    Returns:
        df_resid (pd.DataFrame): DataFrame with dates and residuals from rolling regression.
        df_merged_peaks (pd.DataFrame): Detected and merged peak regions (start and end dates).
        df_summits_filtered (pd.DataFrame): Filtered volatility peaks with high signal strength.
    """
    
    # Step 1: Download historical OHLC data for the stock and benchmark
    stock = yf.Ticker(ticker)
    sp500 = yf.Ticker(benchmark)

    df_stock = stock.history(period="max", interval="1d").reset_index()
    df_sp = sp500.history(period="max", interval="1d").reset_index()

    # Keep only Date, Open, Close; rename for clarity
    df_stock = df_stock[['Date', 'Open', 'Close']].rename(columns={'Open': 'Open_Stock', 'Close': 'Close_Stock'})
    df_sp = df_sp[['Date', 'Open', 'Close']].rename(columns={'Open': 'Open_SP', 'Close': 'Close_SP'})

    # Merge stock and benchmark data on date
    df = pd.merge(df_stock, df_sp, on='Date')

    # Step 2: Calculate absolute returns (intraday volatility proxies)
    df['AbsRet_Stock'] = ((df['Close_Stock'] - df['Open_Stock']) / df['Open_Stock']).abs()
    df['AbsRet_SP']    = ((df['Close_SP']   - df['Open_SP'])   / df['Open_SP']).abs()

    # Step 3: Run rolling linear regression to estimate expected stock volatility from SP500
    residuals = []
    for i in range(lookback, len(df)):
        # Use last `lookback` days to train the model
        X = df['AbsRet_SP'].iloc[i - lookback:i].values.reshape(-1, 1)
        y = df['AbsRet_Stock'].iloc[i - lookback:i].values
        model = LinearRegression().fit(X, y)

        # Predict today’s expected volatility
        x_today = df['AbsRet_SP'].iloc[i]
        pred = model.predict(np.array([[x_today]]))[0]
        actual = df['AbsRet_Stock'].iloc[i]

        residuals.append({'Date': df['Date'].iloc[i], 'Residual': actual - pred})

    df_resid = pd.DataFrame(residuals)

    # Step 4: Scan for local peak events using sliding Z-tests over residuals
    z_scores, p_values, dates = [], [], []
    for i in range(lookback, len(df_resid) - window_size + 1):
        # Define test window and background
        window = df_resid['Residual'].iloc[i:i + window_size]
        background = df_resid['Residual'].iloc[i - lookback:i]

        mu = background.mean()
        sigma = background.std()

        # Calculate Z-score for sum of residuals in the window
        window_sum = window.sum()
        expected = mu * window_size
        std = sigma * np.sqrt(window_size)

        z = (window_sum - expected) / std if std > 0 else 0
        p = 1 - norm.cdf(z)  # one-sided test (looking for high residuals)

        z_scores.append(z)
        p_values.append(p)
        dates.append(df_resid['Date'].iloc[i])

    df_peaks = pd.DataFrame({'Start_Date': dates, 'Z_Score': z_scores, 'P_Value': p_values})

    # Step 5: Apply FDR correction for multiple hypothesis testing
    df_peaks = df_peaks.dropna()
    fdr_results = multipletests(df_peaks['P_Value'], alpha=0.1, method='fdr_bh')
    df_peaks['FDR_Adjusted_P'] = fdr_results[1]
    df_peaks['Significant'] = fdr_results[0]

    # Retain only significant peaks
    df_final_peaks = df_peaks[df_peaks['Significant']].reset_index(drop=True)

    # Step 6: Merge overlapping or adjacent significant windows into continuous peak regions
    df_final_peaks = df_final_peaks.sort_values('Start_Date').reset_index(drop=True)
    merged_peaks = []

    current_start = df_final_peaks.iloc[0]['Start_Date']
    current_end = current_start + pd.Timedelta(days=window_size - 1)

    for i in range(1, len(df_final_peaks)):
        next_start = df_final_peaks.iloc[i]['Start_Date']
        next_end = next_start + pd.Timedelta(days=window_size - 1)

        # If next peak overlaps or is adjacent, extend current region
        if next_start <= current_end + pd.Timedelta(days=1):
            current_end = max(current_end, next_end)
        else:
            merged_peaks.append((current_start, current_end))
            current_start = next_start
            current_end = next_end

    # Append the last peak region
    merged_peaks.append((current_start, current_end))
    df_merged_peaks = pd.DataFrame(merged_peaks, columns=['Merged_Start', 'Merged_End'])

    # Step 7: Find summit (max residual) within each merged region
    summits = []
    for _, row in df_merged_peaks.iterrows():
        start, end = row['Merged_Start'], row['Merged_End']
        window = df_resid[(df_resid['Date'] >= start) & (df_resid['Date'] <= end)]

        if window.empty:
            continue

        summit_row = window.loc[window['Residual'].idxmax()]
        summits.append({
            'Merged_Start': start,
            'Merged_End': end,
            'Summit_Date': summit_row['Date'],
            'Summit_Residual': summit_row['Residual'],
            'Peak_Duration_Days': (end - start).days + 1
        })

    df_summits = pd.DataFrame(summits)

    # Step 8: Filter final peaks by strength and duration
    df_summits_filtered = df_summits[
        (df_summits['Summit_Residual'] >= 0.02) &  # Only keep strong residuals
        (df_summits['Peak_Duration_Days'] >= 3)    # Only keep peaks lasting at least 3 days
    ].reset_index(drop=True)

    return df_resid, df_merged_peaks, df_summits_filtered
