import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression

def call_peak_events(ticker: str, benchmark: str = "^GSPC", lookback: int = 60, window_size: int = 5):
    # Step 1: Download data
    stock = yf.Ticker(ticker)
    sp500 = yf.Ticker(benchmark)

    df_stock = stock.history(period="max", interval="1d").reset_index()
    df_sp = sp500.history(period="max", interval="1d").reset_index()

    df_stock = df_stock[['Date', 'Open', 'Close']].rename(columns={'Open': 'Open_Stock', 'Close': 'Close_Stock'})
    df_sp = df_sp[['Date', 'Open', 'Close']].rename(columns={'Open': 'Open_SP', 'Close': 'Close_SP'})

    df = pd.merge(df_stock, df_sp, on='Date')

    # Step 2: Calculate absolute returns (volatility proxy)
    df['AbsRet_Stock'] = ((df['Close_Stock'] - df['Open_Stock']) / df['Open_Stock']).abs()
    df['AbsRet_SP']    = ((df['Close_SP']   - df['Open_SP'])   / df['Open_SP']).abs()

    # Step 3: Rolling regression residuals
    residuals = []
    for i in range(lookback, len(df)):
        X = df['AbsRet_SP'].iloc[i - lookback:i].values.reshape(-1, 1)
        y = df['AbsRet_Stock'].iloc[i - lookback:i].values
        model = LinearRegression().fit(X, y)

        x_today = df['AbsRet_SP'].iloc[i]
        pred = model.predict(np.array([[x_today]]))[0]
        actual = df['AbsRet_Stock'].iloc[i]

        residuals.append({'Date': df['Date'].iloc[i], 'Residual': actual - pred})

    df_resid = pd.DataFrame(residuals)

    # Step 4: Sliding Z-test
    z_scores, p_values, dates = [], [], []
    for i in range(lookback, len(df_resid) - window_size + 1):
        window = df_resid['Residual'].iloc[i:i + window_size]
        background = df_resid['Residual'].iloc[i - lookback:i]

        mu = background.mean()
        sigma = background.std()

        window_sum = window.sum()
        expected = mu * window_size
        std = sigma * np.sqrt(window_size)

        z = (window_sum - expected) / std if std > 0 else 0
        p = 1 - norm.cdf(z)

        z_scores.append(z)
        p_values.append(p)
        dates.append(df_resid['Date'].iloc[i])

    df_peaks = pd.DataFrame({'Start_Date': dates, 'Z_Score': z_scores, 'P_Value': p_values})

    # Step 5: FDR correction
    df_peaks = df_peaks.dropna()
    fdr_results = multipletests(df_peaks['P_Value'], alpha=0.1, method='fdr_bh')
    df_peaks['FDR_Adjusted_P'] = fdr_results[1]
    df_peaks['Significant'] = fdr_results[0]
    df_final_peaks = df_peaks[df_peaks['Significant']].reset_index(drop=True)

    # Step 6: Merge overlapping peaks
    df_final_peaks = df_final_peaks.sort_values('Start_Date').reset_index(drop=True)
    merged_peaks = []
    current_start = df_final_peaks.iloc[0]['Start_Date']
    current_end = current_start + pd.Timedelta(days=window_size - 1)

    for i in range(1, len(df_final_peaks)):
        next_start = df_final_peaks.iloc[i]['Start_Date']
        next_end = next_start + pd.Timedelta(days=window_size - 1)

        if next_start <= current_end + pd.Timedelta(days=1):
            current_end = max(current_end, next_end)
        else:
            merged_peaks.append((current_start, current_end))
            current_start = next_start
            current_end = next_end

    merged_peaks.append((current_start, current_end))
    df_merged_peaks = pd.DataFrame(merged_peaks, columns=['Merged_Start', 'Merged_End'])

    # Step 7: Identify summit in each region
    summits = []
    for _, row in df_merged_peaks.iterrows():
        start = row['Merged_Start']
        end = row['Merged_End']
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

    # Step 8: Filter final calls
    df_summits_filtered = df_summits[
        (df_summits['Summit_Residual'] >= 0.02) &
        (df_summits['Peak_Duration_Days'] >= 3)
    ].reset_index(drop=True)

    return df_resid, df_merged_peaks, df_summits_filtered
