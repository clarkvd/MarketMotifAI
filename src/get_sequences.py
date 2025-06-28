import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import ta  # technical analysis indicators

def build_feature_matrix(
    ticker: str,
    sentiment_csv='news_sentiment_data.csv',
    cpi_csv='CPI.csv',
    fomc_csv='FOMC.csv'
) -> pd.DataFrame:
    """
    Builds a machine-learning-ready feature matrix for a given stock ticker by integrating:
        - Price and volatility data (from yfinance)
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Macroeconomic sentiment scores (news sentiment, CPI, FOMC)
        - Engineered features from volatility regression residuals

    Parameters:
        ticker (str): Stock ticker (e.g., 'AAPL')
        sentiment_csv (str): Path to news sentiment CSV file
        cpi_csv (str): Path to CPI release date CSV file
        fomc_csv (str): Path to FOMC release date CSV file

    Returns:
        pd.DataFrame: Feature matrix with normalized and engineered indicators for modeling
    """

    # --- Download price and benchmark data ---
    stock = yf.Ticker(ticker)
    sp500 = yf.Ticker("^GSPC")
    vix = yf.Ticker("^VIX")

    df_stock = stock.history(period="max", interval="1d").reset_index()
    df_sp = sp500.history(period="max", interval="1d").reset_index()
    df_vix = vix.history(period="max", interval="1d").reset_index()

    # --- Clean and rename columns ---
    df_stock = df_stock[['Date', 'Open', 'Close', 'Volume']].rename(columns={'Open': 'Open_Stock', 'Close': 'Close_Stock'})
    df_sp = df_sp[['Date', 'Open', 'Close']].rename(columns={'Open': 'Open_SP', 'Close': 'Close_SP'})
    df_vix = df_vix[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})

    for df_ in [df_stock, df_sp, df_vix]:
        df_['Date'] = pd.to_datetime(df_['Date']).dt.tz_localize(None)

    # --- Merge all market data ---
    df = pd.merge(df_stock, df_sp, on='Date')
    df = pd.merge(df, df_vix, on='Date', how='left')

    # --- Compute absolute intraday returns (volatility proxies) ---
    df['AbsRet_Stock'] = ((df['Close_Stock'] - df['Open_Stock']) / df['Open_Stock']).abs()
    df['AbsRet_SP']    = ((df['Close_SP']   - df['Open_SP'])   / df['Open_SP']).abs()

    # --- Rolling regression residuals over SP500 volatility (60-day window) ---
    lookback = 60
    residuals = []
    for i in range(lookback, len(df)):
        X = df['AbsRet_SP'].iloc[i - lookback:i].values.reshape(-1, 1)
        y = df['AbsRet_Stock'].iloc[i - lookback:i].values
        model = LinearRegression().fit(X, y)
        pred = model.predict([[df['AbsRet_SP'].iloc[i]]])[0]
        actual = df['AbsRet_Stock'].iloc[i]
        residuals.append({'Date': df['Date'].iloc[i], 'Residual': actual - pred})
    df_resid = pd.DataFrame(residuals)
    df = pd.merge(df, df_resid, on='Date', how='left')

    # --- Technical indicators using `ta` package ---
    df['RSI'] = ta.momentum.RSIIndicator(df['Close_Stock'], window=14).rsi()
    macd = ta.trend.MACD(df['Close_Stock'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
    bbw = ta.volatility.BollingerBands(close=df['Close_Stock'], window=20)
    df['BBW'] = (bbw.bollinger_hband() - bbw.bollinger_lband()) / df['Close_Stock']

    # --- News sentiment features ---
    sentiment = pd.read_csv(sentiment_csv)
    sentiment['date'] = pd.to_datetime(sentiment['date']).dt.tz_localize(None).dt.normalize()
    sentiment = sentiment[['date', 'News Sentiment']].rename(columns={'date': 'Date', 'News Sentiment': 'Sentiment'})
    sentiment['Sentiment_Mean_3'] = sentiment['Sentiment'].rolling(3).mean()
    sentiment['Sentiment_Mean_7'] = sentiment['Sentiment'].rolling(7).mean()
    sentiment['Sentiment_Std_7']  = sentiment['Sentiment'].rolling(7).std()

    df['Date'] = df['Date'].dt.normalize()
    df = pd.merge(df, sentiment, on='Date', how='left')

    # --- Macro event flags ---
    cpi_df = pd.read_csv(cpi_csv)
    fomc_df = pd.read_csv(fomc_csv)
    cpi_df['Date'] = pd.to_datetime(cpi_df.iloc[:, 0], errors='coerce').dt.normalize()
    fomc_df['Date'] = pd.to_datetime(fomc_df.iloc[:, 0], errors='coerce').dt.normalize()
    df['CPIFlag'] = df['Date'].isin(cpi_df['Date']).astype(int)
    df['FedFlag'] = df['Date'].isin(fomc_df['Date']).astype(int)

    # --- Feature engineering on residuals, volatility, and volume ---
    df['Residual_Mean_5'] = df['Residual'].rolling(5).mean()
    df['Residual_Mean_10'] = df['Residual'].rolling(10).mean()
    df['Residual_Mean_20'] = df['Residual'].rolling(20).mean()
    df['Residual_Std_5'] = df['Residual'].rolling(5).std()
    df['Residual_Std_20'] = df['Residual'].rolling(20).std()
    df['Residual_Momentum_5'] = df['Residual'] - df['Residual'].shift(5)
    df['SP500_Vol_20'] = df['AbsRet_SP'].rolling(20).std()
    df['VIX_Mean_5'] = df['VIX_Close'].rolling(5).mean()
    df['Volume_Mean_5'] = df['Volume'].rolling(5).mean()
    df['Volume_Mean_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Std_20'] = df['Volume'].rolling(20).std()

    # --- Select finalized feature columns ---
    feature_cols = [
        'Date',
        'Residual_Mean_5', 'Residual_Mean_10', 'Residual_Mean_20',
        'Residual_Std_5', 'Residual_Std_20', 'Residual_Momentum_5',
        'SP500_Vol_20', 'VIX_Mean_5',
        'Volume_Mean_5', 'Volume_Mean_20', 'Volume_Std_20',
        'RSI', 'MACD_Diff', 'BBW',
        'Sentiment_Mean_3', 'Sentiment_Mean_7', 'Sentiment_Std_7',
        'CPIFlag', 'FedFlag'
    ]

    df_features = df[feature_cols].dropna().reset_index(drop=True)
    return df_features

def extract_context_sequences(
    df_features: pd.DataFrame,
    df_summits: pd.DataFrame,
    context_window: int = 10,
    selected_cols: list = None
) -> pd.DataFrame:
    """
    Extracts context windows around peak events and matched control dates from a feature matrix.

    Parameters:
        df_features (pd.DataFrame): Feature matrix with market/macro features by date
        df_summits (pd.DataFrame): Peaks detected from volatility analysis (includes summit date)
        context_window (int): Number of days before and after the summit/control date to extract
        selected_cols (list): List of feature columns to include (default: key indicators)

    Returns:
        tuple:
            - df_sequences_all (pd.DataFrame): All sequences (Peak + Control), with aligned windows
            - df_scaled_features (pd.DataFrame): Normalized feature matrix
            - df_summits (pd.DataFrame): Filtered summits (used in this analysis)
    """
    if selected_cols is None:
        selected_cols = [
            'Residual_Mean_5', 'Residual_Momentum_5', 'RSI', 'MACD_Diff',
            'BBW', 'Sentiment_Mean_3', 'VIX_Mean_5', 'Volume_Mean_5'
        ]

    # --- Normalize selected features ---
    scaler = StandardScaler()
    df_scaled_features = df_features.copy()
    df_scaled_features[selected_cols] = scaler.fit_transform(df_scaled_features[selected_cols])

    # --- Normalize datetime columns ---
    df_scaled_features['Date'] = pd.to_datetime(df_scaled_features['Date']).dt.tz_localize(None)
    df_summits['Summit_Date'] = pd.to_datetime(df_summits['Summit_Date']).dt.tz_localize(None)
    df_summits['Merged_Start'] = pd.to_datetime(df_summits['Merged_Start']).dt.tz_localize(None)
    df_summits['Merged_End'] = pd.to_datetime(df_summits['Merged_End']).dt.tz_localize(None)

    # --- Filter peaks with strong signal and duration ---
    df_summits_filtered = df_summits[
        (df_summits['Summit_Residual'] >= 0.02) &
        (df_summits['Peak_Duration_Days'] >= 3)
    ].reset_index(drop=True)

    # --- Extract ±N-day feature sequences for each peak ---
    sequence_data = []
    for _, row in df_summits_filtered.iterrows():
        summit_date = row['Summit_Date']
        start = summit_date - pd.Timedelta(days=context_window)
        end = summit_date + pd.Timedelta(days=context_window)
        window = df_scaled_features[df_scaled_features['Date'].between(start, end)]

        for _, r in window.iterrows():
            sequence_data.append({
                'Summit_Date': summit_date,
                'Date': r['Date'],
                'Days_From_Summit': (r['Date'] - summit_date).days,
                'Type': 'Peak',
                **{col: r[col] for col in selected_cols}
            })

    df_sequences_peak = pd.DataFrame(sequence_data)

    # --- Sample matched control dates outside ±15d windows of any peak ---
    all_peak_dates = set()
    for _, row in df_summits_filtered.iterrows():
        all_peak_dates.update(pd.date_range(
            row['Merged_Start'] - pd.Timedelta(days=15),
            row['Merged_End'] + pd.Timedelta(days=15)
        ))

    df_control_candidates = df_scaled_features[~df_scaled_features['Date'].isin(all_peak_dates)].copy()
    df_control_candidates = df_control_candidates.sample(n=len(df_summits_filtered), random_state=42)

    # --- Extract ±N-day feature sequences for each control ---
    control_data = []
    for dt in df_control_candidates['Date']:
        start = dt - pd.Timedelta(days=context_window)
        end = dt + pd.Timedelta(days=context_window)
        window = df_scaled_features[df_scaled_features['Date'].between(start, end)]

        for _, r in window.iterrows():
            control_data.append({
                'Summit_Date': dt,
                'Date': r['Date'],
                'Days_From_Summit': (r['Date'] - dt).days,
                'Type': 'Control',
                **{col: r[col] for col in selected_cols}
            })

    df_sequences_control = pd.DataFrame(control_data)

    # --- Combine peak and control sequences ---
    df_sequences_all = pd.concat([df_sequences_peak, df_sequences_control], ignore_index=True)
    df_sequences_all = df_sequences_all.sort_values(['Summit_Date', 'Days_From_Summit']).reset_index(drop=True)

    return df_sequences_all, df_scaled_features, df_summits
