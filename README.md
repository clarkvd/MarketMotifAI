# MarketMotif AI

**MarketMotif AI** adapts genomic motif discovery techniques to identify volatility signals in financial markets. Just as transcription factor binding creates subtle, detectable "peaks" in DNA accessibility, volatility events in markets reflect moments of investor decision-making — driven by sentiment, macroeconomic events, and structural risk.

This tool detects localized spikes in volatility (“fragility events”) and learns the common patterns that precede them, using a pipeline that draws inspiration from genomics, statistical learning, and explainable AI.

---

## 🧠 Conceptual Analogy

> **Genomics → Markets**  
> **TF binding → Volatility event**  
> **Motif discovery → Market condition clustering**  
> **Peak calling (MACS) → Fragility detection**  
> **SHAP/DeepLIFT → Feature-based explanation of risk**

---

## 🔁 Workflow Overview

1. **📥 Data Acquisition**
   - News sentiment from the [FRBSF Daily News Sentiment Index](https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/)
   - CPI and FOMC dates from [ALFRED CPI](https://alfred.stlouisfed.org/release/downloaddates?rid=10) and [ALFRED FOMC](https://alfred.stlouisfed.org/release/downloaddates?rid=101)
   - Price/volume data from Yahoo Finance via `yfinance`

2. **📈 Peak Calling**
   - Detect volatility anomalies using residuals from rolling regression of a stock’s intraday movement against the S&P 500
   - Inspired by genomic peak callers like [MACS](https://doi.org/10.1186/gb-2008-9-9-r137)

3. **🔬 Motif Discovery**
   - Contextual windows (±10 days) extracted around volatility peaks
   - PCA + KMeans clustering identifies common patterns (“motifs”) in multi-feature time series

4. **📊 Enrichment Testing**
   - Motif enrichment is assessed using Fisher’s exact test vs. matched control windows

5. **🧠 Modeling**
   - XGBoost models trained to predict fragility within the next 5 days
   - Features include technical indicators, sentiment, volume/volatility metrics, and motif similarity scores

6. **🎯 Threshold Optimization**
   - T1 and T2 thresholds selected to best match a desired 3-class prediction profile (low, moderate, high risk)

7. **🩺 Explainability**
   - SHAP values are computed for per-day interpretability and feature importance tracking

---

## 📁 Directory Structure

├── call_peaks.py
├── get_sequences.py
├── motif_analysis.py
├── train_model.py
├── optimize_thresholds.py
├── download_todays_data.py
├── main_pipeline.py # Master pipeline script
├── bundle.pkl # Full output bundle per ticker
├── xgb_model_<TICKER>.pkl # Individual models
├── LICENSE
├── README.md
└── requirements.txt
