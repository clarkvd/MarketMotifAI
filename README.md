# 📈 MarketMotifAI: Translating Genomic Signal Detection to Market Fragility Forecasting

> 🔗 Visit the companion project: [SignalFrame](https://signalframe.link) — genomic signal processing tools for biological and financial time series.

MarketMotifAI applies concepts from **genomics** to **financial time-series**, detecting patterns of risk buried in noisy volatility signals — much like how MACS detects transcription factor binding events in DNA. 

This interdisciplinary project explores how volatility behaves like genomic signal — governed by structure, context, and motifs — and builds a full ML pipeline to uncover, interpret, and predict emerging fragility in financial markets.

---

## 🔬 Conceptual Framework: From Biology to Markets

In **genomics**, we seek hidden signals — transcription factor bindings, expression shifts — within chaotic DNA landscapes. Tools like **MACS** identify "peaks": statistically enriched windows marking regulatory activity.

In **finance**, the story is similar.

- **Financial prices are noisy.**
- **True signals leave behind subtle traces.**
- **Volatility spikes represent regime shifts.**

This project reframes **market volatility as a biological signal** — where residual volatility acts as the genomic enrichment, and structural motifs in financial behavior echo the logic of DNA regulation.

---

## 💡 Why Volatility?

Volatility isn’t just random movement — it encodes **investor psychology**, **macro responses**, and **institutional action**. Much like transcription factors signal regulatory influence, **volatility deviations mark systemic or idiosyncratic pressure.**

We use **residual volatility** as our signal:

- Compute absolute intraday return from open/close prices
- Model expected volatility via 60-day regression on SP500
- Residual = Actual – Predicted volatility

This isolates **stock-specific noise**, removing benchmark effects and highlighting unique stress signals.

---

## 📊 Step 1: Detecting Volatility Peaks with a MACS-Inspired Algorithm

We adapt MACS’s statistical logic to financial residuals. The method:

1. **Sliding window** across daily residuals (5-day rolling sum)
2. Compute: z = (observed sum – expected sum) / (std × √window size)
3. **One-tailed p-value** from normal distribution
4. Apply **Benjamini–Hochberg FDR correction**
5. Merge overlapping windows
6. Define **peak summits** = day with max residual

🔍 **Output**: Statistically significant volatility “peaks” that deviate from history.

---

## 🧠 Step 2: Context Feature Extraction

Volatility doesn’t occur in isolation — it’s a response. We capture market **context** around each peak using:

### ⏳ Context Window
- 21-day window (±10 days around summit)

### 🧮 Engineered Features
- **Technical Indicators**: RSI, MACD, Bollinger Width, volume
- **Volatility Context**: Short/long-term means & stds
- **Macro Proxies**: SP500 volatility, VIX
- **Sentiment**: 3/7-day rolling news sentiment
- **Event Flags**: CPI, FOMC announcements

### 🧪 Control Sampling
- Select matched windows far from peaks
- Extract same features

This lets us compare **fragility context vs. market background.**

---

## 🧬 Step 3: Discovering Behavioral Motifs

Just like biology has DNA motifs (e.g., TATA box), markets exhibit **recurring behavioral structures** around volatility.

### 📐 Pipeline
1. Flatten each 21-day window into vector
2. Apply **PCA** to reduce dimensionality
3. Cluster using **k-means**
4. Identify enriched clusters using **Fisher’s Exact Test + FDR**
5. Visualize each **cluster centroid** as 21×N motif

These become interpretable **market motifs** — defining how indicators evolve before fragility.

---

## 🔁 Step 4: Scanning for Motif Reappearance

For each day:
- Extract 21-day window
- Compute **Pearson correlation** with each motif
- Add similarity scores as new features

🔮 **Motif scores become early-warning signals** — if a known fragility pattern re-emerges, raise attention.

---

## 🤖 Step 5: Predicting Market Fragility

With context + motif features in place, we train a **supervised machine learning classifier** to detect future fragility.

### 🎯 Task
> Predict if a volatility peak will occur in the **next 5 trading days**.

### 📌 Labels
- `1` if fragility occurs in next 5 days
- `0` otherwise

### ⚙️ Model
- **XGBoost** with Optuna hyperparameter tuning
- Optimize for **ROC AUC**
- Time-aware train/test split (80/20)

### 📈 Evaluation
- ROC curve, Confusion matrix
- Precision, recall, F1-score
- Threshold selection via **Youden’s J statistic**

---

## 🌗 Fragility as a Spectrum

We extend binary labels to a 3-class fragility state:

| Probability Range | Label      |
|-------------------|------------|
| `< T1`            | Stable     |
| `T1–T2`           | Uncertain  |
| `≥ T2`            | Fragile    |

Thresholds are selected via grid search to match **desired misclassification behavior** (e.g., fewer false negatives).

### 📊 Normalized Confusion Matrix Example

|                     | Pred: Stable | Pred: Uncertain | Pred: Fragile |
|---------------------|--------------|-----------------|---------------|
| **True: Stable**    | 0.54         | 0.19            | 0.27          |
| **True: Fragile**   | 0.08         | 0.18            | 0.73          |

---

## 🔍 Interpreting Predictions with SHAP

We use **SHAP (SHapley Additive Explanations)** to interpret model decisions.

### 🔧 What SHAP Tells Us
- Which features drive fragility prediction?
- How do motif similarity scores impact output?
- Which days are flagged due to technical/macro sentiment alignment?

📌 **Motif similarity often outranks traditional indicators in SHAP value.**

---

## 🔎 Visualization Examples

### 📈 Residual Volatility Plot

- Yellow = statistically significant peak regions
- Red dot = summit (max residual)
- Blue line = daily residual volatility

### 📉 SHAP Summary Plot

- Y-axis: Features
- X-axis: SHAP value (impact on prediction)
- Color: Feature value (high/low)

---

## 📂 Repository Structure

```bash
MarketMotifAI/
│
├── data/                  # Preprocessed market data
├── models/                # Sample trained models per ticker
├── src/                   # Core source code (feature extraction, clustering, ML)
│   ├── detection/         # Volatility peak finder (MACS-style)
│   ├── feature_engineering/
│   ├── motif_analysis/
│   ├── model_training/
│   └── visualization/
│
├── notebooks/             # Jupyter walkthroughs and demos
├── tests/                 # Unit tests (pytest)
├── config/                # Config YAMLs or param files
├── examples/              # Scripted examples for end-to-end pipeline
│
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py               # Installable package setup
├── .gitignore
└── .env.example           # Template for environment variables
