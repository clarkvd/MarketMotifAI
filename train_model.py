import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def train_optuna_xgboost_model(
    df_model: pd.DataFrame,
    df_summits: pd.DataFrame,
    significant_clusters: list,
    n_trials: int = 150,
    verbose: bool = True
):
    # --- Label target column ---
    df_model['Date'] = pd.to_datetime(df_model['Date']).dt.tz_localize(None).dt.normalize()
    df_summits['Summit_Date'] = pd.to_datetime(df_summits['Summit_Date']).dt.tz_localize(None).dt.normalize()
    summit_dates = set(df_summits['Summit_Date'])

    def is_within_5_days(current_date):
        return any(current_date < summit <= current_date + pd.Timedelta(days=5) for summit in summit_dates)

    df_model['Fragile_Next_5d'] = df_model['Date'].apply(lambda d: int(is_within_5_days(d)))

    # --- Define feature sets ---
    core_features = [
        'Residual_Mean_5', 'Residual_Mean_10', 'Residual_Mean_20',
        'Residual_Std_5', 'Residual_Std_20', 'Residual_Momentum_5',
        'SP500_Vol_20', 'VIX_Mean_5',
        'Volume_Mean_5', 'Volume_Mean_20', 'Volume_Std_20',
        'RSI', 'MACD_Diff', 'BBW',
        'Sentiment_Mean_3', 'Sentiment_Mean_7', 'Sentiment_Std_7',
        'CPIFlag', 'FedFlag'
    ]
    motif_features = [f'Similarity_Motif_{cid}' for cid in significant_clusters]
    all_features = core_features + motif_features

    df_model = df_model.dropna(subset=all_features + ['Fragile_Next_5d']).sort_values('Date').reset_index(drop=True)

    # --- Time-based split ---
    split_idx = int(len(df_model) * 0.8)
    df_train = df_model.iloc[:split_idx]
    df_test = df_model.iloc[split_idx:]

    X_train, y_train = df_train[all_features], df_train['Fragile_Next_5d']
    X_test, y_test = df_test[all_features], df_test['Fragile_Next_5d']

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # --- Optuna tuning ---
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 6.0),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_prob)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    # --- Best model ---
    best_params = study.best_params
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Find optimal threshold using ROC curve (Youden's J) ---
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    youden_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_index]
    y_pred = (y_proba >= best_threshold).astype(int)

    # --- Evaluation ---
    if verbose:
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, digits=4))

        print("\n--- Confusion Matrix ---")
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Time-Based Split)")
        plt.tight_layout()
        plt.show()

        print(f"\nBest ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"Optimal Threshold (Youden’s J): {best_threshold:.4f}")
        print(f"Best Hyperparameters:\n{best_params}")

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_proba': y_proba,
        'y_pred': y_pred,
        'best_threshold': best_threshold,
        'best_params': best_params,
        'roc_auc': roc_auc_score(y_test, y_proba)
    }, df_test
