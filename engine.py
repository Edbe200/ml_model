import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ticker = "AAPL"  # you can change this later (MSFT, TSLA, etc.)

def download_data(ticker, start): # Start must be in 'YYYY-MM-DD' format
    data =  yf.download(ticker, start=start, end=None) # downloads data up to today


    df = data.copy()

    df.columns = df.columns.get_level_values(0)

    # Adding columns to df

    df["return_1d"] = df["Close"].pct_change()

    df["return_5d_mean"] = df["return_1d"].rolling(window=5).mean()

    df["vol_10d"] = df["return_1d"].rolling(window=10).std()

    df["range_hl"] = (df["High"] - df["Low"]) / df["Open"]

    df["vol_change"] = df["Volume"].pct_change()

    # Adding the ML aspect

    df["tomorrow_close"] = df["Close"].shift(-1)

    df["target_up"] = (df["tomorrow_close"] > df["Close"]).astype(int)

    # Drop NaNs caused by pct_change, rolling windows, and shifting
    df = df.dropna()

    return df

# --- Define feature matrix X and target vector y ---

def build_xy(df):

    feature_cols = [
        "return_1d",
        "return_5d_mean",
        "vol_10d",
        "range_hl",
        "vol_change"
    ]

    X = df[feature_cols].values
    y = df["target_up"].values

    return X, y

# --- Time-based train/test split (80% train, 20% test) ---

def time_split(X, y):

    n = len(X)
    train_size = int(n * 0.8)

    X_train = X[:train_size]
    X_test  = X[train_size:]

    y_train = y[:train_size]
    y_test  = y[train_size:]

    return X_train, X_test, y_train, y_test

# Scaling

def scale_data(X_train, X_test, y_train):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Logistic Regression model

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    return log_reg, X_train_scaled, X_test_scaled, scaler

# Predictions

def make_predictions(log_reg, X_test_scaled, threshold=0.5):

    y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

    # custom threshold version

    y_pred = (y_proba >= threshold).astype(int)

    return y_pred, y_proba

# --- Evaluation ---

def evaluate_model(y_test, y_pred):

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # --- Baseline: always predict the most common class in the test set ---

    baseline_class = np.bincount(y_test).argmax()
    baseline_acc = (y_test == baseline_class).mean()

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "baseline_accuracy": baseline_acc,
        "baseline_class": int(baseline_class),
    }

    return metrics

def full_pipeline(ticker="AAPL", start="2010-01-01", threshold=0.5):    
    df = download_data(ticker, start)
    X, y = build_xy(df)
    X_train, X_test, y_train, y_test = time_split(X, y)
    log_reg, X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test, y_train)
    y_pred, y_proba = make_predictions(log_reg, X_test_scaled, threshold)
    metrics = evaluate_model(y_test, y_pred)

    output = {
        "model": log_reg,
        "scaler": scaler,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
    }
    return output



