# -*- coding: utf-8 -*-
"""
pipeline.py - Konya Hal Fiyatlari: Egitim + Metrik + Tahmin Pipeline
Gercek veri: featured_data_v2.csv (2023-2026)
Train: 2023-2025 | Test: 2026
"""

import pandas as pd
import numpy as np
import os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
import lightgbm as lgb

DATA_PATH = "featured_data_v2.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "yil", "ay", "ceyrek", "yilin_haftasi", "ay_sin", "ay_cos", "mevsim_kod",
    "lag_12", "lag_16", "lag_20", "lag_24", "lag_36",
    "rolling_mean_4", "rolling_std_4", "rolling_min_4", "rolling_max_4",
    "rolling_mean_8", "rolling_std_8",
    "rolling_mean_12", "rolling_std_12",
    "rolling_mean_24", "rolling_std_24",
    "trend_4_8", "gecen_yil_fiyat", "yoy_fark",
    "sicaklik_max_lag", "sicaklik_min_lag", "sicaklik_ort_lag",
    "yagis_mm_lag", "nem_ort_lag", "don_gunu_lag",
    "urun_kod",
]
TARGET = "hedef"


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["tarih_dt"] = pd.to_datetime(df["tarih_dt"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[TARGET, "tarih_dt"])
    return df


def prepare_splits(df):
    avail = [c for c in FEATURE_COLS if c in df.columns]
    train = df[df["tarih_dt"] < "2026-01-01"].copy()
    test  = df[df["tarih_dt"] >= "2026-01-01"].copy()
    X_train = train[avail].fillna(train[avail].median(numeric_only=True))
    y_train = train[TARGET]
    X_test  = test[avail].fillna(train[avail].median(numeric_only=True))
    y_test  = test[TARGET]
    print("Train: " + str(len(X_train)) + " | Test: " + str(len(X_test)) + " | Features: " + str(len(avail)))
    return X_train, y_train, X_test, y_test, test, avail


def mda_score(y_true, y_pred, groups=None):
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if groups is None:
        actual_dir = np.sign(np.diff(y_true_arr))
        pred_dir = np.sign(np.diff(y_pred_arr))
        if len(actual_dir) == 0:
            return np.nan
        return float(np.mean(actual_dir == pred_dir))

    grp_arr = np.array(groups)
    total_correct = 0
    total_steps = 0

    for g in pd.unique(grp_arr):
        mask = grp_arr == g
        true_g = y_true_arr[mask]
        pred_g = y_pred_arr[mask]
        if len(true_g) < 2:
            continue

        actual_dir = np.sign(np.diff(true_g))
        pred_dir = np.sign(np.diff(pred_g))
        total_correct += int(np.sum(actual_dir == pred_dir))
        total_steps += len(actual_dir)

    if total_steps == 0:
        return np.nan
    return float(total_correct / total_steps)


def extreme_f1(y_true, y_pred, threshold_pct=20):
    y_arr = np.array(y_true)
    mean_price = np.mean(y_arr)
    if mean_price == 0:
        return np.nan
    true_extreme = (np.abs(y_arr - mean_price) / mean_price * 100 > threshold_pct).astype(int)
    pred_extreme = (np.abs(np.array(y_pred) - mean_price) / mean_price * 100 > threshold_pct).astype(int)
    if true_extreme.sum() == 0:
        return np.nan
    return float(f1_score(true_extreme, pred_extreme, zero_division=0))


def compute_all_metrics(y_true, y_pred, name, groups=None):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = np.array(y_true) != 0
    mape = float(np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask])
                                / np.array(y_true)[mask])) * 100)
    mda_val = mda_score(y_true, y_pred, groups=groups)
    f1_val  = extreme_f1(y_true, y_pred)
    r2_val  = r2_score(y_true, y_pred)
    return {
        "Model": name,
        "MAE":       round(mae, 3),
        "RMSE":      round(rmse, 3),
        "MAPE(%)":   round(mape, 2),
        "MDA":       round(mda_val, 4) if not np.isnan(mda_val) else None,
        "F1-Extreme": round(f1_val, 4) if not np.isnan(f1_val) else None,
        "R2":        round(r2_val, 4),
    }


def build_models():
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42
    )
    lgbm = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=8, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42
    )
    stacking = StackingRegressor(
        estimators=[
            ("rf",   RandomForestRegressor(n_estimators=150, max_depth=10,
                                           min_samples_leaf=5, n_jobs=-1, random_state=42)),
            ("lgbm", lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                        num_leaves=31, verbose=-1, random_state=42)),
            ("ridge", Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=10.0))])),
            ("svr",   Pipeline([("sc", StandardScaler()), ("s", SVR(C=10, epsilon=0.5))])),
        ],
        final_estimator=ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000),
        cv=5, n_jobs=-1
    )
    return {"Random Forest": rf, "LightGBM": lgbm, "Stacking": stacking}


def run_pipeline():
    print("=" * 60)
    print("[*] Veri yukleniyor...")
    df = load_data()
    X_train, y_train, X_test, y_test, test_df, feat_cols = prepare_splits(df)

    models = build_models()
    all_metrics = []
    predictions = {}

    for name, model in models.items():
        print("\n[>] " + name + " egitiliyor...")
        model.fit(X_train, y_train)
        y_pred = np.maximum(model.predict(X_test), 0)
        metrics = compute_all_metrics(y_test, y_pred, name, groups=test_df["urun"])
        all_metrics.append(metrics)
        predictions[name] = y_pred
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, safe_name + ".pkl"))
        print("   MAE=" + str(metrics["MAE"]) +
              " | RMSE=" + str(metrics["RMSE"]) +
              " | MAPE=" + str(metrics["MAPE(%)"]) + "%" +
              " | MDA=" + str(metrics["MDA"]) +
              " | F1=" + str(metrics["F1-Extreme"]) +
              " | R2=" + str(metrics["R2"]))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(MODELS_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")

    pred_df = test_df[["tarih_dt", "urun", "kategori", TARGET]].copy()
    for name, preds in predictions.items():
        pred_df["tahmin_" + name] = preds
    pred_df["tahmin_ortalama"] = np.mean(list(predictions.values()), axis=0)
    pred_df.to_csv("predictions_2026.csv", index=False, encoding="utf-8-sig")

    fi_data = []
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            for f, imp in zip(feat_cols, model.feature_importances_):
                fi_data.append({"Model": name, "Feature": f, "Importance": imp})
    fi_df = pd.DataFrame(fi_data)
    fi_df.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False, encoding="utf-8-sig")
    joblib.dump(feat_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    print("\n" + "=" * 60)
    print("TAMAMLANDI")
    print(metrics_df.to_string(index=False))
    return metrics_df, pred_df, fi_df


if __name__ == "__main__":
    run_pipeline()
