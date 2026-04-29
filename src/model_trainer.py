"""
model_trainer.py - 3 Kolektif Algoritma Eğitim ve Karşılaştırma Modülü

1. Bagging Temsilcisi: Random Forest Regressor
2. Boosting Temsilcisi: LightGBM Regressor
3. Zirve Noktası: Heterojen Stacking Regressor (RF + LightGBM + Ridge + SVR → ElasticNet)

Eğitim: 2024-01-01 → 2025-12-31
Test:   2026-01-01 → günümüz
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from src.feature_engineer import get_feature_columns

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

TRAIN_END = "2025-12-31"
TEST_START = "2026-01-01"


def load_dataset() -> pd.DataFrame:
    """İşlenmiş haftalık veri setini yükler."""
    path = os.path.join(PROCESSED_DIR, "weekly_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Veri seti bulunamadı: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["tarih"] = pd.to_datetime(df["tarih"])
    return df


def prepare_splits(df: pd.DataFrame) -> tuple:
    """
    Zamansal train/test bölmesi.
    Veri sızıntısını önlemek için shuffle=False.
    """
    feature_cols = get_feature_columns()
    target_col = "sonraki_hafta_fiyat"

    # Yalnızca mevcut feature sütunlarını al
    available_features = [c for c in feature_cols if c in df.columns]

    train_mask = df["tarih"] <= TRAIN_END
    test_mask = df["tarih"] >= TEST_START

    X_train = df.loc[train_mask, available_features].copy()
    y_train = df.loc[train_mask, target_col].copy()
    X_test = df.loc[test_mask, available_features].copy()
    y_test = df.loc[test_mask, target_col].copy()

    # NaN doldurma (güvenlik)
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test = X_test.fillna(X_train.median(numeric_only=True))

    print(f"\n📊 Veri Bölmesi:")
    print(f"   Eğitim: {train_mask.sum()} satır ({df.loc[train_mask, 'tarih'].min().date()} → {df.loc[train_mask, 'tarih'].max().date()})")
    print(f"   Test:   {test_mask.sum()} satır ({df.loc[test_mask, 'tarih'].min().date()} → {df.loc[test_mask, 'tarih'].max().date()})")
    print(f"   Özellik sayısı: {len(available_features)}")

    return X_train, y_train, X_test, y_test, available_features


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Regresyon metriklerini hesaplar."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (0 değerlerini atla)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    metrics = {
        "model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE(%)": round(mape, 2),
        "R²": round(r2, 4),
    }
    print(f"\n   📈 {model_name}:")
    print(f"      MAE:    {mae:.4f} TL")
    print(f"      RMSE:   {rmse:.4f} TL")
    print(f"      MAPE:   {mape:.2f}%")
    print(f"      R²:     {r2:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Random Forest (Bagging)
# ─────────────────────────────────────────────────────────────────────────────

def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        oob_score=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: LightGBM (Boosting)
# ─────────────────────────────────────────────────────────────────────────────

def build_lightgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: Heterojen Stacking Regressor
# ─────────────────────────────────────────────────────────────────────────────

def build_stacking() -> StackingRegressor:
    """
    Base learners: RF + LightGBM + Ridge + SVR
    Meta learner: ElasticNet (overfitting'e karşı koruma)
    """
    base_estimators = [
        ("rf", RandomForestRegressor(
            n_estimators=150, max_depth=10,
            min_samples_leaf=5, n_jobs=-1, random_state=42
        )),
        ("lgbm", lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05,
            num_leaves=31, verbose=-1, random_state=42
        )),
        ("ridge", Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])),
        ("svr", Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=10, epsilon=0.5, kernel="rbf")),
        ])),
    ]

    meta_learner = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000)

    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANA EĞİTİM FONKSİYONU
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(df: pd.DataFrame = None) -> dict:
    """
    Tüm modelleri eğitir, değerlendirir ve kaydeder.
    Dönüş: {'metrics': DataFrame, 'models': {...}, 'predictions': {...}}
    """
    if df is None:
        df = load_dataset()

    X_train, y_train, X_test, y_test, feature_cols = prepare_splits(df)

    os.makedirs(MODELS_DIR, exist_ok=True)

    all_metrics = []
    all_models = {}
    all_predictions = {}

    model_builders = {
        "Random Forest": build_random_forest,
        "LightGBM": build_lightgbm,
        "Stacking": build_stacking,
    }

    print(f"\n{'='*60}")
    print("🤖 Model Eğitimi Başlıyor")
    print(f"{'='*60}")

    for model_name, builder in model_builders.items():
        print(f"\n▶ {model_name} eğitiliyor...")
        start_time = datetime.now()

        model = builder()

        # Stacking için cross-validation → daha uzun sürer
        model.fit(X_train, y_train)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"   ⏱ Eğitim süresi: {elapsed:.1f}s")

        # Tahminler
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Negatif fiyat tahmini engelle
        y_pred_test = np.maximum(y_pred_test, 0)
        y_pred_train = np.maximum(y_pred_train, 0)

        # Metrikleri hesapla
        metrics = compute_metrics(y_test.values, y_pred_test, model_name)
        metrics["egitim_suresi_s"] = round(elapsed, 1)
        all_metrics.append(metrics)

        # Kaydet
        model_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        print(f"   💾 Model kaydedildi: {model_path}")

        all_models[model_name] = model

        # Tahmin DataFrame'i
        test_df = df[df["tarih"] >= TEST_START].copy()
        if len(test_df) == len(y_pred_test):
            test_df[f"tahmin_{model_name}"] = y_pred_test
            all_predictions[model_name] = test_df[
                ["tarih", "urun_adi", "fiyat", "sonraki_hafta_fiyat", f"tahmin_{model_name}"]
            ].copy()

    # Feature columns kaydet
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(MODELS_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("✅ EĞİTİM TAMAMLANDI")
    print(f"{'='*60}")
    print("\n📊 Model Karşılaştırma:")
    print(metrics_df.to_string(index=False))

    # Sonuçları birleştir
    combined_preds = None
    for name, pred_df in all_predictions.items():
        if combined_preds is None:
            combined_preds = pred_df.rename(columns={f"tahmin_{name}": f"tahmin_{name}"})
        else:
            combined_preds = combined_preds.merge(
                pred_df[["tarih", "urun_adi", f"tahmin_{name}"]],
                on=["tarih", "urun_adi"], how="left"
            )

    if combined_preds is not None:
        preds_path = os.path.join(PROCESSED_DIR, "predictions_2026.csv")
        combined_preds.to_csv(preds_path, index=False, encoding="utf-8-sig")
        print(f"\n💾 2026 Tahminleri: {preds_path}")

    return {
        "metrics": metrics_df,
        "models": all_models,
        "predictions": combined_preds,
        "feature_cols": feature_cols,
    }


def get_feature_importance(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """Model özellik önem skorlarını döner."""
    importance = None

    if hasattr(model, "feature_importances_"):  # RF, LightGBM
        importance = model.feature_importances_
    elif hasattr(model, "final_estimator_"):  # Stacking → RF base learner kullan
        for name, est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                importance = est.feature_importances_
                break

    if importance is None:
        return pd.DataFrame()

    imp_df = pd.DataFrame({
        "ozellik": feature_names[:len(importance)],
        "onem_skoru": importance,
    }).sort_values("onem_skoru", ascending=False)

    return imp_df


def load_trained_models() -> dict:
    """Daha önce kaydedilmiş modelleri yükler."""
    models = {}
    for name in ["random_forest", "lightgbm", "stacking"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)

    feature_cols_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    if os.path.exists(feature_cols_path):
        models["feature_cols"] = joblib.load(feature_cols_path)

    return models


if __name__ == "__main__":
    print("Eğitim başlatılıyor (dataset gerekli)...")
    results = train_all_models()
    print("\n✅ Tamamlandı!")
