"""
predictor.py - 2026 Haftalık Fiyat Tahmin Motoru

Eğitilmiş modelleri kullanarak belirli bir ürün ve hafta için
tüm modellerin tahminini döner ve ensemble sonuç üretir.
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

MODEL_DISPLAY_NAMES = {
    "random_forest": "🌲 Random Forest",
    "lightgbm": "⚡ LightGBM",
    "stacking": "🏆 Stacking",
}


def get_week_start(date_input) -> datetime:
    """Verilen tarihin ISO haftasının Pazartesi'sini döner."""
    if isinstance(date_input, str):
        dt = datetime.strptime(date_input, "%Y-%m-%d")
    elif isinstance(date_input, datetime):
        dt = date_input
    else:
        dt = pd.Timestamp(date_input).to_pydatetime()
    return dt - timedelta(days=dt.weekday())  # Pazartesi


def load_models() -> dict:
    """Tüm eğitilmiş modelleri yükler."""
    models = {}
    for name in ["random_forest", "lightgbm", "stacking"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            models[name] = None

    feat_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    if os.path.exists(feat_path):
        models["feature_cols"] = joblib.load(feat_path)
    else:
        models["feature_cols"] = []

    return models


def load_dataset() -> pd.DataFrame:
    """İşlenmiş veri setini yükler."""
    path = os.path.join(PROCESSED_DIR, "weekly_dataset.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["tarih"] = pd.to_datetime(df["tarih"])
    return df


def predict_for_week(
    urun_adi: str,
    target_date,
    models: dict,
    df: pd.DataFrame,
) -> dict:
    """
    Belirli bir ürün ve hafta için tüm modellerden tahmin alır.

    Parametreler:
        urun_adi: Tahmin yapılacak ürün adı
        target_date: Hedef haftanın başlangıç tarihi (Pazartesi)
        models: Yüklü model dict'i
        df: Haftalık işlenmiş veri seti

    Dönüş:
        {
            'urun': str,
            'tarih': datetime,
            'tahminler': {'model_adi': float},
            'ortalama_tahmin': float,
            'guven_araligi': (alt, ust),
            'son_gercek_fiyat': float,
            'gecmis_veriler': DataFrame
        }
    """
    week_start = get_week_start(target_date)
    feature_cols = models.get("feature_cols", [])

    # Ürün geçmişini filtrele
    urun_df = df[df["urun_adi"] == urun_adi].sort_values("tarih")

    if urun_df.empty:
        return {"hata": f"'{urun_adi}' ürünü veri setinde bulunamadı."}

    # Tahmin yapılacak satıra en yakın geçmiş haftayı bul
    past_df = urun_df[urun_df["tarih"] < pd.Timestamp(week_start)]

    if past_df.empty:
        # Tüm veriyi kullan (tarih ileri olabilir)
        past_df = urun_df

    # En son satırı al (tahmin için)
    last_row = past_df.iloc[-1].copy()
    son_gercek_fiyat = last_row.get("fiyat", None)

    # Feature vektörü oluştur
    # Zaman özelliklerini güncelle
    last_row["yil"] = week_start.year
    last_row["hafta_no"] = week_start.isocalendar()[1]
    last_row["ay"] = week_start.month
    last_row["ceyrek"] = (week_start.month - 1) // 3 + 1
    last_row["hafta_sin"] = np.sin(2 * np.pi * last_row["hafta_no"] / 52)
    last_row["hafta_cos"] = np.cos(2 * np.pi * last_row["hafta_no"] / 52)
    last_row["ay_sin"] = np.sin(2 * np.pi * last_row["ay"] / 12)
    last_row["ay_cos"] = np.cos(2 * np.pi * last_row["ay"] / 12)

    # Feature vektörünü DataFrame'e dönüştür
    avail_feats = [c for c in feature_cols if c in last_row.index]
    X_pred = pd.DataFrame([last_row[avail_feats]])
    X_pred = X_pred.fillna(X_pred.median(numeric_only=True))

    # Her modelden tahmin al
    tahminler = {}
    for name in ["random_forest", "lightgbm", "stacking"]:
        model = models.get(name)
        if model is not None:
            try:
                pred = float(model.predict(X_pred)[0])
                tahminler[name] = max(0, round(pred, 2))
            except Exception as e:
                tahminler[name] = None
        else:
            tahminler[name] = None

    valid_preds = [v for v in tahminler.values() if v is not None]
    ortalama = round(np.mean(valid_preds), 2) if valid_preds else None
    std = np.std(valid_preds) if len(valid_preds) > 1 else 0
    guven_alt = round(ortalama - 1.96 * std, 2) if ortalama else None
    guven_ust = round(ortalama + 1.96 * std, 2) if ortalama else None

    return {
        "urun": urun_adi,
        "tarih": week_start,
        "tahminler": {MODEL_DISPLAY_NAMES.get(k, k): v for k, v in tahminler.items()},
        "ortalama_tahmin": ortalama,
        "guven_araligi": (guven_alt, guven_ust),
        "son_gercek_fiyat": son_gercek_fiyat,
        "gecmis_veriler": urun_df[["tarih", "fiyat"]].tail(20),
    }


def predict_batch(
    urun_listesi: list,
    start_week: str,
    n_weeks: int,
    models: dict,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Birden fazla ürün ve hafta için toplu tahmin üretir.
    Örneğin: 2026 Ocak ayının 4 haftası için tüm ürünler.
    """
    results = []
    week_start = get_week_start(start_week)

    for week_offset in range(n_weeks):
        current_week = week_start + timedelta(weeks=week_offset)

        for urun in urun_listesi:
            pred = predict_for_week(urun, current_week, models, df)
            if "hata" not in pred:
                results.append({
                    "hafta": pred["tarih"].strftime("%Y-%m-%d"),
                    "urun": pred["urun"],
                    "Random Forest": pred["tahminler"].get("🌲 Random Forest"),
                    "LightGBM": pred["tahminler"].get("⚡ LightGBM"),
                    "Stacking": pred["tahminler"].get("🏆 Stacking"),
                    "Ortalama": pred["ortalama_tahmin"],
                    "Son Gerçek Fiyat": pred["son_gercek_fiyat"],
                })

    return pd.DataFrame(results)


def get_available_products(df: pd.DataFrame) -> list:
    """Veri setindeki ürün listesini döner."""
    if df is None:
        return []
    return sorted(df["urun_adi"].unique().tolist())


if __name__ == "__main__":
    print("Predictor test...")
    models = load_models()
    df = load_dataset()
    if df is not None:
        products = get_available_products(df)
        print(f"Mevcut ürünler: {products[:5]}")
        if products:
            result = predict_for_week(products[0], "2026-01-06", models, df)
            print(f"\nTahmin: {result}")
