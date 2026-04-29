"""
feature_engineer.py - Özellik Mühendisliği Modülü

Kullanıcının tanımladığı matematiksel kurgu:
1. Lag & Rolling özellikleri (ürün hafızası)
2. İklim interaksiyonları (Konya × Antalya gecikme etkisi)
3. Makroekonomi çarpanları (lojistik maliyet, kur baskısı)
4. Arz-talep göstergeleri (mevsim, enflasyon deflasyonu)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def build_weekly_dataset(
    hal_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    usd_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tüm veri kaynaklarını birleştirip haftalık feature matrix oluşturur.
    """
    print("\n🔧 Feature Engineering başlatılıyor...")

    # ── 1. Ham hal verisini haftalık aggregate et ──────────────────────────
    hal_df = hal_df.copy()
    hal_df["tarih"] = pd.to_datetime(hal_df["tarih"])

    if "ort_fiyat" not in hal_df.columns:
        hal_df["ort_fiyat"] = (
            hal_df["en_dusuk"].fillna(hal_df.get("en_yuksek", 0)) +
            hal_df["en_yuksek"].fillna(hal_df.get("en_dusuk", 0))
        ) / 2

    # Hafta periyodu (ISO Pazartesi başlangıçlı)
    hal_df["hafta_baslangic"] = hal_df["tarih"].dt.to_period("W-MON").apply(
        lambda p: p.start_time
    )

    grp_cols = ["hafta_baslangic", "urun_adi", "ithal_mi", "hassasiyet_katsayisi"]
    weekly_hal = (
        hal_df.groupby(grp_cols, observed=True)
        .agg(
            fiyat=("ort_fiyat", "mean"),
            en_dusuk=("en_dusuk", "mean"),
            en_yuksek=("en_yuksek", "mean"),
            mevsim_faktoru=("mevsim_faktoru", "max"),
            gun_sayisi=("ort_fiyat", "count"),
        )
        .reset_index()
        .rename(columns={"hafta_baslangic": "tarih"})
    )
    weekly_hal["tarih"] = pd.to_datetime(weekly_hal["tarih"])

    # ── 2. Dış verileri haftalık aggregate et ─────────────────────────────
    weather_df = weather_df.copy()
    weather_df["tarih"] = pd.to_datetime(weather_df["tarih"])
    weather_df["hafta"] = weather_df["tarih"].dt.to_period("W-MON").apply(
        lambda p: p.start_time
    )
    weekly_weather = (
        weather_df.groupby("hafta")
        .agg(
            konya_ort_sicaklik=("konya_ort_sicaklik", "mean"),
            konya_toplam_yagis=("konya_toplam_yagis", "sum"),
            konya_nem=("konya_nem", "mean"),
            antalya_ort_sicaklik=("antalya_ort_sicaklik", "mean"),
            antalya_don_var=("antalya_don_var", "max"),
            antalya_don_lag1=("antalya_don_lag1", "max"),
            sicaklik_farki=("sicaklik_farki", "mean"),
        )
        .reset_index()
        .rename(columns={"hafta": "tarih"})
    )
    weekly_weather["tarih"] = pd.to_datetime(weekly_weather["tarih"])

    usd_df = usd_df.copy()
    usd_df["tarih"] = pd.to_datetime(usd_df["tarih"])
    usd_df["hafta"] = usd_df["tarih"].dt.to_period("W-MON").apply(
        lambda p: p.start_time
    )
    weekly_usd = (
        usd_df.groupby("hafta")["dolar_kuru"]
        .mean()
        .reset_index()
        .rename(columns={"hafta": "tarih"})
    )
    weekly_usd["tarih"] = pd.to_datetime(weekly_usd["tarih"])

    macro_df = macro_df.copy()
    macro_df["tarih"] = pd.to_datetime(macro_df["tarih"])
    macro_df["hafta"] = macro_df["tarih"].dt.to_period("W-MON").apply(
        lambda p: p.start_time
    )
    weekly_macro = (
        macro_df.groupby("hafta")
        .agg(mazot_fiyati=("mazot_fiyati", "mean"), aylik_tufe=("aylik_tufe", "mean"))
        .reset_index()
        .rename(columns={"hafta": "tarih"})
    )
    weekly_macro["tarih"] = pd.to_datetime(weekly_macro["tarih"])

    # ── 3. Tüm kaynakları birleştir ───────────────────────────────────────
    df = weekly_hal.copy()
    df = df.merge(weekly_weather, on="tarih", how="left")
    df = df.merge(weekly_usd, on="tarih", how="left")
    df = df.merge(weekly_macro, on="tarih", how="left")

    # ── 4. Zaman Özellikleri ──────────────────────────────────────────────
    df["yil"] = df["tarih"].dt.isocalendar().year.astype(int)
    df["hafta_no"] = df["tarih"].dt.isocalendar().week.astype(int)
    df["ay"] = df["tarih"].dt.month
    df["ceyrek"] = df["tarih"].dt.quarter

    # Sinüsoidal mevsimsellik kodlaması (linear hafta_no'dan daha iyi)
    df["hafta_sin"] = np.sin(2 * np.pi * df["hafta_no"] / 52)
    df["hafta_cos"] = np.cos(2 * np.pi * df["hafta_no"] / 52)
    df["ay_sin"] = np.sin(2 * np.pi * df["ay"] / 12)
    df["ay_cos"] = np.cos(2 * np.pi * df["ay"] / 12)

    # ── 5. Ürün Bazlı Lag & Rolling Özellikler ────────────────────────────
    df = df.sort_values(["urun_adi", "tarih"]).reset_index(drop=True)

    print("  ↪ Lag özellikler hesaplanıyor...")
    for urun, grp in df.groupby("urun_adi", observed=True):
        idx = grp.index
        df.loc[idx, "fiyat_lag1"] = grp["fiyat"].shift(1)
        df.loc[idx, "fiyat_lag2"] = grp["fiyat"].shift(2)
        df.loc[idx, "fiyat_lag4"] = grp["fiyat"].shift(4)
        df.loc[idx, "fiyat_lag8"] = grp["fiyat"].shift(8)
        df.loc[idx, "fiyat_lag13"] = grp["fiyat"].shift(13)  # ~3 ay
        # Rolling ortalamalar
        df.loc[idx, "rolling4_ort"] = grp["fiyat"].shift(1).rolling(4, min_periods=2).mean()
        df.loc[idx, "rolling8_ort"] = grp["fiyat"].shift(1).rolling(8, min_periods=4).mean()
        df.loc[idx, "rolling13_ort"] = grp["fiyat"].shift(1).rolling(13, min_periods=6).mean()
        # Momentum: son hafta değişimi (%)
        df.loc[idx, "fiyat_degisim_pct"] = grp["fiyat"].pct_change(1).shift(1) * 100
        # Volatilite: 4 haftalık std
        df.loc[idx, "fiyat_volatilite"] = grp["fiyat"].shift(1).rolling(4, min_periods=2).std()
        # Trend: 4 haftalık fiyat - 13 haftalık fiyat
        df.loc[idx, "fiyat_trend"] = (
            grp["fiyat"].shift(1).rolling(4, min_periods=2).mean() -
            grp["fiyat"].shift(1).rolling(13, min_periods=6).mean()
        )

    # ── 6. Makroekonomi Çarpanları ─────────────────────────────────────────
    print("  ↪ Makroeko özellikleri hesaplanıyor...")

    # Lojistik Maliyet = Mazot × Hassasiyet Katsayısı
    df["lojistik_maliyeti"] = df["mazot_fiyati"] * df["hassasiyet_katsayisi"]

    # Kur Baskısı = Dolar × İthal Durumu (yerli üründe 0 çıkar)
    df["kur_baskisi"] = df["dolar_kuru"] * df["ithal_mi"]

    # Kümülatif TÜFE deflatörü (2024 başı = 1.0 baz)
    # Reel fiyat = Nominal Fiyat / TÜFE_indeksi
    # TÜFE indeksini hesapla
    tufe_pivot = (
        df[["tarih", "aylik_tufe"]].drop_duplicates()
        .sort_values("tarih")
    )
    tufe_pivot["tufe_kumulatif"] = (1 + tufe_pivot["aylik_tufe"] / 100).cumprod()

    # İlk değeri 1.0'a normalleştir
    if len(tufe_pivot) > 0:
        tufe_pivot["tufe_kumulatif"] = tufe_pivot["tufe_kumulatif"] / tufe_pivot["tufe_kumulatif"].iloc[0]

    df = df.merge(tufe_pivot[["tarih", "tufe_kumulatif"]], on="tarih", how="left")
    df["tufe_kumulatif"] = df["tufe_kumulatif"].fillna(1.0)

    # Reel geçmiş fiyat (enflasyona göre düzeltilmiş lag)
    df["reel_fiyat_lag1"] = df["fiyat_lag1"] / df["tufe_kumulatif"]
    df["reel_fiyat_lag4"] = df["fiyat_lag4"] / df["tufe_kumulatif"]

    # ── 7. İklim Interaksiyonları ──────────────────────────────────────────
    print("  ↪ İklim interaksiyonları ekleniyor...")

    # Donun fiyata gecikmeli etkisi: Antalya don → 1 hafta sonra Konya'ya vuruyor
    # Bu zaten scraper'da yapıldı (antalya_don_lag1), haftalık aggregate'de de var

    # Soğukluk skoru: negatif sıcaklık farkının etkisi
    df["soguk_etki"] = np.where(
        df["sicaklik_farki"] < -5,
        abs(df["sicaklik_farki"]) * 0.1,  # soğuk fark = yerli ürün daha mevcut
        0
    )

    # Yağış etkisi: aşırı yağış lojistiği bozar
    df["yagis_etki"] = np.log1p(df["konya_toplam_yagis"].fillna(0))

    # ── 8. Ürün Kodlaması ─────────────────────────────────────────────────
    print("  ↪ Ürün kategorileri ekleniyor...")
    # Label encoding (eğitim için) — model ürün başına ayrı eğitilmeyecek
    urun_map = {u: i for i, u in enumerate(sorted(df["urun_adi"].unique()))}
    df["urun_kodu"] = df["urun_adi"].map(urun_map)

    # ── 9. Hedef Değişken ─────────────────────────────────────────────────
    # Sonraki haftanın fiyatı (1 adım ileri kaydır)
    df = df.sort_values(["urun_adi", "tarih"])
    for urun, grp in df.groupby("urun_adi", observed=True):
        df.loc[grp.index, "sonraki_hafta_fiyat"] = grp["fiyat"].shift(-1)

    # ── 10. Temizlik ───────────────────────────────────────────────────────
    # Sayısal kolonları interpolasyon ile doldur
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(method="ffill").fillna(method="bfill")

    # Hedef değişkeni olmayan son hafta satırlarını çıkar
    df = df.dropna(subset=["sonraki_hafta_fiyat"])

    # Son olarak satırları sırala
    df = df.sort_values(["urun_adi", "tarih"]).reset_index(drop=True)

    print(f"\n✅ Dataset hazır: {df.shape[0]} satır, {df.shape[1]} kolon")
    print(f"   Ürün sayısı: {df['urun_adi'].nunique()}")
    print(f"   Tarih aralığı: {df['tarih'].min().date()} → {df['tarih'].max().date()}")

    # Kaydet
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "weekly_dataset.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"💾 Kaydedildi: {out_path}")

    return df


def get_feature_columns() -> list:
    """Model eğitiminde kullanılacak özellik sütunlarını döner."""
    return [
        # Zaman
        "yil", "hafta_no", "ay", "ceyrek",
        "hafta_sin", "hafta_cos", "ay_sin", "ay_cos",
        # Ürün
        "urun_kodu", "ithal_mi", "hassasiyet_katsayisi", "mevsim_faktoru",
        # Fiyat geçmişi (lag & rolling)
        "fiyat_lag1", "fiyat_lag2", "fiyat_lag4", "fiyat_lag8", "fiyat_lag13",
        "rolling4_ort", "rolling8_ort", "rolling13_ort",
        "fiyat_degisim_pct", "fiyat_volatilite", "fiyat_trend",
        # Reel fiyat
        "reel_fiyat_lag1", "reel_fiyat_lag4",
        # Hava
        "konya_ort_sicaklik", "konya_toplam_yagis", "konya_nem",
        "antalya_ort_sicaklik", "antalya_don_lag1", "sicaklik_farki",
        "soguk_etki", "yagis_etki",
        # Makro
        "mazot_fiyati", "lojistik_maliyeti",
        "dolar_kuru", "kur_baskisi",
        "aylik_tufe", "tufe_kumulatif",
    ]


def get_feature_groups() -> dict:
    """Özellik gruplarını görselleştirme için döner."""
    return {
        "⏰ Zaman": ["yil", "hafta_no", "ay", "hafta_sin", "hafta_cos"],
        "📦 Ürün": ["urun_kodu", "ithal_mi", "hassasiyet_katsayisi", "mevsim_faktoru"],
        "💰 Fiyat Geçmişi": ["fiyat_lag1", "fiyat_lag2", "fiyat_lag4",
                               "rolling4_ort", "rolling8_ort", "fiyat_trend"],
        "☁ Hava": ["konya_ort_sicaklik", "antalya_don_lag1",
                    "sicaklik_farki", "konya_toplam_yagis"],
        "📊 Makroekonomi": ["mazot_fiyati", "lojistik_maliyeti",
                             "dolar_kuru", "kur_baskisi", "tufe_kumulatif"],
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.scraper import scrape_all, _generate_synthetic_data
    from src.data_collector import collect_all_data

    print("Sentetik veri ile test ediliyor...")
    hal_df = _generate_synthetic_data("2024-01-01", "2026-04-26")
    data = collect_all_data()
    df = build_weekly_dataset(hal_df, data["weather"], data["usd_rates"], data["macro"])
    print(df.describe().to_string())
