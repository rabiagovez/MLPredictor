# -*- coding: utf-8 -*-
"""
pipeline_clean.py - Veri Sizintisi Giderilmis Temiz Pipeline

SORUN TESPITI:
  - ortalama == hedef (dogrudan sizintidir)
  - en_dusuk, en_yuksek: ayni gun fiyatlari (sizinti)
  - lag_12/16/..: kayit siralamali lag, ayni hafta ici fiyatlar (sizinti)
  - rolling_*: mevcut gunu iceren rolling (sizinti)

COZUM:
  1. Veriyi haftaya (ISO hafta) aggregate et -> haftalik ort fiyat
  2. HEDEF: bir sonraki haftanin ort fiyati (shift(-1))
  3. LAG'lari haftalik bazda yeniden hesapla (1,4,8,12,24 hafta oncesi)
  4. Disaridan: hava, kur, yakit ekle (Open-Meteo + TCMB XML)
"""

import pandas as pd
import numpy as np
import os, joblib, warnings, requests, time, xml.etree.ElementTree as ET
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
import lightgbm as lgb

RAW_CSV      = "featured_data_v2.csv"
MODELS_DIR   = "models_clean"
WEEKLY_CSV   = "weekly_clean.csv"
PRED_CSV     = "predictions_clean_2026.csv"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── PARAMETRELER ─────────────────────────────────────────────────────────────

# Sizintili -> kesinlikle KULLANILMAYACAK sutunlar
LEAK_COLS = [
    "ortalama", "en_dusuk", "en_yuksek",
    "lag_12", "lag_16", "lag_20", "lag_24", "lag_36",
    "rolling_mean_4", "rolling_std_4", "rolling_min_4", "rolling_max_4",
    "rolling_mean_8", "rolling_std_8", "rolling_min_8", "rolling_max_8",
    "rolling_mean_12","rolling_std_12","rolling_min_12","rolling_max_12",
    "rolling_mean_24","rolling_std_24","rolling_min_24","rolling_max_24",
    "trend_4_8", "gecen_yil_fiyat", "yoy_fark",
    "hedef",   # target, feature degil
    "tarih", "birim", "kategori", "mevsim", "gun",
]

# ── 1. HAM VERİYİ YÜKLEMEKVEHAFTALIK AGGREGATE ─────────────────────────────

def build_weekly_prices():
    """
    featured_data_v2.csv'den haftalik ort fiyat tablosu uretir.
    Sizintili hicbir sutun dahil edilmez.
    """
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    df["tarih_dt"] = pd.to_datetime(df["tarih_dt"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["tarih_dt", "ortalama"])

    # ISO Pazartesi haftasi
    df["hafta_baslangic"] = df["tarih_dt"].dt.to_period("W-MON").apply(
        lambda p: p.start_time
    )

    # Haftalik aggregate: urun + hafta basina tek fiyat
    weekly = (
        df.groupby(["hafta_baslangic", "urun", "urun_kod", "mevsim_kod"])
        .agg(
            fiyat=("ortalama", "mean"),
        )
        .reset_index()
    )
    weekly["hafta_baslangic"] = pd.to_datetime(weekly["hafta_baslangic"])
    weekly = weekly.sort_values(["urun", "hafta_baslangic"]).reset_index(drop=True)

    print("[+] Ham haftalik fiyat: " + str(weekly.shape) +
          " | Urun: " + str(weekly["urun"].nunique()) +
          " | Hafta: " + str(weekly["hafta_baslangic"].nunique()))
    return weekly


# ── 2. HAFTALIK LAG & ROLLING (TERTEMİZ) ─────────────────────────────────────

def add_lag_features(weekly):
    """
    Hafta bazli gecikme ve rolling ozellikleri ekler.
    Tum lag'lar 1+ hafta geriye bakar -> sizinti yok.
    """
    groups = []
    for urun, grp in weekly.groupby("urun"):
        grp = grp.sort_values("hafta_baslangic").copy()
        p = grp["fiyat"]

        # Lag'lar (hafta cinsinden)
        grp["lag_1h"]  = p.shift(1)   # 1 hafta oncesi
        grp["lag_4h"]  = p.shift(4)   # 1 ay oncesi
        grp["lag_8h"]  = p.shift(8)   # 2 ay oncesi
        grp["lag_12h"] = p.shift(12)  # 3 ay oncesi
        grp["lag_24h"] = p.shift(24)  # 6 ay oncesi
        grp["lag_52h"] = p.shift(52)  # 1 yil oncesi

        # Rolling (shift(1) ile mevcut haftayi dislar -> temiz)
        grp["roll4_ort"]  = p.shift(1).rolling(4,  min_periods=2).mean()
        grp["roll8_ort"]  = p.shift(1).rolling(8,  min_periods=4).mean()
        grp["roll12_ort"] = p.shift(1).rolling(12, min_periods=6).mean()
        grp["roll4_std"]  = p.shift(1).rolling(4,  min_periods=2).std()
        grp["roll8_std"]  = p.shift(1).rolling(8,  min_periods=4).std()

        # Momentum & trend
        grp["momentum_pct"]  = p.shift(1).pct_change(1) * 100
        grp["trend_4_12"]    = grp["roll4_ort"] - grp["roll12_ort"]
        grp["volatilite"]    = grp["roll4_std"]

        # HEDEF: sonraki haftanin fiyati (sizintisiz!)
        grp["hedef_haftalik"] = p.shift(-1)

        groups.append(grp)

    out = pd.concat(groups).reset_index(drop=True)
    # Son haftanin hedefi olmaz
    out = out.dropna(subset=["hedef_haftalik", "lag_1h"])
    print("[+] Lag eklendi: " + str(out.shape))
    return out


# ── 3. HAVA DURUMU (Open-Meteo) ───────────────────────────────────────────────

def fetch_weather(city, lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_mean,temperature_2m_min,precipitation_sum",
        "timezone": "Europe/Istanbul",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({
            "tarih": pd.to_datetime(d["time"]),
            city + "_sic":   d["temperature_2m_mean"],
            city + "_min":   d["temperature_2m_min"],
            city + "_yagis": d["precipitation_sum"],
        })
        df[city + "_don"] = (df[city + "_min"] < 0).astype(int)
        print("[+] " + city + " hava verisi: " + str(len(df)) + " gun")
        return df
    except Exception as e:
        print("[!] " + city + " hava hatasi: " + str(e) + " -> sentetik kullaniliyor")
        return None


def build_weekly_weather(start="2023-01-01", end="2026-04-26"):
    konya   = fetch_weather("konya",   37.87, 32.49, start, end)
    time.sleep(0.5)
    antalya = fetch_weather("antalya", 36.90, 30.71, start, end)

    if konya is None:
        dates = pd.date_range(start, end)
        konya = pd.DataFrame({
            "tarih": dates,
            "konya_sic":   [5 + 18*np.sin(2*np.pi*(d.dayofyear/365)) + np.random.normal(0,2) for d in dates],
            "konya_min":   [0 + 15*np.sin(2*np.pi*(d.dayofyear/365)) + np.random.normal(0,3) for d in dates],
            "konya_yagis": [max(0, np.random.exponential(1.5)) if np.random.random()<0.2 else 0 for d in dates],
            "konya_don":   0,
        })
        konya["konya_don"] = (konya["konya_min"] < 0).astype(int)

    if antalya is None:
        dates = pd.date_range(start, end)
        antalya = pd.DataFrame({
            "tarih": dates,
            "antalya_sic":   [14 + 11*np.sin(2*np.pi*(d.dayofyear/365)) + np.random.normal(0,2) for d in dates],
            "antalya_min":   [9  + 9 *np.sin(2*np.pi*(d.dayofyear/365)) + np.random.normal(0,2) for d in dates],
            "antalya_yagis": [max(0, np.random.exponential(2)) if np.random.random()<0.25 else 0 for d in dates],
            "antalya_don":   0,
        })
        antalya["antalya_don"] = (antalya["antalya_min"] < 0).astype(int)

    merged = konya.merge(antalya, on="tarih", how="inner")
    merged["sic_fark"]     = merged["konya_sic"] - merged["antalya_sic"]
    # Antalya don lag1 (1 gun once)
    merged["antalya_don_lag1"] = merged["antalya_don"].shift(1).fillna(0).astype(int)

    # Haftaya aggregate
    merged["hafta_baslangic"] = merged["tarih"].dt.to_period("W-MON").apply(
        lambda p: p.start_time
    )
    weekly_w = merged.groupby("hafta_baslangic").agg(
        konya_sic    =("konya_sic",    "mean"),
        konya_yagis  =("konya_yagis",  "sum"),
        konya_don    =("konya_don",    "max"),
        antalya_sic  =("antalya_sic",  "mean"),
        antalya_don_lag1=("antalya_don_lag1","max"),
        sic_fark     =("sic_fark",     "mean"),
    ).reset_index()
    weekly_w["hafta_baslangic"] = pd.to_datetime(weekly_w["hafta_baslangic"])
    return weekly_w


# ── 4. DOVIZ KURU (TCMB XML) ──────────────────────────────────────────────────

EPDK_MOTORIN = {
    "2023": 29.0, "2024-01": 47.0, "2024-06": 58.0, "2024-12": 65.0,
    "2025-01": 65.5, "2025-06": 69.0, "2025-12": 74.0,
    "2026-01": 75.0, "2026-04": 78.0,
}

def get_monthly_motorin(year, month):
    key = str(year) + "-" + str(month).zfill(2)
    if key in EPDK_MOTORIN:
        return EPDK_MOTORIN[key]
    year_key = str(year)
    if year_key in EPDK_MOTORIN:
        return EPDK_MOTORIN[year_key]
    return 70.0  # fallback

def fetch_tcmb_rate(year, month):
    """TCMB XML'den aylik ortalama USD/TRY."""
    # Ayin ilk is gunu dene
    for day in range(2, 20):
        try:
            url = "https://www.tcmb.gov.tr/kurlar/{y}{m:02d}/{y}{m:02d}{d:02d}.xml".format(
                y=year, m=month, d=day)
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            for cur in root.findall("Currency"):
                if cur.get("CurrencyCode") == "USD":
                    val = cur.findtext("BanknoteBuying") or cur.findtext("ForexBuying")
                    if val:
                        return float(val.replace(",", "."))
        except:
            pass
    return None

def build_weekly_macro(start="2023-01-01", end="2026-04-26"):
    dates = pd.date_range(start, end, freq="D")
    rate_cache = {}
    print("[*] TCMB kur verisi cekiliyor...")
    for y in range(2023, 2027):
        for m in range(1, 13):
            k = (y, m)
            r = fetch_tcmb_rate(y, m)
            if r:
                rate_cache[k] = r
                print("    " + str(y) + "-" + str(m).zfill(2) + ": " + str(round(r,2)) + " TRY/USD")
            time.sleep(0.15)

    # Fallback: lineer interpolasyon 30 TL (2023) -> 42 TL (2026)
    if not rate_cache:
        print("[!] TCMB erisilemedi, tahmini kur kullaniliyor")

    records = []
    prev = 30.0
    for d in dates:
        k = (d.year, d.month)
        rate = rate_cache.get(k, None)
        if rate:
            prev = rate
        else:
            # Interpolasyon
            months_from_start = (d.year - 2023) * 12 + d.month
            rate = 30.0 + months_from_start * (12.0 / 36)
        records.append({
            "tarih": d,
            "dolar_kuru": rate,
            "mazot_fiyati": get_monthly_motorin(d.year, d.month),
        })

    macro = pd.DataFrame(records)
    macro["hafta_baslangic"] = macro["tarih"].dt.to_period("W-MON").apply(
        lambda p: p.start_time
    )
    weekly_m = macro.groupby("hafta_baslangic").agg(
        dolar_kuru  =("dolar_kuru",   "mean"),
        mazot_fiyati=("mazot_fiyati", "mean"),
    ).reset_index()
    weekly_m["hafta_baslangic"] = pd.to_datetime(weekly_m["hafta_baslangic"])
    return weekly_m


# ── 5. TUM VERİYİ BİRLEŞTİR ─────────────────────────────────────────────────

def build_full_dataset(save=True):
    print("\n" + "=" * 60)
    print("[*] Haftalik fiyat tablosu olusturuluyor...")
    prices = build_weekly_prices()
    prices = add_lag_features(prices)

    print("[*] Hava verisi...")
    weather = build_weekly_weather()

    print("[*] Makro veri (kur + yakit)...")
    macro = build_weekly_macro()

    # Birlestir
    df = prices.merge(weather, on="hafta_baslangic", how="left")
    df = df.merge(macro,   on="hafta_baslangic", how="left")

    # Zaman ozellikleri
    df["yil"]       = df["hafta_baslangic"].dt.isocalendar().year.astype(int)
    df["hafta_no"]  = df["hafta_baslangic"].dt.isocalendar().week.astype(int)
    df["ay"]        = df["hafta_baslangic"].dt.month
    df["ceyrek"]    = df["hafta_baslangic"].dt.quarter
    df["hafta_sin"] = np.sin(2 * np.pi * df["hafta_no"] / 52)
    df["hafta_cos"] = np.cos(2 * np.pi * df["hafta_no"] / 52)

    # Makro interaksiyonlar
    df["lojistik"] = df["mazot_fiyati"] * 1.3  # ortalama hassasiyet
    df["kur_x_yabanci"] = df["dolar_kuru"]     # genel kur etkisi

    # NaN doldur (interpolasyon)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(method="ffill").fillna(method="bfill")

    if save:
        df.to_csv(WEEKLY_CSV, index=False, encoding="utf-8-sig")
        print("[+] Kaydedildi: " + WEEKLY_CSV)

    print("[+] Final dataset: " + str(df.shape))
    return df


# ── 6. FEATURE SECİMİ (TEMİZ) ────────────────────────────────────────────────

CLEAN_FEATURES = [
    # Zaman
    "yil", "hafta_no", "ay", "ceyrek", "hafta_sin", "hafta_cos",
    # Urun kimlik
    "urun_kod", "mevsim_kod",
    # Gecmis fiyat - LAG (haftalik, temiz)
    "lag_1h", "lag_4h", "lag_8h", "lag_12h", "lag_24h", "lag_52h",
    # Rolling (mevcut hafta dislanmis)
    "roll4_ort", "roll8_ort", "roll12_ort",
    "roll4_std", "roll8_std",
    # Momentum
    "momentum_pct", "trend_4_12", "volatilite",
    # Hava
    "konya_sic", "konya_yagis", "antalya_don_lag1", "sic_fark",
    # Makro
    "dolar_kuru", "mazot_fiyati", "lojistik",
]

TARGET = "hedef_haftalik"


# ── 7. METRİKLER ─────────────────────────────────────────────────────────────

def mda_score(y_true, y_pred, groups=None):
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if groups is None:
        a = np.sign(np.diff(y_true_arr))
        p = np.sign(np.diff(y_pred_arr))
        return float(np.mean(a == p)) if len(a) > 0 else np.nan

    grp_arr = np.array(groups)
    total_correct = 0
    total_steps = 0

    for g in pd.unique(grp_arr):
        mask = grp_arr == g
        true_g = y_true_arr[mask]
        pred_g = y_pred_arr[mask]
        if len(true_g) < 2:
            continue

        a = np.sign(np.diff(true_g))
        p = np.sign(np.diff(pred_g))
        total_correct += int(np.sum(a == p))
        total_steps += len(a)

    if total_steps == 0:
        return np.nan
    return float(total_correct / total_steps)

def extreme_f1(y_true, y_pred, thr=20):
    y = np.array(y_true)
    mu = np.mean(y)
    if mu == 0: return np.nan
    te = (np.abs(y - mu) / mu * 100 > thr).astype(int)
    pe = (np.abs(np.array(y_pred) - mu) / mu * 100 > thr).astype(int)
    return float(f1_score(te, pe, zero_division=0)) if te.sum() > 0 else np.nan

def metrics(y_true, y_pred, name, groups=None):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    m    = np.array(y_true) != 0
    mape = float(np.mean(np.abs((np.array(y_true)[m]-np.array(y_pred)[m])/np.array(y_true)[m]))*100)
    return {
        "Model": name,
        "MAE":        round(mae, 3),
        "RMSE":       round(rmse, 3),
        "MAPE(%)":    round(mape, 2),
        "MDA":        round(mda_score(y_true, y_pred, groups=groups), 4),
        "F1-Extreme": round(extreme_f1(y_true, y_pred), 4),
        "R2":         round(r2_score(y_true, y_pred), 4),
    }


# ── 8. MODELLER ───────────────────────────────────────────────────────────────

def build_models():
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=8,
        max_features="sqrt", n_jobs=-1, random_state=42
    )
    lgbm = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42
    )
    stack = StackingRegressor(
        estimators=[
            ("rf",   RandomForestRegressor(n_estimators=150, max_depth=8,
                                           min_samples_leaf=8, n_jobs=-1, random_state=42)),
            ("lgbm", lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                        num_leaves=31, verbose=-1, random_state=42)),
            ("ridge", Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=10))])),
            ("svr",   Pipeline([("sc", StandardScaler()), ("s", SVR(C=5, epsilon=1.0))])),
        ],
        final_estimator=ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000),
        cv=5, n_jobs=-1
    )
    return {"Random Forest": rf, "LightGBM": lgbm, "Stacking": stack}


# ── 9. ANA PIPELINE ───────────────────────────────────────────────────────────

def run():
    df = build_full_dataset()

    avail = [c for c in CLEAN_FEATURES if c in df.columns]
    train = df[df["hafta_baslangic"] < "2026-01-01"].copy()
    test  = df[df["hafta_baslangic"] >= "2026-01-01"].copy()

    med = train[avail].median(numeric_only=True)
    X_tr = train[avail].fillna(med)
    y_tr = train[TARGET]
    X_te = test[avail].fillna(med)
    y_te = test[TARGET]

    print("\n[*] Train: " + str(len(X_tr)) + " | Test: " + str(len(X_te)) +
          " | Features: " + str(len(avail)))

    # Korelasyon kontrolu (>=0.99 uyari)
    corr_check = X_tr.corrwith(y_tr).abs()
    high = corr_check[corr_check >= 0.95].sort_values(ascending=False)
    if not high.empty:
        print("\n[!] UYARI - Yuksek korelasyonlu feature'lar:")
        print(high.to_string())

    all_metrics = []
    predictions = {}
    models = build_models()

    for name, model in models.items():
        print("\n[>] " + name + " egitiliyor...")
        model.fit(X_tr, y_tr)
        yp = np.maximum(model.predict(X_te), 0)
        m  = metrics(y_te, yp, name, groups=test["urun"])
        all_metrics.append(m)
        predictions[name] = yp
        joblib.dump(model, os.path.join(MODELS_DIR, name.lower().replace(" ","_") + ".pkl"))
        print("   MAE=" + str(m["MAE"]) + " | RMSE=" + str(m["RMSE"]) +
              " | MAPE=" + str(m["MAPE(%)"]) + "% | MDA=" + str(m["MDA"]) +
              " | F1=" + str(m["F1-Extreme"]) + " | R2=" + str(m["R2"]))

    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(os.path.join(MODELS_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
    joblib.dump(avail, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(med,   os.path.join(MODELS_DIR, "train_median.pkl"))

    pdf = test[["hafta_baslangic","urun","urun_kod", TARGET]].copy()
    for name, yp in predictions.items():
        pdf["tahmin_" + name] = yp
    pdf["tahmin_ortalama"] = np.mean(list(predictions.values()), axis=0)
    pdf.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    fi_rows = []
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            for f, imp in zip(avail, model.feature_importances_):
                fi_rows.append({"Model": name, "Feature": f, "Importance": imp})
    pd.DataFrame(fi_rows).to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"),
                                  index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("TAMAMLANDI")
    print(mdf.to_string(index=False))
    return mdf, pdf


if __name__ == "__main__":
    run()
