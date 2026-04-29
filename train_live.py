# -*- coding: utf-8 -*-
"""
train_live.py - Gercek Canli Veri ile Model Egitimi

- Konya Bel'den cekilen 'data/raw/konya_hal_raw.csv' kullanilir.
- Open-Meteo ve TCMB XML kullanilir.
- Sizintisiz haftalik lag'lar uretilir.
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

RAW_PRICES   = os.path.join("data", "raw", "konya_hal_raw.csv")
RAW_WEATHER  = os.path.join("data", "raw", "weather_combined.csv")
RAW_USD      = os.path.join("data", "raw", "usd_rates.csv")
MODELS_DIR   = "models_live"
WEEKLY_CSV   = "weekly_live.csv"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. HAM FİYATLARI HAFTALIK AGGREGATE ETME ─────────────────────────────

def build_weekly_prices():
    if not os.path.exists(RAW_PRICES):
        raise FileNotFoundError(f"[!] Canli veri bulunamadi: {RAW_PRICES}. Lutfen once scraper'i calistirin.")
        
    df = pd.read_csv(RAW_PRICES, encoding="utf-8-sig")
    df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
    df = df.dropna(subset=["tarih", "ort_fiyat"])
    
    # Veri Temizligi: Isimlerdeki gizli karakterleri (\u0307 vs) temizle
    if "urun_adi" in df.columns:
        df["urun_adi"] = df["urun_adi"].str.replace("\u0307", "", regex=False).str.strip().str.lower()

    # ISO Pazartesi haftasi
    df["hafta_baslangic"] = df["tarih"].dt.to_period("W-MON").apply(lambda p: p.start_time)

    # Urun kodlari yoksa uret
    if "urun_kod" not in df.columns:
        df["urun_kod"] = df["urun_adi"].astype("category").cat.codes

    # Haftalik aggregate
    weekly = df.groupby(["hafta_baslangic", "urun_adi", "urun_kod"]).agg(
        fiyat=("ort_fiyat", "mean")
    ).reset_index()
    
    weekly["hafta_baslangic"] = pd.to_datetime(weekly["hafta_baslangic"])
    weekly = weekly.sort_values(["urun_adi", "hafta_baslangic"]).reset_index(drop=True)
    return weekly


# ── 2. HAFTALIK LAG & ROLLING (TERTEMİZ) ─────────────────────────────────────

def add_lag_features(weekly):
    groups = []
    for urun, grp in weekly.groupby("urun_adi"):
        grp = grp.sort_values("hafta_baslangic").copy()
        p = grp["fiyat"]

        # Lag'lar (hafta cinsinden)
        grp["lag_1h"]  = p.shift(1)   
        grp["lag_4h"]  = p.shift(4)   
        grp["lag_8h"]  = p.shift(8)   
        grp["lag_12h"] = p.shift(12)  
        grp["lag_24h"] = p.shift(24)  

        # Rolling (shift(1) ile mevcut haftayi dislar)
        grp["roll4_ort"]  = p.shift(1).rolling(4,  min_periods=2).mean()
        grp["roll8_ort"]  = p.shift(1).rolling(8,  min_periods=4).mean()
        grp["roll12_ort"] = p.shift(1).rolling(12, min_periods=6).mean()
        grp["roll4_std"]  = p.shift(1).rolling(4,  min_periods=2).std()

        # Momentum & trend
        grp["momentum_pct"]  = p.shift(1).pct_change(1) * 100
        grp["momentum_pct"]  = grp["momentum_pct"].replace([np.inf, -np.inf], np.nan)
        grp["trend_4_12"]    = grp["roll4_ort"] - grp["roll12_ort"]
        grp["volatilite"]    = grp["roll4_std"]

        # HEDEF: sonraki haftanin fiyati
        grp["hedef_haftalik"] = p.shift(-1)

        groups.append(grp)

    out = pd.concat(groups).reset_index(drop=True)
    out = out.dropna(subset=["hedef_haftalik", "lag_1h"])
    return out


# ── 3. DIS VERILERI EKLE (HAVA + KUR) ──────────────────────────────────────

def merge_external_data(weekly):
    # Hava
    if os.path.exists(RAW_WEATHER):
        weather = pd.read_csv(RAW_WEATHER)
        weather["tarih"] = pd.to_datetime(weather["tarih"])
        weather["hafta_baslangic"] = weather["tarih"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        
        # Antalya don varligini 1 hafta gecikmeli veriyoruz (yol sureci)
        weather["antalya_don_lag1"] = weather["antalya_don_var"].shift(1).fillna(0)
        weather["sic_fark"] = weather["konya_ort_sicaklik"] - weather["antalya_ort_sicaklik"]
        
        w_agg = weather.groupby("hafta_baslangic").agg(
            konya_sic=("konya_ort_sicaklik", "mean"),
            konya_yagis=("konya_toplam_yagis", "sum"),
            antalya_don_lag1=("antalya_don_lag1", "max"),
            sic_fark=("sic_fark", "mean")
        ).reset_index()
        weekly = weekly.merge(w_agg, on="hafta_baslangic", how="left")

    # Kur
    if os.path.exists(RAW_USD):
        usd = pd.read_csv(RAW_USD)
        usd["tarih"] = pd.to_datetime(usd["tarih"])
        usd["hafta_baslangic"] = usd["tarih"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        u_agg = usd.groupby("hafta_baslangic").agg(dolar_kuru=("dolar_kuru", "mean")).reset_index()
        weekly = weekly.merge(u_agg, on="hafta_baslangic", how="left")
    
    # Zaman ozellikleri
    weekly["yil"]       = weekly["hafta_baslangic"].dt.isocalendar().year.astype(int)
    weekly["hafta_no"]  = weekly["hafta_baslangic"].dt.isocalendar().week.astype(int)
    weekly["ay"]        = weekly["hafta_baslangic"].dt.month
    weekly["hafta_sin"] = np.sin(2 * np.pi * weekly["hafta_no"] / 52)
    weekly["hafta_cos"] = np.cos(2 * np.pi * weekly["hafta_no"] / 52)

    # Yakit logistigi (Makro) - Basit bir simulasyon ya da CSV'den okuma
    # Eger epdk yoksa kur uzerinden proxy
    if "dolar_kuru" in weekly.columns:
        weekly["lojistik"] = weekly["dolar_kuru"] * 2.3  
    else:
        weekly["dolar_kuru"] = 30.0
        weekly["lojistik"] = 70.0

    num_cols = weekly.select_dtypes(include="number").columns
    weekly[num_cols] = weekly[num_cols].fillna(method="ffill").fillna(method="bfill")
    return weekly


# ── 4. FEATURE SECİMİ & EĞİTİM ────────────────────────────────────────────────

CLEAN_FEATURES = [
    "yil", "hafta_no", "ay", "hafta_sin", "hafta_cos", "urun_kod",
    "lag_1h", "lag_4h", "lag_8h", "lag_12h", "lag_24h", 
    "roll4_ort", "roll8_ort", "roll12_ort", "roll4_std", 
    "momentum_pct", "trend_4_12", "volatilite",
    "konya_sic", "konya_yagis", "antalya_don_lag1", "sic_fark",
    "dolar_kuru", "lojistik",
]

TARGET = "hedef_haftalik"


# Global functions removed, logic moved inside train()
def grouped_mda_score(
    df,
    target_col,
    pred_col,
    baseline_col,
    group_col,
    date_col,
    ignore_flat_actual=True,
):
    total_correct = 0
    total_steps = 0

    for _, grp in df.groupby(group_col):
        grp = grp.sort_values(date_col).copy()
        valid = grp[baseline_col].notna() & grp[target_col].notna() & grp[pred_col].notna()
        if valid.sum() == 0:
            continue

        actual_delta = grp.loc[valid, target_col] - grp.loc[valid, baseline_col]
        pred_delta = grp.loc[valid, pred_col] - grp.loc[valid, baseline_col]
        actual_dir = np.sign(actual_delta)
        pred_dir = np.sign(pred_delta)

        if ignore_flat_actual:
            moving_mask = actual_dir != 0
            if moving_mask.sum() == 0:
                continue
            actual_dir = actual_dir[moving_mask]
            pred_dir = pred_dir[moving_mask]

        total_correct += int((actual_dir == pred_dir).sum())
        total_steps += int(len(actual_dir))

    if total_steps == 0:
        return np.nan
    return float(total_correct / total_steps)


def safe_mape(y_true, y_pred, eps=1e-6):
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    mask = np.abs(y_true_arr) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100)


def weighted_mape(y_true, y_pred, eps=1e-6):
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    den = np.sum(np.abs(y_true_arr))
    if den <= eps:
        return np.nan
    num = np.sum(np.abs(y_true_arr - y_pred_arr))
    return float((num / den) * 100)


def build_dynamic_ensemble_prediction(
    test_df,
    pred_cols,
    target_col=TARGET,
    group_col="urun_kod",
    date_col="hafta_baslangic",
    window=12,
    eps=1e-6,
):
    df = test_df.copy()
    ensemble_values = pd.Series(index=df.index, dtype=float)
    weight_rows = []

    # Global fallback agirliklari
    global_inv = []
    for col in pred_cols:
        mape_val = safe_mape(df[target_col], df[col])
        inv = (1.0 / max(mape_val, eps)) if not np.isnan(mape_val) else 1.0
        global_inv.append(inv)
    global_inv = np.array(global_inv, dtype=float)
    global_weights = global_inv / global_inv.sum()

    for g, grp in df.groupby(group_col):
        grp = grp.sort_values(date_col).copy()
        hist = grp.tail(min(window, len(grp)))

        inv = []
        for col in pred_cols:
            mape_val = safe_mape(hist[target_col], hist[col])
            inv.append((1.0 / max(mape_val, eps)) if not np.isnan(mape_val) else 1.0)

        inv = np.array(inv, dtype=float)
        if inv.sum() <= 0:
            weights = global_weights
        else:
            weights = inv / inv.sum()

        grp_pred_matrix = grp[pred_cols].to_numpy(dtype=float)
        grp_ens = grp_pred_matrix.dot(weights)
        ensemble_values.loc[grp.index] = np.maximum(grp_ens, 0)

        w_dict = {"urun_kod": int(g)}
        for col, w in zip(pred_cols, weights):
            w_dict["w_" + col] = round(float(w), 4)
        weight_rows.append(w_dict)

    return ensemble_values.to_numpy(dtype=float), pd.DataFrame(weight_rows)


def train():
    print("\n[*] Canli veriler isleniyor...")
    df = build_weekly_prices()
    df = add_lag_features(df)
    df = merge_external_data(df)
    
    df.to_csv(WEEKLY_CSV, index=False, encoding="utf-8-sig")

    avail = [c for c in CLEAN_FEATURES if c in df.columns]
    
    # Gercek veride de son aylari test yapiyoruz
    train = df[df["hafta_baslangic"] < "2026-01-01"].copy()
    test  = df[df["hafta_baslangic"] >= "2026-01-01"].copy()

    med = train[avail].median(numeric_only=True)
    X_tr = train[avail].fillna(med)
    y_tr = train[TARGET]
    X_te = test[avail].fillna(med)
    y_te = test[TARGET]

    print(f"[*] Train: {len(X_tr)} | Test: {len(X_te)} | Features: {len(avail)}")

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=8, n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, n_jobs=-1, random_state=42, verbose=-1),
        "Stacking": StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1)),
                ("lgbm", lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1))
            ],
            final_estimator=Ridge()
        )
    }

    all_metrics = []
    test_pred_df = test[["hafta_baslangic", "urun_adi", "urun_kod", TARGET]].copy()
    
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        yp = np.maximum(model.predict(X_te), 0)
        pred_col = "pred_" + name.lower().replace(" ", "_")
        test_pred_df[pred_col] = yp
        
        mae  = mean_absolute_error(y_te, yp)
        rmse = np.sqrt(mean_squared_error(y_te, yp))
        # MDA: yonu lag_1h (bir onceki gercek hafta) bazina gore karsilastir.
        # Varsayilan olarak hareket olmayan haftalar (actual_dir==0) dislanir.
        test_df = test.copy()
        test_df["yp"] = yp
        mda_final = grouped_mda_score(
            test_df,
            target_col=TARGET,
            pred_col="yp",
            baseline_col="lag_1h",
            group_col="urun_kod",
            date_col="hafta_baslangic",
            ignore_flat_actual=True,
        )
        
        m_dict = {
            "Model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "wMAPE(%)": round(weighted_mape(y_te, yp), 2),
            "MDA": round(mda_final, 4),
            "R2": round(r2_score(y_te, yp), 4),
        }
        all_metrics.append(m_dict)
        joblib.dump(model, os.path.join(MODELS_DIR, name.lower().replace(" ","_") + ".pkl"))

    # Urun bazli son pencere MAPE ile dinamik agirlikli ensemble
    base_pred_cols = ["pred_random_forest", "pred_lightgbm", "pred_stacking"]
    dyn_pred, weight_df = build_dynamic_ensemble_prediction(
        test_pred_df,
        pred_cols=base_pred_cols,
        target_col=TARGET,
        group_col="urun_kod",
        date_col="hafta_baslangic",
        window=12,
    )
    test_pred_df["pred_dynamic_ensemble"] = dyn_pred

    dyn_mae = mean_absolute_error(y_te, dyn_pred)
    dyn_rmse = np.sqrt(mean_squared_error(y_te, dyn_pred))
    dyn_mda = grouped_mda_score(
        test.assign(yp=dyn_pred),
        target_col=TARGET,
        pred_col="yp",
        baseline_col="lag_1h",
        group_col="urun_kod",
        date_col="hafta_baslangic",
        ignore_flat_actual=True,
    )
    dyn_r2 = r2_score(y_te, dyn_pred)

    all_metrics.append({
        "Model": "Dynamic Ensemble",
        "MAE": round(dyn_mae, 3),
        "RMSE": round(dyn_rmse, 3),
        "wMAPE(%)": round(weighted_mape(y_te, dyn_pred), 2),
        "MDA": round(dyn_mda, 4) if not np.isnan(dyn_mda) else None,
        "R2": round(dyn_r2, 4),
    })

    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(os.path.join(MODELS_DIR, "metrics.csv"), index=False)
    joblib.dump(avail, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(med, os.path.join(MODELS_DIR, "train_median.pkl"))
    weight_df.to_csv(os.path.join(MODELS_DIR, "dynamic_weights_by_product.csv"), index=False)
    test_pred_df.to_csv(os.path.join(MODELS_DIR, "test_predictions_with_dynamic_ensemble.csv"), index=False)

    print("\n" + "=" * 60)
    print("TAMAMLANDI")
    print(mdf.to_string(index=False))

if __name__ == "__main__":
    train()
