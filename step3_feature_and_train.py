# -*- coding: utf-8 -*-
"""
step3_feature_and_train.py
Feature engineering + haftalik/aylik model egitimi
First-Order Differencing: hedef = fiyat_farki (delta) -> MDA > %50
"""
import pandas as pd
import numpy as np
import os, warnings, joblib
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

PYTHON = r"C:\Users\halil\AppData\Local\Python\pythoncore-3.14-64\python.exe"

def load_and_merge():
    """Tum veriyi yukle ve birlestir"""
    print("="*60)
    print("[3] Veri yukleniyor ve birlestiriliyor...")
    
    hal = pd.read_csv("data/raw/konya_hal_raw.csv", encoding="utf-8-sig")
    hal["tarih"] = pd.to_datetime(hal["tarih"])
    
    weather = pd.read_csv("data/raw/weather_combined.csv", encoding="utf-8-sig")
    weather["tarih"] = pd.to_datetime(weather["tarih"])
    
    usd = pd.read_csv("data/raw/usd_rates.csv", encoding="utf-8-sig")
    usd["tarih"] = pd.to_datetime(usd["tarih"])
    
    macro = pd.read_csv("data/raw/macro_data.csv", encoding="utf-8-sig")
    macro["tarih"] = pd.to_datetime(macro["tarih"])
    
    print(f"   Hal: {len(hal)}, Weather: {len(weather)}, USD: {len(usd)}, Macro: {len(macro)}")
    return hal, weather, usd, macro


def build_weekly_features(hal, weather, usd, macro):
    """Haftalik feature matrix olustur"""
    print("\n[3a] Haftalik feature matrix...")
    
    hal["hafta"] = hal["tarih"].dt.isocalendar().week.astype(int)
    hal["yil"] = hal["tarih"].dt.year
    hal["ay"] = hal["tarih"].dt.month
    hal["hafta_key"] = hal["tarih"].dt.to_period("W-MON")
    
    # Urun bazli haftalik ortalama
    weekly = hal.groupby(["urun_adi","hafta_key"]).agg(
        ort_fiyat=("ort_fiyat","mean"),
        en_dusuk=("en_dusuk","mean"),
        en_yuksek=("en_yuksek","mean"),
        ithal_mi=("ithal_mi","first"),
        hassasiyet_katsayisi=("hassasiyet_katsayisi","first"),
        mevsim_faktoru=("mevsim_faktoru","max"),
    ).reset_index()
    
    weekly["hafta_start"] = weekly["hafta_key"].apply(lambda x: x.start_time)
    weekly["yil"] = weekly["hafta_start"].dt.year
    weekly["hafta_no"] = weekly["hafta_start"].dt.isocalendar().week.astype(int)
    weekly["ay"] = weekly["hafta_start"].dt.month
    weekly["ceyrek"] = weekly["hafta_start"].dt.quarter
    
    # Zaman kodlamasi (sin/cos)
    weekly["hafta_sin"] = np.sin(2*np.pi*weekly["hafta_no"]/52)
    weekly["hafta_cos"] = np.cos(2*np.pi*weekly["hafta_no"]/52)
    weekly["ay_sin"] = np.sin(2*np.pi*weekly["ay"]/12)
    weekly["ay_cos"] = np.cos(2*np.pi*weekly["ay"]/12)
    
    # Fiyat spread
    weekly["fiyat_spread"] = weekly["en_yuksek"] - weekly["en_dusuk"]
    
    # Urun label encoding
    le = LabelEncoder()
    weekly["urun_kod"] = le.fit_transform(weekly["urun_adi"])
    
    # Urun bazli lag features (SIZINTISIZ - sadece gecmis)
    weekly = weekly.sort_values(["urun_adi","hafta_start"]).reset_index(drop=True)
    
    for lag in [1, 2, 4, 8, 12]:
        weekly[f"fiyat_lag{lag}"] = weekly.groupby("urun_adi")["ort_fiyat"].shift(lag)
    
    # Rolling stats
    for w in [4, 8, 12]:
        weekly[f"roll{w}_ort"] = weekly.groupby("urun_adi")["ort_fiyat"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        weekly[f"roll{w}_std"] = weekly.groupby("urun_adi")["ort_fiyat"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std())
    
    # Momentum
    weekly["momentum_pct"] = weekly.groupby("urun_adi")["ort_fiyat"].pct_change(1)
    weekly["trend_4_12"] = weekly["roll4_ort"] / weekly["roll12_ort"].replace(0, np.nan)
    
    # ── FIRST-ORDER DIFFERENCING ──
    # Gecmis fiyat farklari (feature olarak)
    weekly["fiyat_diff_1"] = weekly.groupby("urun_adi")["ort_fiyat"].diff(1)
    weekly["fiyat_diff_2"] = weekly.groupby("urun_adi")["ort_fiyat"].diff(2)
    weekly["fiyat_diff_4"] = weekly.groupby("urun_adi")["ort_fiyat"].diff(4)
    
    # Fark bazli lag (gecmisteki farklar)
    for dlag in [1, 2, 3]:
        weekly[f"diff_lag{dlag}"] = weekly.groupby("urun_adi")["fiyat_diff_1"].shift(dlag)
    
    # Fark rolling istatistikleri
    weekly["diff_roll4_ort"] = weekly.groupby("urun_adi")["fiyat_diff_1"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    weekly["diff_roll4_std"] = weekly.groupby("urun_adi")["fiyat_diff_1"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).std())
    weekly["diff_roll8_ort"] = weekly.groupby("urun_adi")["fiyat_diff_1"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=2).mean())
    
    # Yon sinyalleri
    weekly["son_yon"] = np.sign(weekly["fiyat_diff_1"]).fillna(0)
    weekly["son2_yon_toplam"] = weekly.groupby("urun_adi")["son_yon"].transform(
        lambda x: x.shift(1).rolling(2, min_periods=1).sum())
    weekly["son4_yon_toplam"] = weekly.groupby("urun_adi")["son_yon"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).sum())
    
    # Volatilite
    weekly["volatilite_4"] = weekly.groupby("urun_adi")["ort_fiyat"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).std() / x.shift(1).rolling(4, min_periods=2).mean())
    
    # HEDEF: Sonraki haftanin FIYAT FARKI (First-Order Differencing)
    weekly["hedef_abs"] = weekly.groupby("urun_adi")["ort_fiyat"].shift(-1)
    weekly["hedef_haftalik"] = weekly["hedef_abs"] - weekly["ort_fiyat"]
    
    # Hava verisi birlestir (haftalik ortalama)
    weather["hafta_start"] = weather["tarih"].dt.to_period("W-MON").apply(lambda x: x.start_time)
    w_cols = ["konya_ort_sicaklik","konya_toplam_yagis","konya_nem","konya_don_var",
              "antalya_ort_sicaklik","sicaklik_farki"]
    existing_cols = [c for c in w_cols if c in weather.columns]
    weather_w = weather.groupby("hafta_start")[existing_cols].mean().reset_index()
    weekly = weekly.merge(weather_w, on="hafta_start", how="left")
    
    # Kur birlestir
    usd["hafta_start"] = usd["tarih"].dt.to_period("W-MON").apply(lambda x: x.start_time)
    usd_w = usd.groupby("hafta_start")["dolar_kuru"].mean().reset_index()
    weekly = weekly.merge(usd_w, on="hafta_start", how="left")
    
    # Makro birlestir
    macro["hafta_start"] = macro["tarih"].dt.to_period("W-MON").apply(lambda x: x.start_time)
    macro_w = macro.groupby("hafta_start")[["mazot_fiyati","aylik_tufe"]].mean().reset_index()
    weekly = weekly.merge(macro_w, on="hafta_start", how="left")
    
    # Lojistik maliyet
    if "dolar_kuru" in weekly.columns and "mazot_fiyati" in weekly.columns:
        weekly["lojistik"] = weekly["dolar_kuru"] * weekly["mazot_fiyati"] / 1000
    
    # NaN drop (lag'lar yuzunden ilk satirlar)
    weekly = weekly.dropna(subset=["hedef_haftalik","fiyat_lag1"])
    
    print(f"   Haftalik dataset: {len(weekly)} satir, {weekly.shape[1]} sutun")
    return weekly, le


def build_monthly_features(hal, weather, usd, macro):
    """Aylik feature matrix olustur"""
    print("\n[3b] Aylik feature matrix...")
    
    hal["ay_key"] = hal["tarih"].dt.to_period("M")
    
    monthly = hal.groupby(["urun_adi","ay_key"]).agg(
        ort_fiyat=("ort_fiyat","mean"),
        en_dusuk=("en_dusuk","mean"),
        en_yuksek=("en_yuksek","mean"),
        ithal_mi=("ithal_mi","first"),
        hassasiyet_katsayisi=("hassasiyet_katsayisi","first"),
        mevsim_faktoru=("mevsim_faktoru","max"),
    ).reset_index()
    
    monthly["ay_start"] = monthly["ay_key"].apply(lambda x: x.start_time)
    monthly["yil"] = monthly["ay_start"].dt.year
    monthly["ay"] = monthly["ay_start"].dt.month
    monthly["ceyrek"] = monthly["ay_start"].dt.quarter
    monthly["ay_sin"] = np.sin(2*np.pi*monthly["ay"]/12)
    monthly["ay_cos"] = np.cos(2*np.pi*monthly["ay"]/12)
    monthly["fiyat_spread"] = monthly["en_yuksek"] - monthly["en_dusuk"]
    
    le = LabelEncoder()
    monthly["urun_kod"] = le.fit_transform(monthly["urun_adi"])
    
    monthly = monthly.sort_values(["urun_adi","ay_start"]).reset_index(drop=True)
    
    for lag in [1, 2, 3, 6, 12]:
        monthly[f"fiyat_lag{lag}"] = monthly.groupby("urun_adi")["ort_fiyat"].shift(lag)
    
    for w in [3, 6]:
        monthly[f"roll{w}_ort"] = monthly.groupby("urun_adi")["ort_fiyat"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        monthly[f"roll{w}_std"] = monthly.groupby("urun_adi")["ort_fiyat"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std())
    
    monthly["momentum_pct"] = monthly.groupby("urun_adi")["ort_fiyat"].pct_change(1)
    
    # ── FIRST-ORDER DIFFERENCING (Aylik) ──
    monthly["fiyat_diff_1"] = monthly.groupby("urun_adi")["ort_fiyat"].diff(1)
    monthly["fiyat_diff_2"] = monthly.groupby("urun_adi")["ort_fiyat"].diff(2)
    for dlag in [1, 2]:
        monthly[f"diff_lag{dlag}"] = monthly.groupby("urun_adi")["fiyat_diff_1"].shift(dlag)
    monthly["diff_roll3_ort"] = monthly.groupby("urun_adi")["fiyat_diff_1"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).mean())
    monthly["diff_roll3_std"] = monthly.groupby("urun_adi")["fiyat_diff_1"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std())
    monthly["son_yon"] = np.sign(monthly["fiyat_diff_1"]).fillna(0)
    monthly["son3_yon_toplam"] = monthly.groupby("urun_adi")["son_yon"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    monthly["volatilite_3"] = monthly.groupby("urun_adi")["ort_fiyat"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std() / x.shift(1).rolling(3, min_periods=2).mean())
    
    # HEDEF: Sonraki ayin FIYAT FARKI
    monthly["hedef_abs"] = monthly.groupby("urun_adi")["ort_fiyat"].shift(-1)
    monthly["hedef_aylik"] = monthly["hedef_abs"] - monthly["ort_fiyat"]
    
    # Hava (aylik)
    weather["ay_start"] = weather["tarih"].dt.to_period("M").apply(lambda x: x.start_time)
    w_cols = ["konya_ort_sicaklik","konya_toplam_yagis","konya_nem","antalya_ort_sicaklik"]
    existing = [c for c in w_cols if c in weather.columns]
    weather_m = weather.groupby("ay_start")[existing].mean().reset_index()
    monthly = monthly.merge(weather_m, on="ay_start", how="left")
    
    usd["ay_start"] = usd["tarih"].dt.to_period("M").apply(lambda x: x.start_time)
    usd_m = usd.groupby("ay_start")["dolar_kuru"].mean().reset_index()
    monthly = monthly.merge(usd_m, on="ay_start", how="left")
    
    macro["ay_start"] = macro["tarih"].dt.to_period("M").apply(lambda x: x.start_time)
    macro_m = macro.groupby("ay_start")[["mazot_fiyati","aylik_tufe"]].mean().reset_index()
    monthly = monthly.merge(macro_m, on="ay_start", how="left")
    
    if "dolar_kuru" in monthly.columns and "mazot_fiyati" in monthly.columns:
        monthly["lojistik"] = monthly["dolar_kuru"] * monthly["mazot_fiyati"] / 1000
    
    monthly = monthly.dropna(subset=["hedef_aylik","fiyat_lag1"])
    
    print(f"   Aylik dataset: {len(monthly)} satir, {monthly.shape[1]} sutun")
    return monthly, le


def calc_metrics_diff(y_diff_true, y_diff_pred, current_prices):
    """
    First-Order Differencing icin metrik hesaplama.
    MDA: Sadece sifir olmayan gercek degisimler uzerinden hesaplanir.
    """
    # Mutlak fiyata geri donusum
    abs_true = current_prices + y_diff_true
    abs_pred = current_prices + y_diff_pred
    
    # Mutlak fiyat metrikleri
    mae = mean_absolute_error(abs_true, abs_pred)
    rmse = np.sqrt(mean_squared_error(abs_true, abs_pred))
    safe_true = abs_true.replace(0, np.nan)
    mape = np.mean(np.abs((abs_true - abs_pred) / safe_true).dropna()) * 100
    r2 = r2_score(abs_true, abs_pred)
    
    # MDA: Sifir olmayan delta'lar uzerinden yon dogrulugu
    actual_dir = np.sign(np.array(y_diff_true))
    pred_dir = np.sign(np.array(y_diff_pred))
    
    # Sifir delta'lari haric tut (fiyat degismemis = yon yok)
    nonzero_mask = actual_dir != 0
    if nonzero_mask.sum() > 0:
        mda = np.mean(actual_dir[nonzero_mask] == pred_dir[nonzero_mask]) * 100
    else:
        mda = 50.0
    
    return {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mape,2),
            "MDA": round(mda,2), "R2": round(r2,4)}


def train_models(df, target_col, mode="haftalik", time_col="hafta_start"):
    """
    First-Order Differencing ile model egitimi.
    Hedef: fiyat_farki (delta = fiyat_t+1 - fiyat_t)
    MDA direkt olarak tahmin edilen delta'nin isaretinden hesaplanir.
    """
    print(f"\n[4] {mode.upper()} model egitimi (DIFFERENCING)...")
    
    # ZAMANA GORE SIRALA (urun bazli degil, zaman bazli split icin)
    time_c = "hafta_start" if "hafta_start" in df.columns else "ay_start"
    df = df.sort_values([time_c, "urun_adi"]).reset_index(drop=True)
    
    # Feature columns
    exclude = [target_col, "hedef_abs", "urun_adi", "hafta_key", "ay_key", "hafta_start", "ay_start",
               "ort_fiyat", "en_dusuk", "en_yuksek"]
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64","float32","int64","int32","uint32","uint8"]]
    
    # NaN doldur
    for c in feature_cols:
        df[c] = df[c].fillna(df[c].median())
    
    # Inf doldur
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in feature_cols:
        df[c] = df[c].fillna(0)
    
    X = df[feature_cols].values
    y = df[target_col].values            # fiyat FARKI (delta)
    current_prices = df["ort_fiyat"].values  # mevcut fiyat (geri donusum icin)
    
    # Zamana gore %80/%20 split (SIZINTISIZ)
    n = len(df)
    split = int(n * 0.8)
    
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    prices_test = current_prices[split:]
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Feature sayisi: {len(feature_cols)}")
    print(f"   Hedef: fiyat FARKI (delta), ort={y_train.mean():.2f}, std={y_train.std():.2f}")
    
    # Delta istatistikleri
    pos_ratio = (y_train > 0).mean() * 100
    neg_ratio = (y_train < 0).mean() * 100
    zero_ratio = (y_train == 0).mean() * 100
    print(f"   Yon dagilimi: +{pos_ratio:.1f}% / -{neg_ratio:.1f}% / 0={zero_ratio:.1f}%")
    
    # StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    results = {}
    models = {}
    
    # 1) Random Forest
    print("   Random Forest egitiliyor...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=3,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    pred_rf = rf.predict(X_test_s)
    results["Random Forest"] = calc_metrics_diff(
        pd.Series(y_test), pred_rf, pd.Series(prices_test))
    models["rf"] = rf
    
    # 2) LightGBM
    if HAS_LGB:
        print("   LightGBM egitiliyor...")
        lg = lgb.LGBMRegressor(n_estimators=500, max_depth=12, learning_rate=0.03,
                                num_leaves=63, min_child_samples=5,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, verbose=-1, n_jobs=-1)
        lg.fit(X_train_s, y_train)
        pred_lg = lg.predict(X_test_s)
        results["LightGBM"] = calc_metrics_diff(
            pd.Series(y_test), pred_lg, pd.Series(prices_test))
        models["lgbm"] = lg
    
    # 3) Stacking (RF + LGBM + GBR -> ElasticNet)
    print("   Stacking egitiliyor...")
    estimators = [
        ("rf", RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)),
    ]
    if HAS_LGB:
        estimators.append(("lgbm", lgb.LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, verbose=-1, n_jobs=-1)))
    
    stack = StackingRegressor(estimators=estimators,
                              final_estimator=ElasticNet(alpha=0.01, l1_ratio=0.5),
                              n_jobs=-1)
    stack.fit(X_train_s, y_train)
    pred_st = stack.predict(X_test_s)
    results["Stacking"] = calc_metrics_diff(
        pd.Series(y_test), pred_st, pd.Series(prices_test))
    models["stacking"] = stack
    
    # Feature importance
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    # Sonuclari kaydet
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    joblib.dump({"models": models, "scaler": scaler, "features": feature_cols}, f"models/{mode}_models.joblib")
    importances.to_csv(f"data/processed/{mode}_feature_importance.csv", index=False)
    
    # Metrik tablosu
    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "Model"
    metrics_df.to_csv(f"data/processed/{mode}_metrics.csv")
    
    print(f"\n   {mode.upper()} METRIK TABLOSU (First-Order Differencing):")
    print("   " + "-"*75)
    print(f"   {'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'MDA%':>8} {'R2':>8}")
    print("   " + "-"*75)
    for model_name, m in results.items():
        mda_marker = " ***" if m['MDA'] > 50 else ""
        print(f"   {model_name:<20} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['MAPE']:>8.2f} {m['MDA']:>8.2f} {m['R2']:>8.4f}{mda_marker}")
    print("   " + "-"*75)
    print("   *** = MDA > %50 basari")
    
    print(f"\n   TOP-10 Feature Importance ({mode}):")
    for i, row in importances.head(10).iterrows():
        print(f"   {row['feature']:<30} {row['importance']:.4f}")
    
    return results, importances


def run():
    hal, weather, usd, macro = load_and_merge()
    
    # Haftalik
    weekly, le_w = build_weekly_features(hal, weather.copy(), usd.copy(), macro.copy())
    weekly.to_csv("data/processed/weekly_dataset.csv", index=False, encoding="utf-8-sig")
    
    w_results, w_imp = train_models(weekly, "hedef_haftalik", "haftalik", "hafta_start")
    
    # Aylik
    monthly, le_m = build_monthly_features(hal, weather.copy(), usd.copy(), macro.copy())
    monthly.to_csv("data/processed/monthly_dataset.csv", index=False, encoding="utf-8-sig")
    
    m_results, m_imp = train_models(monthly, "hedef_aylik", "aylik", "ay_start")
    
    print("\n" + "="*60)
    print("TAMAMLANDI!")
    print(f"  Haftalik: {len(weekly)} satir")
    print(f"  Aylik:    {len(monthly)} satir")
    print(f"  Modeller: models/ dizinine kaydedildi")
    print(f"  Metrikler: data/processed/ dizinine kaydedildi")


if __name__ == "__main__":
    run()
