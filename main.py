# -*- coding: utf-8 -*-
"""
Konya Hal Fiyatlari AI Tahmin Sistemi - Dashboard
Haftalik + Aylik Tahmin, Metrik Tablolari, Feature Importance
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, joblib

st.set_page_config(page_title="Konya Hal Fiyat Tahmini", layout="wide", page_icon="🥬")

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
    padding: 20px; text-align: center; backdrop-filter: blur(10px);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #00d4ff; }
.metric-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-top: 4px; }
.model-badge-rf { background: linear-gradient(135deg, #00b4d8, #0077b6); padding: 4px 12px; border-radius: 20px; color: white; font-size: 0.75rem; font-weight: 600; }
.model-badge-lgbm { background: linear-gradient(135deg, #06d6a0, #1b9aaa); padding: 4px 12px; border-radius: 20px; color: white; font-size: 0.75rem; font-weight: 600; }
.model-badge-stack { background: linear-gradient(135deg, #e63946, #d62828); padding: 4px 12px; border-radius: 20px; color: white; font-size: 0.75rem; font-weight: 600; }
h1 { background: linear-gradient(90deg, #00d4ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("# 🥬 Konya Hal Fiyat Tahmin Sistemi")
st.markdown("*2022-2026 | Haftalık & Aylık AI Tahmin Motoru*")
st.markdown("---")

# ── Data Load ──
@st.cache_data
def load_data():
    data = {}
    files = {
        "hal": "data/raw/konya_hal_raw.csv",
        "weekly_metrics": "data/processed/haftalik_metrics.csv",
        "monthly_metrics": "data/processed/aylik_metrics.csv",
        "weekly_fi": "data/processed/haftalik_feature_importance.csv",
        "monthly_fi": "data/processed/aylik_feature_importance.csv",
        "weekly_ds": "data/processed/weekly_dataset.csv",
        "monthly_ds": "data/processed/monthly_dataset.csv",
    }
    for key, path in files.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path, encoding="utf-8-sig",
                                     index_col=0 if "metrics" in key else None)
    return data

data = load_data()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Ayarlar")
    
    if "hal" in data:
        hal = data["hal"]
        hal["tarih"] = pd.to_datetime(hal["tarih"])
        urunler = sorted(hal["urun_adi"].unique())
        secili_urun = st.selectbox("🥕 Ürün Seçin", urunler, index=urunler.index("domates (muhtelif)") if "domates (muhtelif)" in urunler else 0)
    else:
        secili_urun = None
        st.warning("Veri bulunamadı!")
    
    st.markdown("---")
    st.markdown("### 📊 Proje Bilgileri")
    if "hal" in data:
        st.markdown(f"- **Toplam Kayıt:** {len(hal):,}")
        st.markdown(f"- **Ürün Sayısı:** {hal['urun_adi'].nunique()}")
        st.markdown(f"- **Tarih Aralığı:** {hal['tarih'].min().strftime('%Y-%m-%d')} → {hal['tarih'].max().strftime('%Y-%m-%d')}")
    st.markdown(f"- **Model:** RF + LightGBM + Stacking")
    st.markdown(f"- **Split:** %80 Train / %20 Test")

# ── Tab Layout ──
tab1, tab2, tab3, tab4 = st.tabs(["📈 Fiyat Analizi", "🔬 Haftalık Model", "📅 Aylık Model", "🧬 Feature Importance"])

# ══════════ TAB 1: Fiyat Analizi ══════════
with tab1:
    if secili_urun and "hal" in data:
        urun_data = hal[hal["urun_adi"] == secili_urun].sort_values("tarih")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{urun_data["ort_fiyat"].iloc[-1]:.1f} ₺</div><div class="metric-label">Son Fiyat</div></div>', unsafe_allow_html=True)
        with col2:
            avg = urun_data["ort_fiyat"].mean()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg:.1f} ₺</div><div class="metric-label">Ortalama</div></div>', unsafe_allow_html=True)
        with col3:
            mn = urun_data["ort_fiyat"].min()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{mn:.1f} ₺</div><div class="metric-label">Minimum</div></div>', unsafe_allow_html=True)
        with col4:
            mx = urun_data["ort_fiyat"].max()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{mx:.1f} ₺</div><div class="metric-label">Maksimum</div></div>', unsafe_allow_html=True)
        
        st.markdown("#### 📊 Fiyat Trendi")
        chart_data = urun_data.set_index("tarih")[["ort_fiyat","en_dusuk","en_yuksek"]]
        st.line_chart(chart_data, use_container_width=True)
        
        # Aylık ortalama tablo
        urun_data["ay"] = urun_data["tarih"].dt.to_period("M").astype(str)
        monthly_avg = urun_data.groupby("ay")["ort_fiyat"].mean().reset_index()
        monthly_avg.columns = ["Ay", "Ortalama Fiyat (₺)"]
        monthly_avg["Ortalama Fiyat (₺)"] = monthly_avg["Ortalama Fiyat (₺)"].round(2)
        
        st.markdown("#### 📋 Aylık Ortalama Fiyatlar")
        st.dataframe(monthly_avg.tail(12), use_container_width=True, hide_index=True)

# ══════════ TAB 2: Haftalık Model ══════════
with tab2:
    st.markdown("### 🔬 Haftalık Tahmin Performansı")
    st.markdown("> Zamana göre sıralı **%80 eğitim / %20 test** ile sızıntısız değerlendirme")
    
    if "weekly_metrics" in data:
        metrics = data["weekly_metrics"]
        
        st.markdown("#### 📊 Model Karşılaştırma Tablosu")
        styled = metrics.style.format({
            "MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%",
            "MDA": "{:.2f}%", "R2": "{:.4f}"
        }).background_gradient(subset=["R2"], cmap="Greens"
        ).background_gradient(subset=["MAE","RMSE"], cmap="Reds_r")
        
        st.dataframe(styled, use_container_width=True)
        
        # Metric cards
        best = metrics["R2"].idxmax()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 En İyi R²", f"{metrics.loc[best,'R2']:.4f}", delta=best)
        with col2:
            best_mae = metrics["MAE"].idxmin()
            st.metric("🎯 En Düşük MAE", f"{metrics.loc[best_mae,'MAE']:.2f} ₺", delta=best_mae)
        with col3:
            best_mape = metrics["MAPE"].idxmin()
            st.metric("📉 En Düşük MAPE", f"{metrics.loc[best_mape,'MAPE']:.2f}%", delta=best_mape)
    else:
        st.info("Haftalık model metrikleri bulunamadı. `step3_feature_and_train.py` çalıştırın.")

# ══════════ TAB 3: Aylık Model ══════════
with tab3:
    st.markdown("### 📅 Aylık Tahmin Performansı")
    st.markdown("> Zamana göre sıralı **%80 eğitim / %20 test** ile sızıntısız değerlendirme")
    
    if "monthly_metrics" in data:
        metrics_m = data["monthly_metrics"]
        
        st.markdown("#### 📊 Model Karşılaştırma Tablosu")
        styled_m = metrics_m.style.format({
            "MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%",
            "MDA": "{:.2f}%", "R2": "{:.4f}"
        }).background_gradient(subset=["R2"], cmap="Greens"
        ).background_gradient(subset=["MAE","RMSE"], cmap="Reds_r")
        
        st.dataframe(styled_m, use_container_width=True)
        
        best_m = metrics_m["R2"].idxmax()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 En İyi R²", f"{metrics_m.loc[best_m,'R2']:.4f}", delta=best_m)
        with col2:
            best_mae_m = metrics_m["MAE"].idxmin()
            st.metric("🎯 En Düşük MAE", f"{metrics_m.loc[best_mae_m,'MAE']:.2f} ₺", delta=best_mae_m)
        with col3:
            best_mape_m = metrics_m["MAPE"].idxmin()
            st.metric("📉 En Düşük MAPE", f"{metrics_m.loc[best_mape_m,'MAPE']:.2f}%", delta=best_mape_m)
        
        # Haftalık vs Aylık karşılaştırma
        if "weekly_metrics" in data:
            st.markdown("---")
            st.markdown("#### ⚖️ Haftalık vs Aylık Karşılaştırma")
            wm = data["weekly_metrics"]
            mm = data["monthly_metrics"]
            compare = pd.DataFrame({
                "Metrik": ["MAE", "RMSE", "MAPE (%)", "MDA (%)", "R²"],
                "Haftalık (En İyi)": [
                    f"{wm['MAE'].min():.2f}",
                    f"{wm['RMSE'].min():.2f}",
                    f"{wm['MAPE'].min():.2f}",
                    f"{wm['MDA'].max():.2f}",
                    f"{wm['R2'].max():.4f}",
                ],
                "Aylık (En İyi)": [
                    f"{mm['MAE'].min():.2f}",
                    f"{mm['RMSE'].min():.2f}",
                    f"{mm['MAPE'].min():.2f}",
                    f"{mm['MDA'].max():.2f}",
                    f"{mm['R2'].max():.4f}",
                ],
            })
            st.dataframe(compare, use_container_width=True, hide_index=True)
    else:
        st.info("Aylık model metrikleri bulunamadı.")

# ══════════ TAB 4: Feature Importance ══════════
with tab4:
    st.markdown("### 🧬 Özellik Önem Skorları")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 Haftalık Model")
        if "weekly_fi" in data:
            fi_w = data["weekly_fi"].head(15)
            fi_w.columns = ["Özellik", "Önem Skoru"]
            fi_w["Önem (%)"] = (fi_w["Önem Skoru"] * 100).round(2)
            st.dataframe(fi_w[["Özellik","Önem (%)"]].reset_index(drop=True), use_container_width=True, hide_index=True)
            st.bar_chart(fi_w.set_index("Özellik")["Önem (%)"])
    
    with col2:
        st.markdown("#### 📅 Aylık Model")
        if "monthly_fi" in data:
            fi_m = data["monthly_fi"].head(15)
            fi_m.columns = ["Özellik", "Önem Skoru"]
            fi_m["Önem (%)"] = (fi_m["Önem Skoru"] * 100).round(2)
            st.dataframe(fi_m[["Özellik","Önem (%)"]].reset_index(drop=True), use_container_width=True, hide_index=True)
            st.bar_chart(fi_m.set_index("Özellik")["Önem (%)"])

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.4); font-size:0.8rem;'>
    Konya Hal Fiyat Tahmin Sistemi v2.0 | 2022-2026 | RF + LightGBM + Stacking<br>
    %80/%20 Kronolojik Split | Sızıntısız Akademik Standart
</div>
""", unsafe_allow_html=True)
