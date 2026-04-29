# -*- coding: utf-8 -*-
"""
main.py - Konya Hal Fiyatlari Tahmin Dashboard
Streamlit tabanli premium, dinamik tasarimli arayuz.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── AYARLAR VE TASARIM ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Konya Sebze Meyve Fiyat Tahmini",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS
st.markdown("""
<style>
    /* Dark Mode Premium Theme */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
    }
    
    /* Header Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #58a6ff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(33, 38, 45, 0.6);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(88, 166, 255, 0.2);
        border-color: #58a6ff;
    }
    
    .metric-title {
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #fff;
    }
    
    /* Model Tag */
    .model-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 10px;
    }
    
    .tag-rf { background: rgba(46, 204, 113, 0.2); color: #2ecc71; border: 1px solid #2ecc71; }
    .tag-lgbm { background: rgba(52, 152, 219, 0.2); color: #3498db; border: 1px solid #3498db; }
    .tag-stack { background: rgba(155, 89, 182, 0.2); color: #9b59b6; border: 1px solid #9b59b6; }
    .tag-ensemble { background: rgba(241, 196, 15, 0.2); color: #f1c40f; border: 1px solid #f1c40f; }
</style>
""", unsafe_allow_html=True)


# ── VERI VE MODEL YUKLEME ─────────────────────────────────────────────────────

MODELS_DIR = "models_live"
DATA_PATH = "weekly_live.csv"
UI_HIDDEN_PRODUCTS = {
    "nektari",
    "mandalina",
    "erik (anjelika)",
    "erik (mürdüm)",
    "havuç (ikinci)",
    "hirtlak",
    "incir (siyah)",
    "incir (beyaz)",
    "kavun (ithal)",
    "kiraz (muhtelif)",
    "limon (file)",
    "mandalina (satsuma)",
    "nar (çekirdeksiz)",
    "salatalik (kornişon)",
    "sarimsak (taze)",
    "çağla (badem)",
    "şeftali (muhtelif)",
}

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df["hafta_baslangic"] = pd.to_datetime(df["hafta_baslangic"])
    return df

@st.cache_resource
def load_models():
    models = {}
    for name in ["random_forest", "lightgbm", "stacking"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    feat_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    if os.path.exists(feat_path):
        models["features"] = joblib.load(feat_path)
    
    return models


# ── TAHMIN MOTORU ─────────────────────────────────────────────────────────────

def get_prediction(urun_adi, df, models):
    # Global en son haftayi bul
    global_last_date = df["hafta_baslangic"].max()
    next_date = global_last_date + timedelta(weeks=1)

    urun_df = df[df["urun_adi"] == urun_adi].sort_values("hafta_baslangic")
    if urun_df.empty: return None
    
    # Eger urun son haftalarda satilmadiysa, aradaki haftalari 0 ile doldur
    all_weeks = pd.to_datetime(sorted(df["hafta_baslangic"].unique()))
    urun_df = urun_df.set_index("hafta_baslangic").reindex(all_weeks).rename_axis("hafta_baslangic").reset_index()
    
    # Eksik verileri doldur (fiyat 0 olsun, diger makro verileri ffill yap)
    urun_df["urun_adi"] = urun_adi
    urun_df["urun_kod"] = df[df["urun_adi"] == urun_adi]["urun_kod"].iloc[0]
    urun_df["fiyat"] = urun_df["fiyat"].fillna(0)
    
    # Lag'lari yeniden hesapla (cunku araya 0'lar girdi)
    p = urun_df["fiyat"]
    urun_df["lag_1h"] = p.shift(1).fillna(0)
    urun_df["lag_4h"] = p.shift(4).fillna(0)
    urun_df["lag_8h"] = p.shift(8).fillna(0)
    urun_df["lag_12h"] = p.shift(12).fillna(0)
    urun_df["lag_24h"] = p.shift(24).fillna(0)
    urun_df["roll4_ort"] = p.shift(1).rolling(4, min_periods=1).mean().fillna(0)
    urun_df["roll8_ort"] = p.shift(1).rolling(8, min_periods=1).mean().fillna(0)
    urun_df["roll12_ort"] = p.shift(1).rolling(12, min_periods=1).mean().fillna(0)
    urun_df["roll4_std"] = p.shift(1).rolling(4, min_periods=1).std().fillna(0)
    urun_df["momentum_pct"] = p.shift(1).pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    urun_df["trend_4_12"] = urun_df["roll4_ort"] - urun_df["roll12_ort"]
    urun_df["volatilite"] = urun_df["roll4_std"]

    # Makro verileri ffill ile ileri tasi
    cols_to_ffill = ["konya_sic", "konya_yagis", "antalya_don_lag1", "sic_fark", "dolar_kuru", "lojistik"]
    for c in cols_to_ffill:
        if c in urun_df.columns:
            urun_df[c] = urun_df[c].ffill()

    # En son haftayi (global) al
    last_row = urun_df.iloc[-1].copy()
    
    # Feature vektorunu olustur
    X_dict = {}
    avail_feats = models.get("features", [])
    for f in avail_feats:
        if f in last_row:
            X_dict[f] = last_row[f]
            
    # Gelecek hafta zaman parametreleri
    X_dict["yil"] = next_date.year
    X_dict["hafta_no"] = next_date.isocalendar().week
    X_dict["ay"] = next_date.month
    X_dict["hafta_sin"] = np.sin(2 * np.pi * X_dict["hafta_no"] / 52)
    X_dict["hafta_cos"] = np.cos(2 * np.pi * X_dict["hafta_no"] / 52)
    
    X_df = pd.DataFrame([X_dict])
    X_df = X_df[avail_feats]
    
    # Eger urunun son 1 aylik fiyati 0 ise (tamamen sezonsuz), modeli zorlama ve 0 don
    if last_row["roll4_ort"] == 0 and last_row["fiyat"] == 0:
        preds = {"RF": 0.0, "LGBM": 0.0, "Stacking": 0.0, "Ensemble": 0.0}
    else:
        preds = {}
        try:
            if "random_forest" in models: preds["RF"] = float(models["random_forest"].predict(X_df)[0])
            if "lightgbm" in models: preds["LGBM"] = float(models["lightgbm"].predict(X_df)[0])
            if "stacking" in models: preds["Stacking"] = float(models["stacking"].predict(X_df)[0])
            preds["Ensemble"] = np.mean(list(preds.values()))
        except Exception as e:
            st.error(f"Tahmin hatasi: {e}")
            return None
        
    return {
        "date": next_date,
        "current_price": last_row["fiyat"],
        "preds": preds,
        "history": urun_df[["hafta_baslangic", "fiyat"]].tail(24)
    }


# ── ARAYUZ ────────────────────────────────────────────────────────────────────

def main():
    st.markdown("<h1>🍅 Konya Hal Fiyatları Tahmin Sistemi (AI)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; font-size:1.1rem; margin-bottom:30px;'>Gerçek veriler, Meteoroloji ve TCMB makro göstergeleriyle desteklenen gelişmiş fiyat öngörü motoru.</p>", unsafe_allow_html=True)

    df = load_data()
    models = load_models()

    if df is None or not models:
        st.warning("⏳ Veriler veya modeller yüklenemedi. Lütfen önce veri toplama ve eğitim scriptlerini çalıştırın (train_live.py).")
        return

    exclude_words = ["acur", "barbunya", "salkim çeri", "salçalik", "yaprak", "asma"]
    urunler = sorted([
        u for u in df["urun_adi"].unique()
        if (u.lower() not in UI_HIDDEN_PRODUCTS) and (not any(x in u.lower() for x in exclude_words))
    ])
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h3>🔍 Parametreler</h3>", unsafe_allow_html=True)
        secilen_urun = st.selectbox("Hedef Ürün", urunler)

    # Main Content
    res = get_prediction(secilen_urun, df, models)
    if not res:
        st.error("Bu ürün için yeterli veri yok.")
        return

    # Info Banner
    tarih_str = res["date"].strftime("%d %B %Y")
    hafta_no = res["date"].isocalendar().week
    st.markdown(f"""
    <div style="background: rgba(88, 166, 255, 0.1); border-left: 4px solid #58a6ff; padding: 15px; border-radius: 8px; margin-bottom: 25px;">
        <span style="color: #58a6ff; font-weight: bold; font-size: 1.2rem;">Tahmin Haftası:</span> 
        <span style="color: #fff; font-size: 1.2rem; margin-left: 10px;">{tarih_str} (Yılın {hafta_no}. Haftası)</span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Ortak (Ensemble) Tahmin</div>
            <div class="metric-value">{res['preds']['Ensemble']:.2f} ₺</div>
            <div class="model-tag tag-ensemble">👑 Güvenli Karar</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Random Forest</div>
            <div class="metric-value">{res['preds'].get('RF', 0):.2f} ₺</div>
            <div class="model-tag tag-rf">🌲 Bagging</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">LightGBM</div>
            <div class="metric-value">{res['preds'].get('LGBM', 0):.2f} ₺</div>
            <div class="model-tag tag-lgbm">⚡ Boosting</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Stacking Model</div>
            <div class="metric-value">{res['preds'].get('Stacking', 0):.2f} ₺</div>
            <div class="model-tag tag-stack">🏆 Meta-Learner</div>
        </div>
        """, unsafe_allow_html=True)

    # Trend Değerlendirmesi
    diff = res['preds']['Ensemble'] - res['current_price']
    pct = (diff / res['current_price']) * 100 if res['current_price'] > 0 else 0
    
    if diff > 0:
        trend_msg = f"<span style='color: #e74c3c;'>Fiyatta artış bekleniyor (+{pct:.1f}%)</span>"
    elif diff < 0:
        trend_msg = f"<span style='color: #2ecc71;'>Fiyatta düşüş bekleniyor ({pct:.1f}%)</span>"
    else:
        trend_msg = "<span style='color: #f1c40f;'>Fiyatın stabil kalması bekleniyor.</span>"

    st.markdown(f"### 📈 {secilen_urun} İçin Gelecek Hafta Görünümü")
    st.markdown(f"Mevcut Hafta Fiyatı: **{res['current_price']:.2f} ₺** | Tahmin: **{res['preds']['Ensemble']:.2f} ₺** → {trend_msg}", unsafe_allow_html=True)

    # Grafikler
    st.markdown("<br>", unsafe_allow_html=True)
    
    hist = res["history"].copy()
    # Gelecek tahmini de grafige ekle
    future_row = pd.DataFrame([{
        "hafta_baslangic": res["date"],
        "fiyat": res["preds"]["Ensemble"],
        "Tip": "Tahmin"
    }])
    hist["Tip"] = "Gerçek Veri"
    
    plot_df = pd.concat([hist, future_row])
    
    fig = px.line(plot_df, x="hafta_baslangic", y="fiyat", color="Tip",
                  color_discrete_map={"Gerçek Veri": "#58a6ff", "Tahmin": "#f1c40f"},
                  markers=True, title=f"{secilen_urun} Son 6 Aylık Fiyat Trendi ve Gelecek Hafta Tahmini")
                  
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        xaxis_title="Tarih",
        yaxis_title="Ortalama Fiyat (₺)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Grid çizgilerini hafiflet
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    # Nokta boyutunu artir
    fig.update_traces(marker=dict(size=8))
    
    # Tahmin noktasini vurgula
    fig.add_annotation(
        x=res["date"], y=res["preds"]["Ensemble"],
        text="Gelecek Hafta Tahmini",
        showarrow=True, arrowhead=2, arrowcolor="#f1c40f", arrowsize=1, arrowwidth=2,
        font=dict(color="#f1c40f", size=12),
        ay=-40
    )

    st.plotly_chart(fig, use_container_width=True)
    
if __name__ == "__main__":
    main()
