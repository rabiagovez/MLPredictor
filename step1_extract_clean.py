# -*- coding: utf-8 -*-
"""
step1_extract_clean.py - hal_fiyatlari.csv'den 2022+ veri çıkar, temizle, konya_hal_raw formatına dönüştür
"""
import pandas as pd
import numpy as np
import unicodedata, os

ITHAL = {"muz":1,"avokado":1,"ananas":1,"kivi":1,"hindistan cevizi":1}
HASSASIYET = {
    "domates":1.5,"çilek":1.8,"salatalık":1.5,"salatalik":1.5,"marul":1.6,
    "ıspanak":1.7,"ispanak":1.7,"maydanoz":1.6,"maydonoz":1.6,"dereotu":1.7,
    "dere otu":1.7,"nane":1.6,"kiraz":1.8,"şeftali":1.6,"erik":1.5,
    "biber":1.4,"patlıcan":1.3,"patlican":1.3,"kabak":1.2,"soğan":1.0,
    "sogan":1.0,"sarımsak":1.0,"sarimsak":1.0,"patates":1.0,"havuç":1.1,
    "havuc":1.1,"pırasa":1.3,"pirasa":1.3,"brokoli":1.5,"karnabahar":1.4,
    "lahana":1.2,"elma":1.1,"armut":1.1,"portakal":1.0,"limon":1.0,
    "mandalina":1.1,"muz":1.3,"kivi":1.2,"nar":1.1,"incir":1.6,
    "üzüm":1.4,"uzum":1.4,"kavun":1.2,"karpuz":1.0,"fasulye":1.3,
}
HASAT = {
    "domates":[6,7,8,9,10],"biber":[7,8,9,10],"salatalık":[5,6,7,8,9],
    "salatalik":[5,6,7,8,9],"patlıcan":[6,7,8,9],"patlican":[6,7,8,9],
    "kabak":[6,7,8],"elma":[9,10,11],"armut":[8,9,10],"kiraz":[5,6],
    "çilek":[4,5,6],"şeftali":[7,8],"erik":[6,7,8],"kavun":[7,8,9],
    "karpuz":[6,7,8,9],"üzüm":[8,9,10],"nar":[9,10,11],"incir":[7,8,9],
    "soğan":list(range(1,13)),"sogan":list(range(1,13)),
    "patates":list(range(1,13)),"havuç":list(range(1,13)),"havuc":list(range(1,13)),
    "lahana":[10,11,12,1,2,3],"pırasa":[10,11,12,1,2],"pirasa":[10,11,12,1,2],
    "ıspanak":[11,12,1,2,3],"ispanak":[11,12,1,2,3],
    "marul":[3,4,5,6,10,11],"portakal":[11,12,1,2,3],
    "limon":list(range(1,13)),"muz":list(range(1,13)),
}

def normalize_name(s):
    s = str(s).strip()
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u0307","").replace("\u0130","İ")
    s = s.lower().strip()
    return s

def get_ithal(name):
    for k,v in ITHAL.items():
        if k in name: return v
    return 0

def get_hassasiyet(name):
    for k,v in HASSASIYET.items():
        if k in name: return v
    return 1.2

def get_mevsim(name, month):
    for k,months in HASAT.items():
        if k in name and month in months: return 1
    return 0

def run():
    print("="*60)
    print("[1] hal_fiyatlari.csv'den 2022+ veri çıkarılıyor...")
    
    df = pd.read_csv("hal_fiyatlari.csv", encoding="utf-8-sig")
    df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
    df = df.dropna(subset=["tarih"])
    
    # 2022+ filtrele
    df = df[df["tarih"] >= "2022-01-01"].copy()
    print(f"   2022+ ham kayıt: {len(df)}")
    
    # Kolon dönüştürme
    df["urun_adi"] = df["urun_ad"].apply(normalize_name)
    df["birim"] = df["birim"].fillna("Kg")
    df["en_dusuk"] = pd.to_numeric(df["en_dusuk_fiyat"], errors="coerce").fillna(0)
    df["en_yuksek"] = pd.to_numeric(df["en_yuksek_fiyat"], errors="coerce").fillna(0)
    
    # Temizlik: en_dusuk > en_yuksek -> swap
    mask = df["en_dusuk"] > df["en_yuksek"]
    df.loc[mask, ["en_dusuk","en_yuksek"]] = df.loc[mask, ["en_yuksek","en_dusuk"]].values
    
    # İkisi de 0 -> sil
    df = df[~((df["en_dusuk"]==0) & (df["en_yuksek"]==0))]
    
    # Metadata ekle
    df["ithal_mi"] = df["urun_adi"].apply(get_ithal)
    df["hassasiyet_katsayisi"] = df["urun_adi"].apply(get_hassasiyet)
    month = df["tarih"].dt.month
    df["mevsim_faktoru"] = [get_mevsim(n,m) for n,m in zip(df["urun_adi"], month)]
    df["ort_fiyat"] = (df["en_dusuk"] + df["en_yuksek"]) / 2
    
    # ort_fiyat 0 veya negatif -> sil
    df = df[df["ort_fiyat"] > 0]
    
    print(f"   Temizlik sonrası: {len(df)} kayıt")
    
    # Duplikat temizle (aynı tarih+ürün birden fazla -> ortalama al)
    df = df.groupby(["tarih","urun_adi","birim"]).agg(
        en_dusuk=("en_dusuk","mean"),
        en_yuksek=("en_yuksek","mean"),
        ithal_mi=("ithal_mi","first"),
        hassasiyet_katsayisi=("hassasiyet_katsayisi","first"),
        mevsim_faktoru=("mevsim_faktoru","max"),
        ort_fiyat=("ort_fiyat","mean"),
    ).reset_index()
    
    print(f"   Duplikat temizliği sonrası: {len(df)} kayıt")
    
    # İstikrarsız ürünleri çıkar: toplam hafta sayısı < 20 olan ürünler
    df["hafta"] = df["tarih"].dt.to_period("W-MON")
    urun_hafta = df.groupby("urun_adi")["hafta"].nunique()
    stabil_urunler = urun_hafta[urun_hafta >= 20].index
    removed = set(df["urun_adi"].unique()) - set(stabil_urunler)
    if removed:
        print(f"   Kaldırılan istikrarsız ürünler ({len(removed)}): {sorted(removed)[:10]}...")
    df = df[df["urun_adi"].isin(stabil_urunler)].copy()
    df = df.drop(columns=["hafta"])
    
    # IQR outlier clipping (ürün bazlı)
    clipped = 0
    for urun in df["urun_adi"].unique():
        mask = df["urun_adi"] == urun
        q1 = df.loc[mask,"ort_fiyat"].quantile(0.05)
        q3 = df.loc[mask,"ort_fiyat"].quantile(0.95)
        iqr = q3 - q1
        lo = max(0, q1 - 2*iqr)
        hi = q3 + 2*iqr
        before = df.loc[mask,"ort_fiyat"].copy()
        df.loc[mask,"ort_fiyat"] = df.loc[mask,"ort_fiyat"].clip(lo, hi)
        df.loc[mask,"en_dusuk"] = df.loc[mask,"en_dusuk"].clip(lo, hi)
        df.loc[mask,"en_yuksek"] = df.loc[mask,"en_yuksek"].clip(lo, hi)
        clipped += (before != df.loc[mask,"ort_fiyat"]).sum()
    print(f"   Outlier clip: {clipped} değer")
    
    # Tarih formatı
    df["tarih"] = df["tarih"].dt.strftime("%Y-%m-%d")
    
    # Kaydet
    out_cols = ["tarih","urun_adi","birim","en_dusuk","en_yuksek","ithal_mi","hassasiyet_katsayisi","mevsim_faktoru","ort_fiyat"]
    df = df[out_cols].sort_values(["tarih","urun_adi"]).reset_index(drop=True)
    
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/konya_hal_raw.csv", index=False, encoding="utf-8-sig")
    
    print(f"\n   ✅ Kaydedildi: data/raw/konya_hal_raw.csv")
    print(f"   Toplam: {len(df)} kayıt, {df['urun_adi'].nunique()} ürün")
    print(f"   Tarih: {df['tarih'].min()} → {df['tarih'].max()}")
    print(f"   Örnek ürünler: {sorted(df['urun_adi'].unique())[:10]}")

if __name__ == "__main__":
    run()
