# -*- coding: utf-8 -*-
import pandas as pd

hal = pd.read_csv("data/raw/konya_hal_raw.csv", encoding="utf-8-sig")
hal["tarih"] = pd.to_datetime(hal["tarih"])
print("GENEL:")
print(f"  Toplam kayit: {len(hal)}")
urun_count = hal["urun_adi"].nunique()
print(f"  Urun sayisi: {urun_count}")
print(f"  Tarih: {hal['tarih'].min().date()} -> {hal['tarih'].max().date()}")
print(f"  Ort fiyat min/max: {hal['ort_fiyat'].min():.2f} / {hal['ort_fiyat'].max():.2f}")
print()

for y in sorted(hal["tarih"].dt.year.unique()):
    sub = hal[hal["tarih"].dt.year == y]
    print(f"  {y}: {len(sub)} kayit, {sub['urun_adi'].nunique()} urun")
print()

top = hal.groupby("urun_adi").size().sort_values(ascending=False).head(10)
print("TOP 10 URUN:")
for u, c in top.items():
    print(f"  {u}: {c}")
print()

wm = pd.read_csv("data/processed/haftalik_metrics.csv", index_col=0)
mm = pd.read_csv("data/processed/aylik_metrics.csv", index_col=0)
print("HAFTALIK:")
print(wm.to_string())
print()
print("AYLIK:")
print(mm.to_string())
print()

wfi = pd.read_csv("data/processed/haftalik_feature_importance.csv")
mfi = pd.read_csv("data/processed/aylik_feature_importance.csv")
print("HAFTALIK FI TOP15:")
for _, r in wfi.head(15).iterrows():
    print(f"  {r['feature']}: {r['importance']:.4f}")
print()
print("AYLIK FI TOP15:")
for _, r in mfi.head(15).iterrows():
    print(f"  {r['feature']}: {r['importance']:.4f}")
