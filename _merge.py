# -*- coding: utf-8 -*-
import pandas as pd

# Step1 output - 2022-2023
df1 = pd.read_csv("data/raw/konya_hal_raw_backup.csv", encoding="utf-8-sig")
df1["tarih"] = pd.to_datetime(df1["tarih"])
df1 = df1[df1["tarih"] < "2024-01-01"]
print(f"2022-2023: {len(df1)}")

# Git restored - 2024+
df2 = pd.read_csv("data/raw/konya_hal_raw.csv", encoding="utf-8-sig")
df2["tarih"] = pd.to_datetime(df2["tarih"])
df2 = df2.drop_duplicates(subset=["tarih","urun_adi"], keep="last")
print(f"2024+: {len(df2)}")

# Birlestir
df = pd.concat([df1, df2], ignore_index=True)
df = df.sort_values(["tarih","urun_adi"]).reset_index(drop=True)
df = df.drop_duplicates(subset=["tarih","urun_adi"], keep="last")

df.to_csv("data/raw/konya_hal_raw.csv", index=False, encoding="utf-8-sig")
tmin = df["tarih"].min()
tmax = df["tarih"].max()
nu = df["urun_adi"].nunique()
print(f"Birlestirildi: {len(df)} kayit, {tmin} -> {tmax}, {nu} urun")
