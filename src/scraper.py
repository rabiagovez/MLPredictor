"""
scraper.py - Konya Büyükşehir Belediyesi Hal Fiyatları Web Scraper
Kaynak: https://www.konya.bel.tr/hal-fiyatlari?tarih=YYYY-MM-DD

Her tarih için ürün fiyatlarını (En Düşük, En Yüksek) çeker ve
haftalık ortalamaya dönüştürerek CSV olarak kaydeder.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import re

BASE_URL = "https://www.konya.bel.tr/hal-fiyatlari"
RAW_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "konya_hal_raw.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Ürünlerin ithal durumu (1=ithal, 0=yerli)
ITHAL_DURUMU = {
    "muz": 1, "avokado": 1, "ananas": 1, "mandalina": 0,
    "portakal": 0, "limon": 0, "elma": 0, "armut": 0, "kiraz": 0,
    "çilek": 0, "kavun": 0, "karpuz": 0, "üzüm": 0, "şeftali": 0,
    "erik": 0, "domates": 0, "biber": 0, "salatalık": 0, "patlıcan": 0,
    "kabak": 0, "soğan": 0, "sarımsak": 0, "patates": 0, "havuç": 0,
    "ıspanak": 0, "marul": 0, "maydanoz": 0, "dereotu": 0, "nane": 0,
    "pırasa": 0, "brokoli": 0, "karnabahar": 0, "lahana": 0,
    "kivi": 1, "greyfurt": 0, "nar": 0, "incir": 0,
}

# Ürün hassasiyet katsayısı (lojistik maliyet çarpanı için)
# 1.0=normal, 1.5=hassas/çabuk bozulan
HASSASIYET_KATSAYISI = {
    "domates": 1.5, "çilek": 1.8, "salatalık": 1.5, "marul": 1.6,
    "ıspanak": 1.7, "maydanoz": 1.6, "dereotu": 1.7, "nane": 1.6,
    "kiraz": 1.8, "şeftali": 1.6, "erik": 1.5, "biber": 1.4,
    "patlıcan": 1.3, "kabak": 1.2, "soğan": 1.0, "sarımsak": 1.0,
    "patates": 1.0, "havuç": 1.1, "pırasa": 1.3, "brokoli": 1.5,
    "karnabahar": 1.4, "lahana": 1.2, "elma": 1.1, "armut": 1.1,
    "portakal": 1.0, "limon": 1.0, "mandalina": 1.1, "muz": 1.3,
    "kivi": 1.2, "nar": 1.1, "incir": 1.6, "avokado": 1.4,
    "üzüm": 1.4, "kavun": 1.2, "karpuz": 1.0,
}

# Mevsim faktörü: hangi ay hangi ürünler hasat sezonunda
HASAT_AYLARI = {
    "domates": [6, 7, 8, 9, 10],
    "biber": [7, 8, 9, 10],
    "salatalık": [5, 6, 7, 8, 9],
    "patlıcan": [6, 7, 8, 9],
    "kabak": [6, 7, 8],
    "elma": [9, 10, 11],
    "armut": [8, 9, 10],
    "kiraz": [5, 6],
    "çilek": [4, 5, 6],
    "şeftali": [7, 8],
    "erik": [6, 7, 8],
    "kavun": [7, 8, 9],
    "karpuz": [6, 7, 8, 9],
    "üzüm": [8, 9, 10],
    "nar": [9, 10, 11],
    "incir": [7, 8, 9],
    "soğan": list(range(1, 13)),  # yıl boyu
    "patates": list(range(1, 13)),
    "havuç": list(range(1, 13)),
    "lahana": [10, 11, 12, 1, 2, 3],
    "pırasa": [10, 11, 12, 1, 2],
    "ıspanak": [11, 12, 1, 2, 3],
    "marul": [3, 4, 5, 6, 10, 11],
    "portakal": [11, 12, 1, 2, 3],
    "mandalina": [10, 11, 12, 1],
    "limon": list(range(1, 13)),
    "muz": list(range(1, 13)),  # ithal, yıl boyu mevcut
}


def get_available_dates(start_date: str = "2024-01-01", end_date: str = None) -> list:
    """Belirtilen tarih aralığındaki tüm günleri döner."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def parse_price(price_str: str) -> float:
    """Fiyat metnini float'a dönüştürür. Örn: '12,50 TL' → 12.50"""
    if not price_str:
        return None
    clean = re.sub(r"[^\d,\.]", "", price_str.strip())
    clean = clean.replace(",", ".")
    try:
        return float(clean)
    except ValueError:
        return None


def scrape_date(date_str: str, session: requests.Session) -> list:
    """
    Belirli bir tarih için hal fiyatlarını çeker.
    Dönüş: [{'tarih': ..., 'urun': ..., 'birim': ..., 'en_dusuk': ..., 'en_yuksek': ...}, ...]
    """
    url = f"{BASE_URL}?tarih={date_str}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "utf-8"
        if resp.status_code != 200:
            return []
    except requests.RequestException as e:
        print(f"  ⚠ {date_str} isteği başarısız: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    records = []

    # Tablolar: genellikle 'table' etiketleri veya fiyat listesi div'leri
    tables = soup.find_all("table")

    if not tables:
        # Alternatif: liste yapısını ara
        price_items = soup.select(".hal-fiyat-item, .product-price-item, [class*='fiyat']")
        if not price_items:
            return []

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            texts = [c.get_text(strip=True) for c in cells]
            # Başlık satırını atla
            if any(kw in texts[0].lower() for kw in ["ürün", "mal", "stok", "madde"]):
                continue
            urun = texts[0].lower().strip()
            if len(urun) < 2:
                continue

            # Birim ve fiyat kolonlarını bul
            birim = texts[1] if len(texts) > 1 else "kg"
            en_dusuk = parse_price(texts[2]) if len(texts) > 2 else None
            en_yuksek = parse_price(texts[3]) if len(texts) > 3 else en_dusuk

            if en_dusuk is None and en_yuksek is None:
                continue

            # İthal durumu
            ithal_mi = 0
            for k, v in ITHAL_DURUMU.items():
                if k in urun:
                    ithal_mi = v
                    break

            # Hassasiyet katsayısı
            hassasiyet = 1.2  # varsayılan
            for k, v in HASSASIYET_KATSAYISI.items():
                if k in urun:
                    hassasiyet = v
                    break

            # Mevsim faktörü
            ay = datetime.strptime(date_str, "%Y-%m-%d").month
            mevsim = 0
            for k, months in HASAT_AYLARI.items():
                if k in urun and ay in months:
                    mevsim = 1
                    break

            records.append({
                "tarih": date_str,
                "urun_adi": urun,
                "birim": birim,
                "en_dusuk": en_dusuk,
                "en_yuksek": en_yuksek,
                "ithal_mi": ithal_mi,
                "hassasiyet_katsayisi": hassasiyet,
                "mevsim_faktoru": mevsim,
            })

    return records


def scrape_all(start_date: str = "2024-01-01", end_date: str = None,
               delay: float = 0.5, output_path: str = None) -> pd.DataFrame:
    """
    Tüm tarih aralığını scrape eder.
    """
    if output_path is None:
        output_path = RAW_OUTPUT

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dates = get_available_dates(start_date, end_date)
    print(f"[*] {len(dates)} tarih taranacak: {dates[0]} -> {dates[-1]}")

    session = requests.Session()
    all_records = []
    success_count = 0
    fail_count = 0

    for i, date_str in enumerate(dates):
        records = scrape_date(date_str, session)
        if records:
            all_records.extend(records)
            success_count += 1
            if (i + 1) % 50 == 0:
                print(f"  [+] {i+1}/{len(dates)} tarih islendi, {len(all_records)} kayit")
        else:
            fail_count += 1

        time.sleep(delay)

    print(f"\n[OK] Tamamlandi: {success_count} basarili, {fail_count} basarisiz")
    print(f"   Toplam kayıt: {len(all_records)}")

    if not all_records:
        print("[!] Hic veri cekilemedi! Sentetik veri uretiliyor...")
        return _generate_synthetic_data(start_date, end_date)

    df = pd.DataFrame(all_records)
    df["tarih"] = pd.to_datetime(df["tarih"])
    df["ort_fiyat"] = (df["en_dusuk"].fillna(df["en_yuksek"]) +
                       df["en_yuksek"].fillna(df["en_dusuk"])) / 2

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Kaydedildi: {output_path}")
    return df


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Günlük veriyi haftalık ortalamaya dönüştürür.
    ISO hafta kullanır (Pazartesi başlangıç).
    """
    df = df.copy()
    df["tarih"] = pd.to_datetime(df["tarih"])
    df["hafta_baslangic"] = df["tarih"].dt.to_period("W-MON").apply(
        lambda r: r.start_time
    )
    df["yil"] = df["tarih"].dt.isocalendar().year.astype(int)
    df["hafta_no"] = df["tarih"].dt.isocalendar().week.astype(int)

    weekly = (
        df.groupby(["hafta_baslangic", "yil", "hafta_no", "urun_adi",
                    "birim", "ithal_mi", "hassasiyet_katsayisi"])
        .agg(
            ort_fiyat=("ort_fiyat", "mean"),
            en_dusuk=("en_dusuk", "mean"),
            en_yuksek=("en_yuksek", "mean"),
            mevsim_faktoru=("mevsim_faktoru", "max"),
            veri_sayisi=("ort_fiyat", "count"),
        )
        .reset_index()
        .rename(columns={"hafta_baslangic": "tarih"})
    )
    return weekly.sort_values(["urun_adi", "tarih"]).reset_index(drop=True)


def _generate_synthetic_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Site erişilemez olduğunda gerçekçi sentetik veri üretir.
    Gerçek Türkiye enflasyonu ve mevsimsellik göz önüne alınır.
    """
    import numpy as np

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    np.random.seed(42)

    # Temel ürünler ve 2024 başı fiyatları (TL/kg, gerçeğe yakın)
    base_prices = {
        "domates": 18.0, "biber": 22.0, "salatalık": 15.0, "patlıcan": 20.0,
        "kabak": 14.0, "soğan": 12.0, "patates": 10.0, "havuç": 10.0,
        "ıspanak": 20.0, "marul": 18.0, "lahana": 8.0, "brokoli": 30.0,
        "karnabahar": 25.0, "pırasa": 22.0, "elma": 24.0, "armut": 28.0,
        "portakal": 20.0, "limon": 30.0, "mandalina": 25.0, "muz": 50.0,
        "kiraz": 80.0, "çilek": 45.0, "şeftali": 40.0, "üzüm": 35.0,
        "kavun": 16.0, "karpuz": 8.0, "nar": 28.0, "incir": 55.0,
    }

    # Aylık mevsimsel çarpanlar (1.0=normal, >1=pahalı, <1=ucuz)
    seasonal_factors = {
        "domates":   [2.5, 2.8, 2.0, 1.5, 1.0, 0.7, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0],
        "salatalık": [2.0, 2.2, 1.5, 1.0, 0.8, 0.7, 0.7, 0.8, 0.9, 1.2, 1.5, 1.8],
        "biber":     [2.2, 2.5, 1.8, 1.2, 0.9, 0.7, 0.6, 0.7, 0.8, 1.1, 1.6, 2.0],
        "patates":   [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0],
        "soğan":     [1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0],
        "elma":      [0.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.2, 0.8, 0.7, 0.8, 0.9],
        "muz":       [1.0] * 12,
        "portakal":  [0.8, 0.8, 0.9, 1.1, 1.3, 1.4, 1.5, 1.4, 1.2, 1.0, 0.8, 0.7],
        "kiraz":     [3.0, 3.5, 2.5, 1.5, 0.7, 0.8, 1.5, 3.0, 4.0, 4.0, 4.0, 3.5],
    }

    dates = get_available_dates(start_date, end_date)
    all_records = []

    # Aylık enflasyon etkisi ~%3 (yıllık ~%40)
    monthly_inflation = 0.03

    for date_str in dates:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        ay = dt.month
        # 2024 başından kaç ay geçti
        months_elapsed = (dt.year - 2024) * 12 + (dt.month - 1)
        inflation_mult = (1 + monthly_inflation) ** months_elapsed

        for urun, base in base_prices.items():
            sf_list = seasonal_factors.get(urun, [1.0] * 12)
            sf = sf_list[ay - 1]

            # Gürültü ±%10
            noise = np.random.uniform(0.9, 1.1)
            ort_fiyat = base * sf * inflation_mult * noise

            en_dusuk = ort_fiyat * 0.85
            en_yuksek = ort_fiyat * 1.15

            # Metadata
            ithal = ITHAL_DURUMU.get(urun, 0)
            hassasiyet = HASSASIYET_KATSAYISI.get(urun, 1.2)
            hasat_aylar = HASAT_AYLARI.get(urun, [])
            mevsim = 1 if ay in hasat_aylar else 0

            all_records.append({
                "tarih": date_str,
                "urun_adi": urun,
                "birim": "kg",
                "en_dusuk": round(en_dusuk, 2),
                "en_yuksek": round(en_yuksek, 2),
                "ort_fiyat": round(ort_fiyat, 2),
                "ithal_mi": ithal,
                "hassasiyet_katsayisi": hassasiyet,
                "mevsim_faktoru": mevsim,
            })

    df = pd.DataFrame(all_records)
    df["tarih"] = pd.to_datetime(df["tarih"])
    print(f"[+] Sentetik veri uretildi: {len(df)} kayit")
    return df


if __name__ == "__main__":
    print("[*] Konya Hal Fiyatlari Scraper baslatiliyor...")
    df_raw = scrape_all(start_date="2024-01-01", delay=0.3)
    print(f"\n[*] Ham veri istatistikleri:")
    print(df_raw.head(10).to_string())
    print(f"\nŞekil: {df_raw.shape}")
    print(f"Ürünler: {df_raw['urun_adi'].nunique()}")
    print(f"Tarih aralığı: {df_raw['tarih'].min()} → {df_raw['tarih'].max()}")
