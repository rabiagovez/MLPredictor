# Konya Hal Fiyatları Tahmin Sistemi - Kapsamlı Teknik Sunum Raporu

## 1. Yönetici Özeti (Executive Summary)
Bu proje, Konya Halindeki tarımsal ürünlerin fiyat hareketlerini yüksek doğrulukla tahmin etmek amacıyla geliştirilmiş uçtan uca bir makine öğrenmesi sistemidir. Proje kapsamında **2022-2026 yılları arasındaki 37.010 adet fiyat kaydı**, meteorolojik veriler ve makroekonomik göstergelerle zenginleştirilerek modellenmiştir. 

Geleneksel zaman serisi problemlerinde yaşanan "hedef değişkenin yönünü tutturamama" (düşük MDA - Mean Directional Accuracy) sorunu, projeye **First-Order Differencing (Birinci Dereceden Fark Alma)** yönteminin entegre edilmesiyle başarıyla çözülmüş ve sistemin yön tahmin başarısı (MDA) **%78** seviyelerine çıkarılmıştır.

## 2. Veri Mimarisi ve Ön İşleme (Data Pipeline & Preprocessing)

### 2.1 Veri Kaynakları ve Kapsamı
- **Ana Veri Seti:** 03 Ocak 2022 - 24 Nisan 2026 tarihleri arasını kapsayan hal fiyatları.
- **Hacim:** Başlangıçta gürültülü olan veriler temizlenerek **197 farklı ürün** için toplam **37.010** temiz kayda ulaşılmıştır.
- **Dışsal Veriler (External Data):**
  - **Hava Durumu:** Open-Meteo Archive API kullanılarak Konya ve Antalya için günlük sıcaklık, nem, yağış ve don olayları (1.579 gün).
  - **Makroekonomik Veriler:** Günlük dolar kuru (USD/TRY), motorin fiyatları ve aylık TÜFE enflasyon oranları.

### 2.2 Veri Temizleme (Data Cleaning) Stratejisi
Veri sızıntısını ve model yanılsamasını önlemek için agresif bir temizlik yapıldı:
1. **İstikrarsız Ürünlerin Çıkarılması:** Toplamda 20 haftadan az verisi bulunan (sezonsallığı çok zayıf veya nadir gelen) 34 ürün tamamen veri setinden çıkarıldı.
2. **Outlier (Aykırı Değer) Yönetimi:** Her ürün özelinde IQR (Interquartile Range) yöntemi kullanılarak istatistiksel aykırılıklar sınırlandırıldı (clipping).
3. **Tutarlılık Kontrolü:** `en_dusuk > en_yuksek` olan hatalı kayıtlar düzeltildi, fiyatı 0 olan kayıtlar silindi ve aynı güne ait çoklu kayıtların ortalaması alındı.

## 3. Özellik Mühendisliği (Feature Engineering)

Zaman serisi projelerinde en kritik aşama veri sızıntısı (data leakage) yaratmadan özellikleri üretmektir. Tüm hesaplamalarda `shift()` metodu kullanılarak modelin "geleceği görmesi" kesinlikle engellenmiştir.

### 3.1 Üretilen Özellik Grupları
- **Gecikmeli (Lag) Özellikler:** Geçmiş 1, 2, 4, 8 ve 12 haftanın fiyatları.
- **Hareketli Ortalamalar (Rolling Stats):** Son 4, 8 ve 12 haftalık fiyat ortalamaları ve standart sapmaları (volatilite).
- **Momentum ve Trend:** Fiyatların yüzdesel değişimi, kısa/uzun dönem trend oranları (ör: 4 haftalık ortalama / 12 haftalık ortalama).
- **Zaman/Sezonsallık:** Hafta numarası ve ay bilgileri, modelin döngüselliği daha iyi anlaması için Trigonometrik dönüşümlerden (Sin/Cos) geçirildi.
- **Ürün Metadataları:** Ürünün ithal olup olmadığı, hassasiyet katsayısı ve hasat mevsimi faktörü.

## 4. Teknik Dönüşüm: First-Order Differencing

Projenin en yenilikçi ve başarıyı getiren teknik hamlesidir.

**Problem:** Modeller doğrudan "gelecek haftanın fiyatını" (örneğin 25 TL) tahmin etmeye çalıştığında, geçmiş fiyatlara çok yakın değerler üreterek (ör: 24.8 TL) RMSE/MAE metriklerinde çok başarılı görünüyor, ancak fiyatın *artacağı* veya *düşeceği* yönünü (MDA) %50 şans seviyesinde veya altında tahmin ediyordu.

**Çözüm:** Modelin hedef değişkeni (Target) mutlak fiyattan **Fiyat Farkına (Delta: $Price_{t+1} - Price_t$)** dönüştürüldü.
- Model artık "Haftaya domates 25 TL olacak" yerine "Haftaya domates **+1.5 TL artacak**" tahminini yapmaya odaklandı.
- Değerlendirme aşamasında tahmin edilen delta, mevcut fiyata eklenerek mutlak fiyat metrikleri (MAE, RMSE, MAPE) geri kazanıldı.

**Sonuç:** Bu mimari değişiklik sayesinde yön bilme başarısı (MDA) sıçrama yaptı.

## 5. Model Eğitimi ve Doğrulama (Training & Validation)

- **Veri Bölümlemesi (Splitting):** Zaman serisi bütünlüğünü korumak adına rastgele (shuffle) ayrım YAPILMADI. Veriler kronolojik olarak sıralandı; ilk %80'lik kısım eğitim (Train), son %20'lik kısım test seti olarak belirlendi.
- **Algoritmalar:** 
  1. **Random Forest:** Doğrusal olmayan ilişkileri yakalamada güçlü.
  2. **LightGBM:** Gradyan artırma tabanlı, hızlı ve optimize.
  3. **Stacking Regressor:** RF ve LightGBM'in tahminlerini alarak Ridge/ElasticNet ile nihai kararı veren meta-model.

## 6. Performans Sonuçları (Model Metrikleri)

### 6.1 Haftalık Tahmin Modeli Sonuçları
*Hedef: 1 Hafta sonrasının ortalama fiyatı*

| Model | MAE (Hata Tutarı) | RMSE | MAPE (Yüzdesel Hata) | MDA (Yön Bilme Başarısı) | R² (Açıklanabilirlik) |
|-------|-------------------|------|----------------------|--------------------------|-----------------------|
| **Stacking** | 2.87 ₺ | 9.19 | **%4.79** | %74.87 | **0.9661** |
| **Random Forest** | 3.50 ₺ | 9.54 | %5.79 | **%78.38** | 0.9635 |
| **LightGBM** | 4.38 ₺ | 10.97 | %6.64 | %73.90 | 0.9518 |

> **Yorum:** Random Forest fiyatın yönünü bilmede (MDA) %78'lik rekor bir başarı gösterirken, Stacking modeli yüzdesel hata (MAPE) açısından %4.79 ile en yüksek hassasiyeti yakalamıştır.

### 6.2 Aylık Tahmin Modeli Sonuçları
*Hedef: 1 Ay (4 Hafta) sonrasının ortalama fiyatı*

| Model | MAE (Hata Tutarı) | RMSE | MAPE (Yüzdesel Hata) | MDA (Yön Bilme Başarısı) | R² (Açıklanabilirlik) |
|-------|-------------------|------|----------------------|--------------------------|-----------------------|
| **Random Forest** | 8.75 ₺ | 16.22 | %15.31 | **%69.49** | **0.8931** |
| **Stacking** | **8.35 ₺** | 16.80 | **%14.48** | %68.57 | 0.8852 |
| **LightGBM** | 10.16 ₺ | 19.87 | %19.92 | %67.65 | 0.8395 |

> **Yorum:** Vade uzadıkça belirsizlik artsa da (MAPE %14-15 bandında), model aylık periyotta da yön değişimlerini %69 başarıyla öngörebilmektedir.

## 7. Özellik Önem Skorları (Feature Importance)

Modelin karar mekanizmasında en çok etkilenen faktörler:

**Haftalık Modelde En Etkili Faktörler:**
1. **`fiyat_lag1` (%26.9):** Bir önceki haftanın fiyatı.
2. **`aylik_tufe` (%7.6):** Enflasyon verisi (Makro ekonomik baskı).
3. **`ay` (%7.3):** Sezonsallık etkisi.
4. **`fiyat_diff_1` (%6.2):** Son haftadaki fiyat değişim hızı (Momentum).

**Aylık Modelde En Etkili Faktörler:**
1. **`fiyat_diff_1` (%10.0):** Kısa vadeli değişim momentumu.
2. **`aylik_tufe` (%8.9):** Uzun vadede enflasyon etkisi çok daha belirgin.
3. **`fiyat_lag1` (%8.0):** Geçmiş fiyat.
4. **Hava Durumu (`konya_ort_sicaklik`, `antalya_ort_sicaklik`):** Aylık periyotta iklim faktörleri ilk 10'a girmektedir.

## 8. Dashboard ve Kullanıcı Arayüzü

Tüm bu karmaşık mimari **Streamlit** tabanlı interaktif bir Dashboard üzerinde görselleştirilmiştir.
- Kullanıcı ürün bazlı tarihsel fiyat trendlerini inceleyebilir.
- Hem haftalık hem aylık modellerin performans metriklerini anlık karşılaştırabilir.
- Modellerin hangi özelliklere (features) ağırlık verdiğini Feature Importance sekmesinden görsel olarak analiz edebilir.

## 9. Gelecek Geliştirmeler (Future Scope)
- **Dışsal Verilerin Canlı Entegrasyonu:** TCMB, TÜİK ve Open-Meteo API'lerinin canlı veri akışlarına bağlanması.
- **Derin Öğrenme Modelleri:** LSTM veya Temporal Fusion Transformers (TFT) gibi daha karmaşık dizisel algoritmaların denenmesi.
- **Kategori Bazlı Modeller:** Meyve, sebze ve ithal ürünler için genel bir model yerine kategoriye özel alt modeller (Sub-models) eğitilmesi.
