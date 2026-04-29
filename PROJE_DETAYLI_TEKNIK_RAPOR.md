# Konya Hal Fiyat Tahmin Sistemi - Detayli Teknik Rapor

## 1. Raporun Amaci

Bu dokuman, projenin basindan sonuna kadar nasil kurgulandigini teknik ve operasyonel acidan ayrintili sekilde anlatmak icin hazirlanmistir. Raporun hedefi su sorulara eksiksiz cevap vermektir:

- Sistem hangi problemi cozuyor?
- Veriler nereden geliyor, nasil toplaniyor ve nasil isleniyor?
- Hangi model/algoritmalar neden secildi?
- Parametreler hangi amacla kullanildi?
- Karsilastirma hangi metriklerle ve hangi mantikla yapildi?
- Sonuclar nasil yorumlanmali?

Bu rapor, projenin mevcut canli akisina odaklanir ve su dosyalardaki gercek uygulamayi esas alir:

- `train_live.py`
- `main.py`
- `generate_report.py`
- destekleyici olarak: `src/scraper.py`, `src/feature_engineer.py`, `src/model_trainer.py`, `src/predictor.py`

---

## 2. Problem Tanimi ve Is Hedefi

Sistem, Konya Hal piyasasindaki urun fiyatlarini haftalik bazda tahmin etmeyi amaclar. Buradaki temel is hedefleri:

- Gelecek haftaya ait olasi fiyat seviyesini ongormek
- Fiyat yonunu (artis/azalis) dogru yakalamak
- Farkli model yaklasimlarini ayni test setinde karsilastirarak en iyi karar modelini secmek
- Sonuclari yonetim/operasyon tarafinin okuyabilecegi raporlar halinde sunmak

Bu nedenle sistem sadece tek metrik odakli degildir. Hem mutlak hata, hem bagil hata, hem yon dogrulugu, hem de aciklayicilik (R2) birlikte degerlendirilir.

---

## 3. Mimari Genel Bakis

Proje 4 ana katmandan olusur:

1. **Veri katmani**  
   Hal fiyatlari + dis etken verileri (hava, kur, lojistik proxy) toplanir.

2. **Ozellik muhendisligi katmani**  
   Haftalik gecmis hafiza (lag), trend, volatilite, mevsimsellik gibi model girdileri uretilir.

3. **Modelleme katmani**  
   3 ana model (Random Forest, LightGBM, Stacking) ve bunlarin uzerinden Dynamic Ensemble kurulur.

4. **Sunum katmani**  
   Streamlit uygulamasi (`main.py`) ve otomatik gorsel rapor uretimi (`generate_report.py`).

Bu kurgu ile sistem hem gelistirme tarafinda bilimsel olarak izlenebilir, hem de kullanici tarafinda karar destek aracina donusur.

---

## 4. Veri Kaynaklari ve Elde Etme Yontemi

## 4.1 Hal fiyat verisi

Kaynak belediye hal fiyat sayfasidir (`src/scraper.py`):

- Site: Konya BSB hal fiyatlari
- Alanlar: urun, birim, en dusuk, en yuksek
- Turetilen alan: `ort_fiyat = (en_dusuk + en_yuksek)/2`

Scraper su adimlari uygular:

- Tarih araligindaki gunleri olusturur
- Her tarih icin web istegi atar
- Tablo satirlarini parse eder
- Fiyat metinlerini sayisal degere cevirir
- Urun bazli ek etiketler ekler (ithal durumu, hassasiyet katsayisi, mevsim bilgisi)
- Ham sonucu CSV olarak kaydeder

Canli egitim akisinda ham fiyat girdisi:

- `data/raw/konya_hal_raw.csv`

## 4.2 Hava verisi

Canli egitimde hava verisi su dosyadan okunur:

- `data/raw/weather_combined.csv`

Kullanilan ana etkiler:

- Konya ortalama sicaklik
- Konya toplam yagis
- Antalya don etkisinin 1 hafta gecikmeli proxy'si (`antalya_don_lag1`)
- Konya-Antalya sicaklik farki (`sic_fark`)

## 4.3 Kur verisi

Canli egitimde kur verisi su dosyadan okunur:

- `data/raw/usd_rates.csv`

Haftalik ortalama `dolar_kuru` feature olarak kullanilir.

## 4.4 Veri birlestirme prensibi

Tum kaynaklar haftalik eksene cekilir ve `hafta_baslangic` uzerinden birlestirilir. Bu secim:

- Gunluk gürültüyü azaltir
- Modeli daha stabil hale getirir
- Is kararlarina daha yakin bir zaman periyodu sunar (haftalik planlama)

---

## 5. Veri On Isleme ve Haftalik Veri Seti Kurgusu

Canli sistemde (`train_live.py`) akis:

1. Ham fiyatlar okunur
2. `tarih` parse edilir
3. Bos/hatali satirlar temizlenir
4. Urun isimleri normalize edilir (gizli unicode karakter temizligi, lowercase)
5. ISO hafta baslangici hesaplanir
6. Urun + hafta bazinda ortalama fiyat uretilir
7. Lag/rolling/momentum/trend ozellikleri urun bazinda hesaplanir
8. Hava/kur verisi merge edilir
9. Zaman ve makro ozellikler eklenir
10. Hedef degisken (`hedef_haftalik`) olusturulur

Sonuc:

- `weekly_live.csv` egitim ve uygulama tarafinin ortak veri seti olur.

---

## 6. Ozellik Muhendisligi (Parametreler, Amaclar ve Kullanimi)

Bu bolumde her ozellik grubunun islevini netlestiriyorum.

## 6.1 Zaman ozellikleri

- `yil`
- `hafta_no`
- `ay`
- `hafta_sin`
- `hafta_cos`

**Amac:** mevsimsel donguyu lineer olmayan sekilde modele tasimak.  
`hafta_sin/hafta_cos` ile yil sonu-yil basi gecisi (52->1) model tarafinda kopukluk yaratmaz.

## 6.2 Urun kimligi

- `urun_kod`

**Amac:** tek modelin farkli urun segmentlerini ayirt etmesini saglamak.

## 6.3 Gecmis fiyat hafizasi (lag feature'lar)

- `lag_1h`, `lag_4h`, `lag_8h`, `lag_12h`, `lag_24h`

**Amac:** urunun gecmis davranisini modele bellek olarak vermek.

- `lag_1h`: son haftanin dogrudan etkisi
- `lag_4h`: kisa donem (yaklasik 1 ay)
- `lag_8h`, `lag_12h`: orta donem trend
- `lag_24h`: daha uzun dongu etkisi

## 6.4 Rolling istatistikler

- `roll4_ort`, `roll8_ort`, `roll12_ort`, `roll4_std`

**Amac:** ham fiyatlari yumusatmak, trend ve volatiliteyi ozetlemek.

- rolling ortalamalar trendi
- rolling std oynakligi temsil eder

## 6.5 Momentum / trend / volatilite

- `momentum_pct`
- `trend_4_12`
- `volatilite`

**Amac:** fiyatin ivmesini, yon gucunu ve riskli hareket seviyesini modele acikca gostermek.

## 6.6 Hava etkileri

- `konya_sic`
- `konya_yagis`
- `antalya_don_lag1`
- `sic_fark`

**Amac:** arz tarafindaki iklim etkisini modele yansitmak.

Ozellikle `antalya_don_lag1`, tedarik/lojistik gecikmesi fikrine dayanan pratik bir proxy'dir.

## 6.7 Makro etkiler

- `dolar_kuru`
- `lojistik`

**Amac:** maliyet baskisini temsil etmek.

Canli akista `lojistik`, `dolar_kuru * 2.3` seklinde bir proxy ile uretilir.

---

## 7. Hedef Degisken ve Tahmin Problemi Tanimi

Hedef:

- `hedef_haftalik = bir sonraki haftanin fiyatı`

Bu tanimla problem:

- **One-step ahead weekly forecasting** (1 adim ileri haftalik tahmin)

Model, bugunun feature setinden gelecek haftanin fiyatini tahmin eder.

---

## 8. Train/Test Ayrimi ve Degerlendirme Protokolu

Canli egitimde zaman bazli ayrim:

- Train: `hafta_baslangic < 2026-01-01`
- Test: `hafta_baslangic >= 2026-01-01`

Neden bu ayrim?

- Gelecek veriyi gecmise sizdirmamak
- Gercek operasyon senaryosunu taklit etmek
- Backtest mantigi ile uyumlu olmak

Eksik degerler train median ile doldurulur. Bu sayede test seti train bilgisini assiri kullanmadan sayisal stabilite korunur.

---

## 9. Kullanilan Algoritmalar ve Calisma Mantiklari

## 9.1 Random Forest (Bagging)

Yapi:

- Birden fazla karar agaci egitilir
- Ağaçlar farkli alt örnekler/özellikler gorur
- Cikis ortalamasi alınır

Guclu yonleri:

- Tabular veride guclu baseline
- Asiri hassas ayar gerektirmeden iyi performans
- Gürültüye dayaniklilik

## 9.2 LightGBM (Boosting)

Yapi:

- Zayif ogreniciler ardisk olarak egitilir
- Her adimda onceki hatalarin kalintisi ogrenilir

Guclu yonleri:

- Kompleks dogrusal olmayan iliskileri iyi yakalar
- Performans/aciklik dengesi iyidir

## 9.3 Stacking

Yapi:

- Birden fazla modelin tahminlerini ikinci katmanda birlestirir
- Farkli bias/variance profilindeki modelleri tek karar katmaninda bir araya getirir

Bu projede:

- Base katman: RF + LGBM (canli akista)
- Meta katman: lineer birlestirme mantigi (Ridge tabanli)

Guclu yonleri:

- Tek modelin kacirdigi paterni diger model telafi edebilir
- Ortalama almaktan daha esnek kombinasyon uretir

## 9.4 Dynamic Ensemble

Yapi:

- RF/LGBM/Stacking tahminlerini urun bazli agirliklandirir
- Agirliklar son pencere performansina gore hesaplanir (hata dusukse agirlik yuksek)

Amac:

- Tek bir global agirlik yerine urune ozel adaptif birlestirme
- Farkli urunlerde farkli modelin daha iyi olmasi durumunu avantaja cevirmek

---

## 10. Metrikler ve Matematiksel Anlamlari

Su an aktif metrikler:

- MAE
- RMSE
- wMAPE(%)
- MDA
- R2

## 10.1 MAE (Mean Absolute Error)

Form:

- `MAE = mean(|y - yhat|)`

Anlam:

- Ortalama mutlak hata (TL cinsinden)
- Duserse model ortalama seviyede gercege yaklasiyor demektir

## 10.2 RMSE (Root Mean Squared Error)

Form:

- `RMSE = sqrt(mean((y - yhat)^2))`

Anlam:

- Buyuk hatalari daha sert cezalandirir
- MAE'ye gore outlier duyarliligi daha yuksektir

## 10.3 wMAPE (Weighted Mean Absolute Percentage Error)

Form:

- `wMAPE = sum(|y - yhat|) / sum(|y|) * 100`

Anlam:

- Toplam hata / toplam gercek hacim
- Is etkisi odakli metrik; dusuk hacimli satirlarin asiri sisirme etkisini azaltir

## 10.4 MDA (Mean Directional Accuracy)

Bu projede yon dogrulugu, onceki gercek haftaya gore olculur:

- `actual_dir = sign(y_t - lag_1h)`
- `pred_dir = sign(yhat_t - lag_1h)`
- `MDA = yonu dogru tahmin edilen adim orani`

Ayrica:

- Duz hareket (actual_dir=0) adimlari dislanir (`ignore_flat_actual=True`)

Anlam:

- Fiyat seviyesinden bagimsiz olarak artis/azalis yonu ne kadar dogru?

## 10.5 R2 (R-squared)

Form:

- `R2 = 1 - SS_res / SS_tot`

Anlam:

- Modelin varyansi ne kadar acikladigini olcer
- 1'e yakin daha iyi; 0 civari zayif; negatif ise ortalama modelden bile kotu

---

## 11. Metriklerle Karsilastirma Mantigi

Model secimi tek metrikle yapilmamali.

Bu projedeki pratik oncelik:

1. `wMAPE` (is etkisine yakin bagil hata)
2. `MAE/RMSE` (mutlak hata seviyesi ve buyuk hata cezasi)
3. `MDA` (trend yonu)
4. `R2` (aciklayicilik)

Neden?

- Sadece R2 yuksek diye model secmek operasyonel olarak hatali olabilir
- Sadece MAE dusuk diye model secmek de yon bilgisini kacirabilir
- Birlikte okuma daha saglikli karar verir

---

## 12. Guncel Sonuclar ve Yorum

Mevcut `models_live/metrics.csv` degerleri:

- Random Forest: `MAE=9.873`, `RMSE=22.616`, `wMAPE=13.35`, `MDA=0.6749`, `R2=0.8433`
- LightGBM: `MAE=9.968`, `RMSE=22.129`, `wMAPE=13.48`, `MDA=0.6589`, `R2=0.8500`
- Stacking: `MAE=9.517`, `RMSE=21.680`, `wMAPE=12.87`, `MDA=0.6735`, `R2=0.8560`
- Dynamic Ensemble: `MAE=9.227`, `RMSE=21.327`, `wMAPE=12.47`, `MDA=0.6778`, `R2=0.8607`

Yorum:

- **Dynamic Ensemble** tum ana metriklerde en dengeli sonucu vermektedir.
- **Stacking**, tek model olarak gucludur ama dinamik birlestirme daha iyi genel performans vermistir.
- **LightGBM**, R2 guclu olmasina ragmen yon dogrulugunda (MDA) digerlerine gore zayif kalmistir.

---

## 13. Uygulama Katmani (`main.py`) Nasıl Calisiyor?

Arayuz akis ozeti:

1. `weekly_live.csv` yuklenir
2. Model dosyalari (`random_forest.pkl`, `lightgbm.pkl`, `stacking.pkl`) yuklenir
3. Kullanici urun secer
4. Secili urun icin son veri noktasi uzerinden feature vektoru uretilir
5. Uc model tahmini alinır
6. Ortak/ensemble tahmin hesaplanir ve kartlarda gosterilir
7. Son 24 hafta gecmisi + gelecek hafta tahmini grafiklenir
8. Metrik tablosu expander icinde sunulur

Not:

- UI'da gosterilen "Ensemble", anlik ortalama kurgusuyla calisir.
- Egitimde ayrica "Dynamic Ensemble" hesaplanip raporlanir.

---

## 14. Raporlama Katmani (`generate_report.py`) Cikti Seti

Uretilen dosyalar:

- `reports_live/0_metrik_ozet_tablosu.png`
- `reports_live/1_metrik_karsilastirma.png`
- `reports_live/2_gercek_vs_tahmin.png`
- `reports_live/3_rezidual_dagilim.png`
- `reports_live/4_zaman_serisi.png`

Bu set sayesinde:

- KPI tablosu
- Model bazli metrik bar grafikleri
- Gercek-tahmin uyumu (scatter)
- Hata dagilim analizi (residual)
- Urun bazli zaman serisi kiyasi

tek bir dashboard klasoru altinda yonetime sunulabilir hale gelir.

---

## 15. Parametreler Neden Boyle Secildi?

## 15.1 Model hiperparametre mantigi

- RF: orta-yuksek agac sayisi + kontrollu derinlik = variance azaltma
- LGBM: dusuk learning rate + yeterli estimators = daha stabil boosting
- Stacking: model cesitliligi ile genelleme gucu artirma

## 15.2 Feature pencereleri

- 1,4,8,12,24 hafta secimi:
  - kisa donem + orta donem + uzun donem etkisini birlikte yakalamak icin

## 15.3 Dynamic ensemble pencere uzunlugu

- Son 12 hafta secimi:
  - hem yeterli ornek hem de guncel rejime yakinlik dengesi

## 15.4 MDA hesaplama tercihi

- Onceki gercek haftaya gore yon kiyaslama:
  - One-step ahead tahmin probleminin dogasina daha uygun

---

## 16. Guclu Yonler, Sinirlar, Riskler

## Guclu yonler

- Cok kaynakli veri (fiyat + hava + kur + lojistik)
- Zaman sızıntısını azaltan kurgu
- Coklu model + ensemble mimarisi
- Otomatik raporlama

## Sinirlar

- Dis veri kalitesi model performansina dogrudan etkili
- Urun bazinda veri yogunlugu dengesiz olabilir
- UI ensemble ile egitimdeki dynamic ensemble mantigi farkli katmanlarda ele alinir

## Riskler

- Veri akisi kesintisi (web kaynagi / CSV guncellenmeme)
- Rejim degisimi (ani piyasa soku) durumunda gecmis paternlerin gecerliligi azalabilir

---

## 17. Gelistirme Onerileri

1. UI tarafina dynamic ensemble agirliklarini da tasimak  
2. Haftalik model drift raporu eklemek  
3. Urun segmentlerine gore alt modeller kurmak (meyve/sebze/ithal gibi)  
4. Rolling backtest tablosu eklemek  
5. Feature importance dosyasini live pipeline'da da standart uretmek  

---

## 18. Sonuc

Bu proje, haftalik hal fiyat tahmini icin:

- veri toplama,
- ozellik muhendisligi,
- modelleme,
- metrik bazli karsilastirma,
- gorsel raporlama

zincirini uctan uca kurmus durumdadir.

Mevcut sonuclarda en iyi genel performansi **Dynamic Ensemble** vermektedir. Bununla birlikte sistem, tek bir metrikle degil coklu metrik dengesiyle yonetildiginde daha dogru karar uretmektedir.

Bu rapor, projenin hem teknik ekibe hem yonetsel tarafa aciklanabilir sekilde aktarilmasi icin referans dokuman olarak kullanilabilir.
