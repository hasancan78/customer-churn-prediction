# Müşteri Ayrılma Tahmini (Customer Churn Prediction)
**Türkiye Yapay Zeka Akademisi - Makine Öğrenmesi Ara Ödevi**

Bu projede, sentetik müşteri verisi kullanılarak temel makine öğrenmesi yaşam döngüsü (Veri Hazırlama, Ön İşleme, Feature Engineering, Model Eğitimi ve Değerlendirme) uygulanmıştır.

---

## Proje Amacı

Müşterilerin hizmeti bırakma (churn) durumlarını tahmin etmek amacıyla **Logistic Regression** ve **K-Nearest Neighbors (KNN)** algoritmaları eğitilmiş, doğrulama (validation) setindeki performanslarına göre en iyi model seçilerek test seti üzerinde nihai değerlendirmesi yapılmıştır.

---

## Proje Yapısı

* `customer_churn.py`: Veri üretimi, ön işleme, model eğitimi ve test adımlarını içeren ana Python kodu.
* `requirements.txt`: Proje için gerekli Python kütüphaneleri.
* `README.md`: Proje dokümantasyonu.

---

## Uygulanan Adımlar

1. **Veri Üretimi & İnceleme:** 200 satırlık sentetik veri seti oluşturuldu. İlk satırlar, veri tipleri, hedef değişken dağılımı ve eksik değerler incelendi.
2. **Feature Engineering:** `destek_talebi_var_mi` (binary) ve `gelir_grubu` (kategorik) öznitelikleri türetildi.
3. **Veri Ön İşleme (Pipeline):**
   * Sayısal değişkenler için `SimpleImputer(median)` ve `StandardScaler` kullanıldı.
   * Kategorik değişkenler için `SimpleImputer(most_frequent)` ve `OneHotEncoder` uygulandı.
4. **Veri Bölme:** Veri seti Train (%70), Validation (%15) ve Test (%15) olarak `stratify` parametresiyle bölündü.
5. **Model Eğitimi & Karşılaştırma:**
   * Logistic Regression ve KNN modelleri eğitildi.
   * Validation setindeki **F1-Score** değerlerine göre en iyi model seçildi.
6. **Test Değerlendirmesi:** Seçilen model test verisi üzerinde çalıştırılarak Confusion Matrix, Accuracy, Precision, Recall ve F1-Score metrikleri hesaplandı.

---

## Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için:

1. Depoyu klonlayın veya indirin:
   ```bash
   git clone <REPOSITORY_LINKI>
   cd <KLASOR_ADI>
2. Gerekli kütüphaneleri yükleyin ve python dosyasını çalıştırın:
   py -m pip install -r requirements.txt
   py customer_churn.py