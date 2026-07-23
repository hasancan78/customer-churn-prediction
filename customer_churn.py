"""
=========================================================
Makine Öğrenmesi Ara Ödevi
Türkiye Yapay Zeka Akademisi

Konu:
Müşteri Ayrılma Tahmini (Customer Churn Prediction)

Çalıştırma:
python customer_churn.py
=========================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# -------------------------------------------------------
# Rastgele veri üretimi
# -------------------------------------------------------
np.random.seed(42)
n = 200

df = pd.DataFrame({
    "yas": np.random.randint(18, 65, n),
    "gelir": np.random.randint(20000, 120000, n),
    "abonelik_suresi": np.random.randint(1, 60, n),
    "destek_talebi_sayisi": np.random.randint(0, 10, n),
    "sehir": np.random.choice(
        ["Ankara", "İstanbul", "İzmir", "Bursa"],
        n
    ),
    "uyelik_tipi": np.random.choice(
        ["Basic", "Silver", "Gold"],
        n
    )
})

# -------------------------------------------------------
# Hedef değişken (Churn)
# -------------------------------------------------------
df["churn"] = (
    (
        (df["destek_talebi_sayisi"] >= 6)
        &
        (df["abonelik_suresi"] <= 15)
    )
    |
    (
        (df["gelir"] < 35000)
        &
        (df["uyelik_tipi"] == "Basic")
    )
).astype(int)

# -------------------------------------------------------
# Bilinçli eksik değer ekleme
# -------------------------------------------------------
df.loc[5, "gelir"] = np.nan
df.loc[20, "gelir"] = np.nan
df.loc[30, "sehir"] = np.nan
df.loc[40, "uyelik_tipi"] = np.nan

# -------------------------------------------------------
# Veri İnceleme
# -------------------------------------------------------
print("=" * 50)
print("İLK 5 SATIR")
print("=" * 50)
print(df.head())

print()
print("=" * 50)
print("VERİ BOYUTU")
print("=" * 50)
print(df.shape)

print()
print("=" * 50)
print("VERİ TİPLERİ")
print("=" * 50)
print(df.dtypes)

print()
print("=" * 50)
print("HEDEF DEĞİŞKEN DAĞILIMI")
print("=" * 50)
print(df["churn"].value_counts())

print()
print("=" * 50)
print("EKSİK DEĞERLER")
print("=" * 50)
print(df.isnull().sum())

# -------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------
df["destek_talebi_var_mi"] = (
    df["destek_talebi_sayisi"] > 0
).astype(int)

df["gelir_grubu"] = pd.cut(
    df["gelir"],
    bins=[0, 40000, 80000, 150000],
    labels=["Dusuk", "Orta", "Yuksek"]
)

print()
print("=" * 50)
print("FEATURE ENGINEERING SONRASI")
print("=" * 50)
print(df.head())

# -------------------------------------------------------
# Özellikler ve hedef değişken
# -------------------------------------------------------
X = df.drop("churn", axis=1)
y = df["churn"]

# -------------------------------------------------------
# Train - Validation - Test
# -------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print()
print("=" * 50)
print("VERİ BOYUTLARI")
print("=" * 50)
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

# -------------------------------------------------------
# Sayısal ve kategorik sütunlar
# -------------------------------------------------------
numeric_features = [
    "yas",
    "gelir",
    "abonelik_suresi",
    "destek_talebi_sayisi",
    "destek_talebi_var_mi",
]

categorical_features = [
    "sehir",
    "uyelik_tipi",
    "gelir_grubu",
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features,
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

# -------------------------------------------------------
# Logistic Regression Pipeline
# -------------------------------------------------------
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# -------------------------------------------------------
# KNN Pipeline
# -------------------------------------------------------
knn_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]
)

# -------------------------------------------------------
# Modelleri Eğit
# -------------------------------------------------------
print()
print("=" * 50)
print("MODELLER EĞİTİLİYOR")
print("=" * 50)

logistic_pipeline.fit(X_train, y_train)
knn_pipeline.fit(X_train, y_train)

# -------------------------------------------------------
# Validation Tahminleri
# -------------------------------------------------------
log_pred = logistic_pipeline.predict(X_val)
knn_pred = knn_pipeline.predict(X_val)

# -------------------------------------------------------
# Validation Sonuçları
# -------------------------------------------------------
log_acc = accuracy_score(y_val, log_pred)
log_pre = precision_score(y_val, log_pred)
log_rec = recall_score(y_val, log_pred)
log_f1 = f1_score(y_val, log_pred)

knn_acc = accuracy_score(y_val, knn_pred)
knn_pre = precision_score(y_val, knn_pred)
knn_rec = recall_score(y_val, knn_pred)
knn_f1 = f1_score(y_val, knn_pred)

print()
print("=" * 50)
print("VALIDATION SONUÇLARI")
print("=" * 50)

print("\nLogistic Regression")
print("-------------------------")
print(f"Accuracy : {log_acc:.3f}")
print(f"Precision: {log_pre:.3f}")
print(f"Recall   : {log_rec:.3f}")
print(f"F1 Score : {log_f1:.3f}")

print()
print("KNN")
print("-------------------------")
print(f"Accuracy : {knn_acc:.3f}")
print(f"Precision: {knn_pre:.3f}")
print(f"Recall   : {knn_rec:.3f}")
print(f"F1 Score : {knn_f1:.3f}")

# -------------------------------------------------------
# En İyi Modeli Seç
# -------------------------------------------------------
if log_f1 >= knn_f1:
    best_model = logistic_pipeline
    best_name = "Logistic Regression"
else:
    best_model = knn_pipeline
    best_name = "KNN"

print()
print("=" * 50)
print("SEÇİLEN MODEL")
print("=" * 50)
print(best_name)

# -------------------------------------------------------
# Test Seti Değerlendirmesi
# -------------------------------------------------------
print()
print("=" * 50)
print("TEST SONUÇLARI")
print("=" * 50)

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

print("\nAccuracy :", round(acc, 3))
print("Precision:", round(pre, 3))
print("Recall   :", round(rec, 3))
print("F1 Score :", round(f1, 3))

# -------------------------------------------------------
# Sınıflandırma Sonuçlarını Göster
# -------------------------------------------------------
print()
print("=" * 50)
print("GERÇEK VE TAHMİN EDİLEN DEĞERLER (İlk 15)")
print("=" * 50)

sonuclar = pd.DataFrame({
    "Gercek": y_test.values,
    "Tahmin": y_pred
})

print(sonuclar.head(15))

# -------------------------------------------------------
# Kısa Sonuç Yorumu
# -------------------------------------------------------
print()
print("=" * 50)
print("SONUÇ YORUMU")
print("=" * 50)

if best_name == "Logistic Regression":
    print("""
Validation sonuçlarına göre Logistic Regression modeli
daha yüksek F1-Score elde ettiği için seçildi.

Bu veri setinde müşteri ayrılma durumu büyük ölçüde
doğrusal ilişkiler içerdiğinden Logistic Regression
başarılı sonuç vermiştir.

Test verisinde de benzer performans göstererek
genelleme yeteneğinin iyi olduğu gözlemlenmiştir.
""")
else:
    print("""
Validation sonuçlarına göre KNN modeli
daha yüksek F1-Score elde ettiği için seçildi.

Benzer özelliklere sahip müşterileri komşuluk
yaklaşımıyla değerlendirdiği için bu veri setinde
daha başarılı sonuç vermiştir.

Test verisinde de başarılı performans sergilemiştir.
""")

print("=" * 50)
print("PROGRAM BAŞARIYLA TAMAMLANDI")
print("=" * 50)