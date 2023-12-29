# Impor pustaka
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Membuat data sintetis sebagai contoh
np.random.seed(42)
# Fitur: Usia, Tekanan Darah, Kolesterol
X = np.random.rand(100, 3) * 100
# Label: 0 (Tidak Ada Penyakit), 1 (Penyakit)
y = (X[:, 0] + X[:, 1] + X[:, 2] > 150).astype(int)

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Lakukan prediksi pada data uji
y_pred = dt_model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy:.2f}')

# Visualisasi hasil prediksi
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', label='Kelas Sebenarnya')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.coolwarm, marker='x', s=80, label='Kelas Prediksi')
plt.title('Hasil Klasifikasi menggunakan Decision Tree')
plt.xlabel('Usia')
plt.ylabel('Tekanan Darah')
plt.legend()
plt.show()
