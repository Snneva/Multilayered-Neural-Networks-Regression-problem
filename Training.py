import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Membaca dataset
csv_file = 'dataset_jarak.csv'
try:
    df = pd.read_csv(csv_file)
    print(f"Dataset berhasil dimuat! Total sampel: {len(df)}")
except FileNotFoundError:
    print(f"File '{csv_file}' tidak ditemukan. Jalankan Live_Dataset.py terlebih dahulu!")
    exit()

# Cek kolom target dan fitur
target_col = 'Distance_cm'
y = df[target_col].values

# 2. Definisikan kombinasi input sesuai instruksi dosen
input_experiments = {
    'Model_Face_Only': ['F'],
    'Model_Eyes_Only': ['LE', 'RE'],
    'Model_All_Features': ['LE', 'RE', 'F']
}

results = []
best_rmse = float('inf')
best_model_info = None

print("\n" + "="*60)
print("           HASIL PELATIHAN NEURAL NETWORK (MLP)           ")
print("="*60)

for name, features in input_experiments.items():
    X = df[features].values

    # Split dataset menjadi Data Latih (80%) dan Data Uji (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalisasi data (Neural Network sangat membutuhkan scaling karena pixel bernilai ratusan/ribuan)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Multilayer Neural Network (Backpropagation)
    # Struktur: 2 hidden layers (64 neuron, 32 neuron), aktivasi ReLU, optimizer Adam
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1500,
        random_state=42
    )

    # Latih model
    model.fit(X_train_scaled, y_train)

    # Prediksi data uji
    y_pred = model.predict(X_test_scaled)

    # Hitung metrik evaluasi
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n[+] {name}")
    print(f"    Fitur Input : {features}")
    print(f"    RMSE        : {rmse:.3f} cm")
    print(f"    R^2 Score   : {r2:.3f}")

    # Simpan model dengan performa terbaik
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_info = {
            'name': name,
            'features': features,
            'model': model,
            'scaler': scaler,
            'rmse': rmse
        }

# 4. Simpan model dan scaler terbaik untuk Tahap 3 (Live Demo)
print("\n" + "="*60)
print(f"Model Terbaik : {best_model_info['name']} (RMSE: {best_model_info['rmse']:.3f} cm)")
joblib.dump(best_model_info['model'], 'best_distance_model.pkl')
joblib.dump(best_model_info['scaler'], 'best_scaler.pkl')
joblib.dump(best_model_info['features'], 'model_features.pkl')
print("Model dan Scaler berhasil disimpan ke:")
print(" - 'best_distance_model.pkl'")
print(" - 'best_scaler.pkl'")
print(" - 'model_features.pkl'")
print("="*60)