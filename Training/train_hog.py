import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# =====================================================================
# 0. SETUP PERANGKAT & KONFIGURASI PATH
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat komputasi: {device}")

PARQUET_FILE = 'DAiSEE_Raw_Features_HOG.parquet'
LABELS_CSV = 'DAiSEE/Labels/TrainLabels.csv' # Pastikan path ini benar!
MODEL_OUTPUT = 'Engagement_BiLSTM_HOG.pth'
SCALER_OUTPUT = 'Feature_Scaler_HOG.pkl'

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 20

# =====================================================================
# 1. DATA LOADING & LABEL MERGING
# =====================================================================
CSV_FILE = 'all_data50.csv' # Ubah dengan nama file CSV dari rekan Anda
LABELS_CSV = 'DAiSEE/Labels/TrainLabels.csv'
SEQUENCE_LENGTH = 30
NUM_FEATURES =  70

print("Memuat data CSV...")
df = pd.read_csv(CSV_FILE)

# Konversi kolom 'features' dari tipe string ke tipe list Python menggunakan ast.literal_eval
print("Mengonversi string array ke list numerik...")
df['features'] = df['features'].apply(ast.literal_eval)

# Asumsi: Kolom 'id' di CSV rekan Anda adalah ClipID DAiSEE (misal: 1100011002)
# Kita pastikan id bertipe string (tambahkan .avi jika perlu) agar bisa di-join dengan TrainLabels.csv
df['ClipID'] = df['id'].astype(str) + '.avi'

# Memuat label asli DAiSEE
labels_df = pd.read_csv(LABELS_CSV)

# Gabungkan data fitur dengan label
df = df.merge(labels_df[['ClipID', 'Engagement']], on='ClipID', how='inner')
df = df.rename(columns={'Engagement': 'engagement_label'})

# Urutkan berdasarkan ID lalu berdasarkan Timestamp agar urutan waktunya benar
df = df.sort_values(by=['id', 'timestamp'])

grouped = df.groupby('id')

X_list = []
y_list = []
print("Membentuk sekuens waktu (Temporal Sequencing)...")
for video_name, group_data in grouped:
    # Ubah list of lists di kolom 'features' menjadi 2D Numpy Array
    features = np.array(group_data['features'].tolist())
    label = group_data['engagement_label'].iloc[0]

    # PAD atau TRUNCATE menjadi 30 Frame
    if len(features) >= SEQUENCE_LENGTH:
        features = features[:SEQUENCE_LENGTH]
    else:
        pad_length = SEQUENCE_LENGTH - len(features)
        # Duplikasi frame terakhir jika video terlalu pendek
        pad_array = np.tile(features[-1], (pad_length, 1))
        features = np.vstack((features, pad_array))

    X_list.append(features)
    y_list.append(label)

X = np.array(X_list)
y = np.array(y_list, dtype=np.int64)

print(f"Total Video Tersedia: {len(X)} | Bentuk Matriks: {X.shape}")

# =====================================================================
# 2. SCALING & DATA SPLIT
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

print("Melakukan Standard Scaling (Wajib untuk PCA)...")
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, NUM_FEATURES)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape[0], SEQUENCE_LENGTH, NUM_FEATURES)

X_test_flat = X_test.reshape(-1, NUM_FEATURES)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], SEQUENCE_LENGTH, NUM_FEATURES)

joblib.dump(scaler, SCALER_OUTPUT)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test)), batch_size=BATCH_SIZE, shuffle=False)

# =====================================================================
# 3. ARSITEKTUR MODEL (Sama Persis dengan train.py Asli)
# =====================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))
    def forward(self, lstm_outputs):
        attn_weights = torch.softmax(self.attention(lstm_outputs), dim=1)
        return torch.sum(attn_weights * lstm_outputs, dim=1), attn_weights

class EngagementBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EngagementBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        context, _ = self.attention(out)
        out = self.dropout(self.relu(self.fc1(context)))
        return self.fc2(out)

model = EngagementBiLSTM(input_dim=NUM_FEATURES, hidden_dim=64, num_classes=4).to(device)

# =====================================================================
# 4. PEMBOBOTAN KELAS & OPTIMIZER
# =====================================================================
classes_unique = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train)
class_weights = np.clip(class_weights, a_min=None, a_max=3.0)
tensor_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=tensor_weights, label_smoothing=0.15)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# =====================================================================
# 5. LOOP PELATIHAN (Dengan Early Stopping)
# =====================================================================
print("\n--- Memulai Pelatihan Model HOG-PCA ---")
best_val_loss = float('inf')
epochs_no_improve = 0 # TAMBAHKAN Variabel Counter Ini
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    # --- TRAINING ---
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = correct_train / total_train

    # --- VALIDATION ---
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = correct_val / total_val

    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss / len(test_loader))
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Train Loss: {history['train_loss'][-1]:.4f} - Acc: {train_acc*100:.2f}% || "
          f"Val Loss: {history['val_loss'][-1]:.4f} - Acc: {val_acc*100:.2f}%")

    # LOGIKA EARLY STOPPING & CHECKPOINTING
    if history['val_loss'][-1] < best_val_loss:
        best_val_loss = history['val_loss'][-1]
        torch.save(model.state_dict(), MODEL_OUTPUT)
        epochs_no_improve = 0 # Reset counter jika model membaik
    else:
        epochs_no_improve += 1
        print(f"  -> Tidak ada perbaikan Val Loss ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print(f"\n[!] EARLY STOPPING DIMULAI: Model berhenti belajar di Epoch {epoch+1}.")
            break # Hentikan proses pelatihan (keluar dari loop for)

print(f"\nPelatihan Selesai! Model terbaik disimpan sebagai: {MODEL_OUTPUT}")

# =====================================================================
# 6. VISUALISASI HASIL & MATRIKS (Identik dengan train.py)
# =====================================================================
plt.figure(figsize=(12, 5))


# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Akurasi Pelatihan')
plt.plot(history['val_acc'], label='Akurasi Validasi')
plt.title('Grafik Akurasi')
plt.ylabel('Akurasi (Accuracy)')
plt.xlabel('Iterasi (Epoch)')
plt.legend(loc='lower right')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Kesalahan Pelatihan')
plt.plot(history['val_loss'], label='Kesalahan Validasi')
plt.title('Grafik Kesalahan')
plt.ylabel('Kesalahan (Loss)')
plt.xlabel('Iterasi (Epoch)')
plt.legend(loc='upper right')

plt.tight_layout()

# Run Inference untuk Confusion Matrix
# (Memuat model terbaik yang baru saja disimpan)
model.load_state_dict(torch.load(MODEL_OUTPUT, weights_only=True))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Karena di skrip ini kita tidak memakai `LabelEncoder` (label sudah angka 0-3 dari CSV),
# kita definisikan variabel `classes_` secara manual agar sesuai dengan format asli Anda.
classes_ = ['0', '1', '2', '3']

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes_, yticklabels=classes_)
plt.title('Matriks Confusion (Data Pengujian)')
plt.ylabel('Kategori Asli')
plt.xlabel('Prediksi Model')

# Classification Report
report = classification_report(all_labels, all_preds, target_names=classes_)
print("Laporan Klasifikasi Data Pengujian:\n")
print(report)

plt.show()
