import os
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import csv
import json
import threading
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import joblib

import grpc
from concurrent import futures
import data_pb2
import data_pb2_grpc

# =====================================================================
# 1. PYTORCH MODEL ARCHITECTURE
# =====================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
    def forward(self, lstm_outputs):
        attention_weights = torch.softmax(self.attention(lstm_outputs), dim=1)
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)
        return context_vector, attention_weights

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
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        context, _ = self.attention(lstm_out)
        out = self.dropout(self.relu(self.fc1(context)))
        return self.fc2(out)

# =====================================================================
# 2. INITIALIZATION
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sistem Server HOG-PCA berjalan pada: {device}")

NUM_FEATURES = 70
model = EngagementBiLSTM(input_dim=NUM_FEATURES, hidden_dim=64, num_classes=4).to(device)

try:
    model.load_state_dict(torch.load('Engagement_BiLSTM_HOG.pth', map_location=device, weights_only=True))
    model.eval()
    scaler = joblib.load('Feature_Scaler_HOG.pkl')
    print("✅ Model HOG-PCA dan Scaler berhasil dimuat.")
except Exception as e:
    print(f"❌ Error memuat model/scaler: {e}")

class_names = {
    0: "Sangat Rendah (Very Low)", 
    1: "Rendah (Low)", 
    2: "Tinggi (High)", 
    3: "Sangat Tinggi (Very High)"
}

feature_buffer = deque(maxlen=30)
current_prediction = "Mengkalkulasi Jaringan..."
confidence_str = "Confidence: N/A"

def ui_text(cv2_im, text, position, font_size, color=(255, 255, 255)):
    pil_im = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    try: 
        font = ImageFont.truetype("arial.ttf", font_size)
    except: 
        font = ImageFont.load_default()
    b, g, r = color
    draw.text(position, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# =====================================================================
# 3. BIDIRECTIONAL GRPC SERVICER
# =====================================================================
class DataStreamServicer(data_pb2_grpc.DataStreamServiceServicer):
    def __init__(self):
        self.csv_path = "all_data_hog.csv"
        self.init_csv()

    def init_csv(self):
        file_exists = os.path.isfile(self.csv_path)
        if not file_exists:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "timestamp", "latency_ms", "features", "prediction", "confidence"])

    def StreamFeatures(self, request_iterator, context):
        global current_prediction, confidence_str, feature_buffer
        
        # Continuously read streaming items sent by the client
        for request in request_iterator:
            receive_time = time.time_ns()
            latency_ms = (receive_time - request.timestamp) / 1_000_000.0
            raw_data = list(request.features)

            probs_list = []
            response_prediction = "Mengumpulkan Buffer..."

            if len(raw_data) >= NUM_FEATURES:
                model_features = raw_data[:NUM_FEATURES]

                feature_buffer.append(model_features)
                
                if len(feature_buffer) == 30:
                    spatial_seq = np.array(feature_buffer)
                    scaled_seq = scaler.transform(spatial_seq).reshape(1, 30, NUM_FEATURES)
                    input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_idx = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_idx].item() * 100
                        
                        probs_list = probabilities[0].tolist()

                    current_prediction = class_names[predicted_idx]
                    confidence_str = f"Confidence: {confidence:.1f}%"
                    response_prediction = current_prediction

                # Save metrics to local server log file
                features_str = json.dumps(model_features)
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([request.id, request.timestamp, latency_ms, features_str, current_prediction, confidence_str])
            
            # YIELD the response back instantly. If the buffer wasn't full, 
            # probs_list stays empty, signaling to the client that it's still buffering.
            yield data_pb2.FeatureResponse(
                success=True,
                message="Frame Processed",
                probabilities=probs_list,
                predicted_class=response_prediction
            )

def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_pb2_grpc.add_DataStreamServiceServicer_to_server(DataStreamServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("🚀 Bidirectional gRPC Server running on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    grpc_thread = threading.Thread(target=serve_grpc, daemon=True)
    grpc_thread.start()

    print("🖥️ Starting Server UI Dashboard...")
    while True:
        dashboard = np.ones((600, 800, 3), dtype=np.uint8) * 40
        dashboard = ui_text(dashboard, "Central Server: HOG-PCA (19 Features)", (30, 20), 28, (255, 200, 0))
        buf_len = len(feature_buffer)
        status_color = (0, 255, 255) if buf_len < 30 else (0, 255, 0)
        dashboard = ui_text(dashboard, f"STATUS: Buffer Aktif {buf_len}/30 Frames", (30, 70), 20, status_color)
        pred_color = (0, 255, 0) if buf_len == 30 else (200, 200, 200)
        dashboard = ui_text(dashboard, f"Prediksi: {current_prediction}", (30, 130), 26, pred_color)
        dashboard = ui_text(dashboard, confidence_str, (30, 170), 19, (200, 200, 200))

        dashboard = ui_text(dashboard, "SERVER METRICS:", (500, 70), 16, (255, 255, 0))
        dashboard = ui_text(dashboard, "Model: Bi-LSTM HOG-PCA", (500, 95), 14, (200, 200, 200))
        dashboard = ui_text(dashboard, "Port: [::]:50051", (500, 120), 14, (200, 200, 200))
        
        cv2.imshow('Server Administrator Dashboard', dashboard)
        if cv2.waitKey(100) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()
