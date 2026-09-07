import grpc
from concurrent import futures
import time
import csv
import os
import json

import data_pb2
import data_pb2_grpc


class DataStreamServicer(data_pb2_grpc.DataStreamServiceServicer):

    # Pass the PCA configuration during server startup
    def __init__(self, pca_feature_count=60):
        self.csv_path = "training_data.csv"
        self.pca_feature_count = pca_feature_count
        self.init_csv()

    def init_csv(self):
        file_exists = os.path.isfile(self.csv_path)
        if not file_exists:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "timestamp", "latency_ms", "features"])

    def StreamFeatures(self, request_iterator, context):
        for request in request_iterator:
            receive_time = time.time_ns()
            latency_ms = (receive_time - request.timestamp) / 1_000_000.0

            raw_data = list(request.features)
            person_id = request.id

            # Extract the local configuration setting
            pca_len = self.pca_feature_count
            min_required = pca_len + 2  # PCA + at least gaze vector

            if len(raw_data) >= min_required:
                # 1. Slice PCA features up to your chosen length
                pca_features = raw_data[:pca_len]
                
                # 2. Slice emotion probabilities (everything in the middle)
                emotion_probs = raw_data[pca_len:-2]
                
                # 3. Slice the gaze vector (always the last 2 items)
                gaze_vector = raw_data[-2:]

                full_features = pca_features + emotion_probs + gaze_vector
                features_str = json.dumps(full_features)

                # Append directly to dataset CSV
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        person_id,
                        request.timestamp,
                        latency_ms,
                        features_str
                    ])

                print(f"Saved ID {person_id} | Sliced {pca_len} PCA comps | Latency {latency_ms:.2f} ms")

                # ==============================================================
                # DUMMY RESPONSE: Sent back to satisfy the client's listener
                # ==============================================================
                yield data_pb2.FeatureResponse(
                    success=True,
                    message="Saved to CSV",
                    probabilities=[0.1, 0.2, 0.6, 0.1],  # Dummy 4-class probabilities
                    predicted_class="Tinggi (High)"      # Dummy label
                )
            else:
                print(f"⚠️ Rejection: Expected at least {min_required} features for PCA configuration {pca_len}, but received {len(raw_data)}")
                yield data_pb2.FeatureResponse(
                    success=False,
                    message=f"Short payload: expected {min_required}, got {len(raw_data)}",
                    probabilities=[0.0, 0.0, 0.0, 0.0],  # Empty/Zero dummy array for error
                    predicted_class="Error"
                )


def serve():
    # ==================== CONFIGURABLE VARIABLE ====================
    # Change this number here whenever you change your client's PCA_COMPONENTS
    PCA_FEATURE_COUNT = 60  
    # ===============================================================

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_pb2_grpc.add_DataStreamServiceServicer_to_server(
        DataStreamServicer(pca_feature_count=PCA_FEATURE_COUNT), 
        server
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print(f"🚀 Server running on port 50051 configured for {PCA_FEATURE_COUNT} PCA components...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    serve()
