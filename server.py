import grpc
from concurrent import futures
import time
import numpy as np

# Import the compiled Protocol Buffer files
import data_pb2
import data_pb2_grpc

# ================= 1. DEFINE THE RECEIVER CLASS =================
class DataStreamServicer(data_pb2_grpc.DataStreamServiceServicer):
    
    def SendFeatures(self, request, context):
        receive_time = time.time_ns()
        latency_ms = (receive_time - request.timestamp) / 1_000_000.0

        raw_data = list(request.features)

        # We expect at least 12 items (10 PCA + >=1 Emotion + 2 Gaze)
        if len(raw_data) >= 12:
            
            # Slice the array back into its original parts
            pca_features = raw_data[:10]       # First 10 items
            emotion_probs = raw_data[10:-2]    # Everything from 10 up to the last 2
            gaze_vector = raw_data[-2:]        # Exactly the last 2 items

            print(f"--- New Frame (Latency: {latency_ms:.2f}ms) ---")
            print(f"PCA Features:   {np.round(pca_features, 4).tolist()}")
            
            if len(emotion_probs) > 0:
                winning_emotion = int(np.argmax(emotion_probs))
                print(f"Emotion Probs:  {np.round(emotion_probs, 4).tolist()}")
                print(f"Winning Class:  [{winning_emotion}]")
                
            print(f"Gaze (L, R):    {np.round(gaze_vector, 4).tolist()}")
            print("-" * 50)
        else:
            print(f"Warning: Received unusually short payload ({len(raw_data)} items).")

        return data_pb2.FeatureResponse(success=True, message="Data processed")


# ================= 2. START THE SERVER =================
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_pb2_grpc.add_DataStreamServiceServicer_to_server(DataStreamServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    
    print("🚀 gRPC Receiver Server is running and listening on port 50051...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server.")

if __name__ == '__main__':
    serve()
