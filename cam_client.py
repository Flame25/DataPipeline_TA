import sys
import os
import cv2
import time
import torch
import numpy as np
import threading
import queue

import mediapipe as mp
from skimage.feature import hog
from sklearn.decomposition import PCA

# ================= PATH FIX =================
BASE_DIR = os.getcwd()
sys.path.append(os.path.join(BASE_DIR, "DI-Retinex"))

import mymodel  # DI-Retinex
from insightface.app import FaceAnalysis
from openface.multitask_model import MultitaskPredictor

# Import generated gRPC files
import data_pb2
import data_pb2_grpc
import grpc

# ================= CONFIG =================
SERVER = "localhost:50051"
ALPHA = 0.4
PCA_COMPONENTS = 70
SKIP_FRAMES = 3 
patch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= gRPC BIDIRECTIONAL SETUP =================
grpc_request_queue = queue.Queue(maxsize=30) 

latest_prediction = "Mengkalkulasi..."
latest_probabilities = [0.0, 0.0, 0.0, 0.0]

def request_generator():
    """Continuously yields frames from the queue to the gRPC server tunnel"""
    while True:
        item = grpc_request_queue.get()
        if item is None:
            break
        yield item

def prediction_listener(response_stream):
    """Continuously listens for predictions sent back from the server"""
    global latest_prediction, latest_probabilities
    try:
        for response in response_stream:
            if len(response.probabilities) > 0:
                latest_prediction = response.predicted_class
                latest_probabilities = list(response.probabilities)
            else:
                latest_prediction = response.predicted_class
    except grpc.RpcError as e:
        print(f"gRPC Stream Disconnected: {e}")

def start_grpc_stream():
    """Initializes the bidirectional stream"""
    channel = grpc.insecure_channel(SERVER)
    stub = data_pb2_grpc.DataStreamServiceStub(channel)
    
    response_stream = stub.StreamFeatures(request_generator())
    threading.Thread(target=prediction_listener, args=(response_stream,), daemon=True).start()
    print("✅ gRPC Bidirectional Stream Started")

# ================= TRACKING (EMA SMOOTHING) =================
class SmoothTracker:
    def __init__(self, alpha=0.5):
        self.bbox = None
        self.alpha = alpha 

    def update(self, new_bbox):
        if self.bbox is None:
            self.bbox = np.array(new_bbox, dtype=np.float32)
        else:
            self.bbox = (np.array(new_bbox, dtype=np.float32) * self.alpha) + \
                        (self.bbox * (1.0 - self.alpha))
        return self.bbox.astype(int)

face_tracker = SmoothTracker(alpha=0.5)

# ================= INITIALIZE MODELS =================
print("Initializing Models...")

# 1. DI-Retinex
net = mymodel.enhance_net_nopool(6, 64).to(device)
net.load_state_dict(torch.load("./DI-Retinex/weights/model.pth", map_location=device, weights_only=True))
net.eval()

# 2. InsightFace
app_face = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app_face.prepare(ctx_id=0, det_size=(320, 320))

# 3. OpenFace
model = MultitaskPredictor(model_path="./OpenFace/weights/MTL_backbone.pth", device=device)

# 4. MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ================= PSFP & PCA SETUP =================
MASTER_SFP = {
    "P1": 61, "P4": 291, "P9": 194, "P11": 418, "P10": 200, "P20": 164, 
    "P16": 168, "P17": 9, "P14": 118, "P15": 347, "P18": 130, "P19": 263, 
    "P3": 120, "P6": 349, "P2": 50, "P7": 123, "P8": 147, "P5": 280, "P12": 352, "P13": 376 
}

SFP_FRONTAL = {k: MASTER_SFP[k] for k in MASTER_SFP.keys()}
SFP_LEFT = {k: MASTER_SFP[k] for k in ["P1", "P2", "P3", "P7", "P8", "P9", "P10", "P14", "P16", "P17", "P18", "P20"]}
SFP_RIGHT = {k: MASTER_SFP[k] for k in ["P4", "P5", "P6", "P10", "P11", "P12", "P13", "P15", "P16", "P17", "P19", "P20"]}

face_3d_points = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

pca_frontal = PCA(n_components=PCA_COMPONENTS)
pca_profile = PCA(n_components=PCA_COMPONENTS)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dummy_data_frontal = np.zeros((PCA_COMPONENTS + 1, 5760)) 
    pca_frontal.fit(dummy_data_frontal)
    dummy_data_profile = np.zeros((PCA_COMPONENTS + 1, 3456))
    pca_profile.fit(dummy_data_profile)

print("Models & PCA Ready.")

def get_active_patches(yaw_angle):
    if -22.5 <= yaw_angle <= 22.5: return SFP_FRONTAL, "FRONTAL"
    elif 22.5 < yaw_angle <= 67.5: return SFP_RIGHT, "RIGHT_45"
    elif yaw_angle > 67.5: return SFP_RIGHT, "RIGHT_90"
    elif -67.5 <= yaw_angle < -22.5: return SFP_LEFT, "LEFT_45"
    else: return SFP_LEFT, "LEFT_90"

def extract_psfp_patch(image, center_x, center_y, M=32, N=32):
    h, w, _ = image.shape
    start_x = max(0, int(center_x - (M / 2) + 1))
    start_y = max(0, int(center_y - (N / 2) + 1))
    end_x = min(w, int(center_x + (M / 2)))
    end_y = min(h, int(center_y + (N / 2)))
    
    patch = image[start_y:end_y, start_x:end_x]
    
    if patch.shape[0] != N or patch.shape[1] != M:
        padded = np.zeros((N, M, 3), dtype=np.uint8)
        padded[0:patch.shape[0], 0:patch.shape[1]] = patch
        return padded
    return patch

def extract_hog_features(patch):
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # CHANGED: visualize=True to get the HOG image back
    features, hog_image = hog(
        gray_patch, orientations=8, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), block_norm='L2-Hys',
        visualize=True, feature_vector=True
    )
    return features, hog_image

# ================= RUNNER (WEBCAM LOOP) =================
if __name__ == "__main__":
    
    # 1. Initialize Bidirectional Streaming
    start_grpc_stream()
    
    # 2. Buka Webcam (0 biasanya adalah webcam default laptop/PC)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        sys.exit()

    print("\n" + "="*50)
    print(" WEBCAM MONITORING WINDOW ACTIVE")
    print(" Press 'q' inside a monitoring window to terminate.")
    print("="*50 + "\n")

    # State Variables
    frame_count = 0
    has_face = False
    last_bbox = (0, 0, 0, 0)
    last_emotion = 0
    last_emotion_array = np.array([]) 
    last_gaze_center = (0, 0)
    last_gaze_end = (0, 0)
    last_gaze_vector = [0.0, 0.0]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame dari webcam.")
            break

        frame_count += 1
        start = time.perf_counter()

        # 1. DI-RETINEX
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced, _, _, _ = net(tensor)

        enhanced = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

        result = cv2.addWeighted(frame, 1 - ALPHA, enhanced, ALPHA, 0)
        img_h, img_w, _ = result.shape

        # 2. FRAME SKIPPING / OPENFACE
        if frame_count % SKIP_FRAMES == 1 or not has_face:
            faces = app_face.get(result)

            if len(faces) > 0:
                has_face = True
                f = faces[0]
                
                x1, y1, x2, y2 = face_tracker.update(f.bbox.astype(int))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                last_bbox = (x1, y1, x2, y2)
                
                face_crop = result[y1:y2, x1:x2]

                if face_crop.size > 0:
                    emotion_logits, gaze_output, au_output = model.predict(face_crop)
                    if hasattr(emotion_logits, "detach"): 
                        emotion_logits = emotion_logits.detach().cpu().numpy()
                    
                    last_emotion_array = emotion_logits.flatten()
                    last_emotion = int(np.argmax(last_emotion_array))

                    gaze = gaze_output.detach().cpu().numpy().reshape(-1)
                    gaze_left, gaze_right = float(gaze[0]), float(gaze[1])
                    
                    last_gaze_vector = [gaze_left, gaze_right]
                    
                    last_gaze_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    last_gaze_end = (int(last_gaze_center[0] + gaze_left * 150), 
                                     int(last_gaze_center[1] - gaze_right * 150))
            else:
                has_face = False

        # 3. MEDIAPIPE ROI & FEATURE EXTRACTION
        canvas = np.zeros((patch_size * 4, patch_size * 5, 3), dtype=np.uint8)
        hog_canvas = np.zeros((patch_size * 4, patch_size * 5), dtype=np.uint8)  # CHANGED: Added HOG canvas
        grpc_payload = [] 

        if has_face:
            x1, y1, x2, y2 = last_bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(result, last_gaze_center, last_gaze_end, (255, 0, 0), 2)
            cv2.putText(result, f"Emotion: {last_emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            margin_w, margin_h = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
            mx1, my1 = max(0, x1 - margin_w), max(0, y1 - margin_h)
            mx2, my2 = min(img_w, x2 + margin_w), min(img_h, y2 + margin_h)

            mp_roi_rgb = cv2.cvtColor(result[my1:my2, mx1:mx2], cv2.COLOR_BGR2RGB)
            roi_h, roi_w, _ = mp_roi_rgb.shape

            mp_results = face_mesh.process(mp_roi_rgb)
            
            if mp_results.multi_face_landmarks:
                landmarks = mp_results.multi_face_landmarks[0]
                
                def get_abs_pt(lm_idx):
                    lm = landmarks.landmark[lm_idx]
                    return int(lm.x * roi_w) + mx1, int(lm.y * roi_h) + my1
                
                face_2d_points = np.array([
                    get_abs_pt(4), get_abs_pt(152), get_abs_pt(263), 
                    get_abs_pt(33), get_abs_pt(291), get_abs_pt(61)
                ], dtype=np.float64)

                cam_matrix = np.array([ [img_w, 0, img_w / 2], [0, img_w, img_h / 2], [0, 0, 1] ])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d_points, face_2d_points, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                
                yaw = angles[1] 
                active_dict, pose_label = get_active_patches(yaw)
                cv2.putText(result, f"Pose: {pose_label}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                extracted_patches = {}
                extracted_hogs = {}  # CHANGED: Store HOG images here
                frame_features = []

                for patch_name in sorted(active_dict.keys()):
                    cx, cy = get_abs_pt(active_dict[patch_name])
                    patch = extract_psfp_patch(result, cx, cy, M=patch_size, N=patch_size)
                    extracted_patches[patch_name] = patch
                    
                    # CHANGED: Unpack both feature array and visual image
                    hog_array, hog_img = extract_hog_features(patch)
                    extracted_hogs[patch_name] = hog_img
                    frame_features.extend(hog_array)

                    cv2.rectangle(result, (cx - patch_size//2, cy - patch_size//2), 
                                 (cx + patch_size//2, cy + patch_size//2), (0, 255, 0), 1)

                # --- COMBINE PAYLOAD ---
                final_feature_vector = np.array(frame_features)
                
                if len(active_dict) == 20:
                    reduced_data = pca_frontal.transform(final_feature_vector.reshape(1, -1))[0]
                else:
                    reduced_data = pca_profile.transform(final_feature_vector.reshape(1, -1))[0]
                    
                if last_emotion_array.size > 0:
                    grpc_payload = np.concatenate((reduced_data, last_emotion_array, last_gaze_vector)).tolist()

                # Render SFP and HOG canvases
                row, col = 0, 0
                for patch_name in extracted_patches.keys():
                    # 1. RGB Patch
                    rgb_patch = extracted_patches[patch_name]
                    canvas[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = rgb_patch
                    
                    # 2. HOG Patch (CHANGED)
                    h_img = extracted_hogs[patch_name]
                    # Scale HOG float values to 0-255 uint8 for OpenCV to display
                    if np.max(h_img) > 0:
                        h_img_scaled = (h_img / np.max(h_img) * 255).astype(np.uint8)
                    else:
                        h_img_scaled = h_img.astype(np.uint8)
                        
                    hog_canvas[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = h_img_scaled

                    col += 1
                    if col >= 5: col, row = 0, row + 1

        # 4. gRPC STREAMING
        if len(grpc_payload) > 0:
            req = data_pb2.FeatureRequest(id="webcam_client", features=grpc_payload, timestamp=time.time_ns())
            try:
                grpc_request_queue.put_nowait(req)
            except queue.Full:
                pass  

        # 5. FPS & UI UPDATE
        fps = 1.0 / (time.perf_counter() - start)
        cv2.putText(result, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(result, f"Server Pred: {latest_prediction}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        if sum(latest_probabilities) > 0:
            for idx, prob in enumerate(latest_probabilities):
                cv2.putText(result, f"Class {idx}: {prob*100:.1f}%", (20, 140 + (idx*25)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Main Feed (Webcam)", result)
        if has_face:
            cv2.imshow("Active SFP Patches", canvas)
            cv2.imshow("HOG Features", hog_canvas) # CHANGED: Show the HOG canvas

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping monitor windows...")
            grpc_request_queue.put(None)
            break

    cap.release()
    cv2.destroyAllWindows()
