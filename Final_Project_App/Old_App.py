import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import dlib
from joblib import load
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import resnet34
import av
import time
import sys
import asyncio

# ─── macOS EVENT LOOP FIX ───────────────────────────────────────────────
if sys.platform.startswith("darwin"):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

# ─── DEVICE SETUP ────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ─── STREAMLIT TITLE ─────────────────────────────────────────────────────
st.title("📸 Real-Time BMI Prediction")
st.write("Facial landmarks and BMI estimation shown in real time.")

# ─── VGGFACE2 MODEL ──────────────────────────────────────────────────────
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=torch.device("cpu"))
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ─── FAIRFACE MODEL ──────────────────────────────────────────────────────
fair_model = resnet34(weights=None)
fair_model.fc = torch.nn.Linear(fair_model.fc.in_features, 18)
fair_model.load_state_dict(torch.load("FairFace/fair_face_models/res34_fair_align_multi_4_20190809.pt", map_location=DEVICE))
fair_model = fair_model.to(DEVICE).eval()

# ─── BEST Trained MODEL - Takes input from FairFace Model / Landmark Features / Embeddings ───────────────────────────────────────────────────────────
xgb_model = load("saved_models/best_bmi_model_xgboost.joblib")

# ─── LANDMARK DETECTION ────────────────────────────────────────────────────
predictor = dlib.shape_predictor("dlib_models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
gender_labels = ['Male', 'Female']
age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

### Landmark Features
def get_landmark_features(img, shape):
    points = np.array([(p.x, p.y) for p in shape.parts()])
    def dist(p1, p2): return np.linalg.norm(points[p1] - points[p2])
    return [
        dist(0, 16), dist(8, 27), dist(1, 15), dist(31, 35),
        dist(48, 54), dist(51, 57), dist(39, 42),
        dist(36, 39), dist(42, 45),
        dist(0, 16) / dist(8, 27) if dist(8, 27) != 0 else 0,
        dist(48, 54) / dist(31, 35) if dist(31, 35) != 0 else 0
    ]

class FaceAndBMIPredictor(VideoTransformerBase):
    def __init__(self):
        self.last_prediction = None
        self.last_timestamp = 0
        self.prediction_interval = 5  # seconds

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        boxes, _ = mtcnn.detect(pil_img)
        if boxes is None or len(boxes) == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        x1, y1, x2, y2 = boxes[0].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face_crop = pil_img.crop((x1, y1, x2, y2))
        face_tensor = inference_transform(face_crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            fair_out = fair_model(face_tensor).cpu().numpy().squeeze()
            embedding = embedding_model(face_tensor)

        race_idx = np.argmax(fair_out[:7])
        gender_idx = np.argmax(fair_out[7:9])
        age_idx = np.argmax(fair_out[9:18])

        race_keep = [1, 2, 3, 0]
        age_keep = [2, 3, 1, 4, 5, 6]

        race_onehot = np.zeros(4)
        if race_idx in race_keep:
            race_onehot[race_keep.index(race_idx)] = 1

        gender_onehot = np.zeros(2)
        gender_onehot[gender_idx] = 1

        age_onehot = np.zeros(6)
        if age_idx in age_keep:
            age_onehot[age_keep.index(age_idx)] = 1

        img_rgb = np.array(pil_img)
        dets = detector(img_rgb, 1)
        if len(dets) == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        shape = predictor(img_rgb, dets[0])
        landmark_features = get_landmark_features(img_rgb, shape)

        for pt in shape.parts():
            cv2.circle(img, (pt.x, pt.y), 1, (0, 255, 255), -1)

        cv2.putText(img, "Facial Landmarks Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        full_features = np.concatenate([
            embedding.cpu().numpy().squeeze(),
            landmark_features,
            race_onehot,
            gender_onehot,
            age_onehot
        ]).reshape(1, -1)

        # Only update prediction every 5 seconds
        current_time = time.time()
        if current_time - self.last_timestamp > self.prediction_interval:
            self.last_prediction = xgb_model.predict(full_features)[0]
            self.last_timestamp = current_time

        race_label = race_labels[race_idx]
        gender_label = gender_labels[gender_idx]
        age_label = age_labels[age_idx]

        # ─── DISPLAY RESULTS ──────────────────────────────────────────────────
        if self.last_prediction is not None:
            cv2.putText(img, f"BMI: {self.last_prediction:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(img, f"Race: {race_label}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"Gender: {gender_label}", (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"Age: {age_label}", (x1, y2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─── STREAMLIT APP ───────────────────────────────────────────────────────
webrtc_streamer(
    key="bmi-live-interval",
    video_processor_factory=FaceAndBMIPredictor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
