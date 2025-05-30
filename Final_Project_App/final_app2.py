import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from joblib import load
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models import resnet34
import av
import time
import sys
import asyncio
import mediapipe as mp

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

mode = st.radio("Select input mode:", ["Webcam", "Upload Image"])

# ─── MODELS ──────────────────────────────────────────────────────────────
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=torch.device("cpu"))
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

fair_model = resnet34(weights=None)
fair_model.fc = torch.nn.Linear(fair_model.fc.in_features, 18)
fair_model.load_state_dict(torch.load("FairFace/fair_face_models/res34_fair_align_multi_4_20190809.pt", map_location=DEVICE))
fair_model = fair_model.to(DEVICE).eval()

xgb_model = load("saved_models/best_bmi_model_xgboost.joblib")

# ─── LABELS AND TRANSFORMS ───────────────────────────────────────────────
race_labels = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
gender_labels = ['Male', 'Female']
age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
race_keep = [1, 2, 3, 0]
age_keep = [2, 3, 1, 4, 5, 6]

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_landmarks(image_np):
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    results = mp_face.process(image_np)
    mp_face.close()
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0]
    x_coords = [int(lm.x * image_np.shape[1]) for lm in face_landmarks.landmark]
    y_coords = [int(lm.y * image_np.shape[0]) for lm in face_landmarks.landmark]
    return x_coords, y_coords

def process_image(pil_img):
    img_rgb = np.array(pil_img)
    coords = extract_landmarks(img_rgb)
    if coords is None:
        return None, "No face or landmarks detected.", None
    x_coords, y_coords = coords
    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    face_crop = pil_img.crop((x1, y1, x2, y2))
    face_tensor = inference_transform(face_crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fair_out = fair_model(face_tensor).cpu().numpy().squeeze()
        embedding = embedding_model(face_tensor)

    race_idx = np.argmax(fair_out[:7])
    gender_idx = np.argmax(fair_out[7:9])
    age_idx = np.argmax(fair_out[9:18])

    race_onehot = np.zeros(4)
    if race_idx in race_keep:
        race_onehot[race_keep.index(race_idx)] = 1

    gender_onehot = np.zeros(2)
    gender_onehot[gender_idx] = 1

    age_onehot = np.zeros(6)
    if age_idx in age_keep:
        age_onehot[age_keep.index(age_idx)] = 1

    landmark_features = np.array(x_coords[:5] + y_coords[:5])

    full_features = np.concatenate([
        embedding.cpu().numpy().squeeze(),
        landmark_features,
        race_onehot,
        gender_onehot,
        age_onehot
    ]).reshape(1, -1)

    bmi = xgb_model.predict(full_features)[0]

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for x, y in zip(x_coords, y_coords):
        cv2.circle(img_bgr, (x, y), 1, (0, 255, 255), -1)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    result_text = f"BMI: {bmi:.2f}"
    annotated_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return bmi, result_text, annotated_img

if mode == "Upload Image":
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Original uploaded image", use_column_width=True)
        bmi, result_text, annotated_img = process_image(image)
        if bmi is None:
            st.warning(result_text)
        else:
            st.image(annotated_img, caption="Detected face and landmarks", use_column_width=True)
            st.success(result_text)

else:
    class FaceAndBMIPredictor(VideoTransformerBase):
        def __init__(self):
            self.last_prediction = None
            self.last_timestamp = 0
            self.prediction_interval = 5
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if not results.multi_face_landmarks:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            face_landmarks = results.multi_face_landmarks[0]
            x_coords = [int(lm.x * img.shape[1]) for lm in face_landmarks.landmark]
            y_coords = [int(lm.y * img.shape[0]) for lm in face_landmarks.landmark]
            x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

            pil_img = Image.fromarray(img_rgb)
            face_crop = pil_img.crop((x1, y1, x2, y2))
            face_tensor = inference_transform(face_crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                fair_out = fair_model(face_tensor).cpu().numpy().squeeze()
                embedding = embedding_model(face_tensor)

            race_idx = np.argmax(fair_out[:7])
            gender_idx = np.argmax(fair_out[7:9])
            age_idx = np.argmax(fair_out[9:18])

            race_onehot = np.zeros(4)
            if race_idx in race_keep:
                race_onehot[race_keep.index(race_idx)] = 1

            gender_onehot = np.zeros(2)
            gender_onehot[gender_idx] = 1

            age_onehot = np.zeros(6)
            if age_idx in age_keep:
                age_onehot[age_keep.index(age_idx)] = 1

            landmark_features = np.array(x_coords[:5] + y_coords[:5])

            current_time = time.time()
            if current_time - self.last_timestamp > self.prediction_interval:
                full_features = np.concatenate([
                    embedding.cpu().numpy().squeeze(),
                    landmark_features,
                    race_onehot,
                    gender_onehot,
                    age_onehot
                ]).reshape(1, -1)
                self.last_prediction = xgb_model.predict(full_features)[0]
                self.last_timestamp = current_time

            if self.last_prediction is not None:
                cv2.putText(img, f"BMI: {self.last_prediction:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            for x, y in zip(x_coords, y_coords):
                cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="bmi-live-interval",
        video_processor_factory=FaceAndBMIPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )