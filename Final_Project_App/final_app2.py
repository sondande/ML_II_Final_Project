import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import mediapipe as mp
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

DEVICE = torch.device("cpu")

st.title("📸 Real-Time BMI Prediction")
st.write("Facial embeddings + landmark ratios + FairFace outputs → BMI")

mode = st.radio("Select input mode:", ["Webcam", "Upload Image"])

# Models
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=DEVICE)
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

fair_model = resnet34(weights=None)
fair_model.fc = torch.nn.Linear(fair_model.fc.in_features, 18)
fair_model.load_state_dict(torch.load("Final_Project_App/saved_models/res34_fair_align_multi_4_20190809.pt", map_location=DEVICE))
fair_model = fair_model.to(DEVICE).eval()

xgb_model = load("Final_Project_App/saved_models/best_bmi_model_xgboost.joblib")

race_keep = [1, 2, 3, 0]
age_keep = [2, 3, 1, 4, 5, 6]
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def extract_landmarks(image):
    results = mp_face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    coords = np.array([(lm.x, lm.y) for lm in landmarks])
    return coords

def compute_landmark_ratios(coords):
    if coords.shape[0] < 468:
        return [0.0] * 11
    # Example ratios: use a few landmark distances
    def dist(i, j): return np.linalg.norm(coords[i] - coords[j])
    return [
        dist(10, 152),  # face height
        dist(234, 454), # face width
        dist(33, 133),  # eye distance
        dist(61, 291),  # mouth width
        dist(0, 17),    # jawline edge
        dist(10, 338),  # diagonal
        dist(61, 291) / (dist(10, 152) + 1e-5),
        dist(234, 454) / (dist(10, 152) + 1e-5),
        dist(33, 133) / (dist(234, 454) + 1e-5),
        dist(0, 17) / (dist(10, 152) + 1e-5),
        dist(10, 338) / (dist(234, 454) + 1e-5),
    ]

def process_image(pil_img):
    image_np = np.array(pil_img)
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    landmarks = extract_landmarks(rgb_image)
    if landmarks is None:
        return None, "No landmarks detected", None

    boxes, _ = mtcnn.detect(pil_img)
    if boxes is None:
        return None, "No face detected", None
    x1, y1, x2, y2 = boxes[0].astype(int)
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

    landmark_features = compute_landmark_ratios(landmarks)

    full_features = np.concatenate([
        embedding.cpu().numpy().squeeze(),
        landmark_features,
        race_onehot,
        gender_onehot,
        age_onehot
    ]).reshape(1, -1)

    bmi = xgb_model.predict(full_features)[0]
    result_text = f"BMI: {bmi:.2f}"
    return bmi, result_text, image_np

if mode == "Upload Image":
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image")
        bmi, result_text, annotated_img = process_image(image)
        if bmi is None:
            st.warning(result_text)
        else:
            st.image(annotated_img, caption="Processed Image")
            st.success(result_text)
else:
    class FaceAndBMIPredictor(VideoTransformerBase):
        def __init__(self):
            self.last_prediction = None
            self.last_timestamp = 0
            self.prediction_interval = 5

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
                cv2.circle(img, (pt.x, pt.y), 2, (0, 255, 255), -1)

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

            #cv2.putText(img, f"Race: {race_labels[race_idx]}", (x1, y2 + 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            #cv2.putText(img, f"Gender: {gender_labels[gender_idx]}", (x1, y2 + 40),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            #cv2.putText(img, f"Age: {age_labels[age_idx]}", (x1, y2 + 60),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="bmi-live-interval",
        video_processor_factory=FaceAndBMIPredictor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={  # Add this line
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )