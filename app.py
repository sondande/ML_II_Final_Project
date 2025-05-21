import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN  
from PIL import Image
import time
from torchvision import transforms
from bmi_multimodal_training import BMIModel 
import torch

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
 
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=DEVICE)
mtcnn_detect = MTCNN(image_size=160, margin=20, keep_all=False, device="cpu")

model = BMIModel().to(DEVICE)
checkpoint = torch.load("saved_models/best_bmi_multimodal_model.pth", map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

normalize = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("BMI Prediction via Webcam")
st.write("Let's detect your BMI based off your face in real-time.")

mode = st.radio("Select input mode:", ["Webcam", "Upload Image"])
st.write("📋 Current mode:", mode)

gender_str = st.selectbox("Gender", ["Male", "Female"])
gender_idx = 0 if gender_str == "Male" else 1

def predict_bmi_from_image(pil_img: Image.Image, gender_idx: int) -> float:
    boxes, _ = mtcnn_detect.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return None
    

    x1, y1, x2, y2 = boxes[0].astype(int)
    face = pil_img.crop((x1, y1, x2, y2))
    tensor = inference_transform(face).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        bmi_pred, _ = model(tensor, torch.tensor([gender_idx], device=DEVICE))
    return bmi_pred.item()

if mode == "Upload Image":

    uploaded = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        # st.image(image, caption="Uploaded image", use_column_width=True)
        bmi = predict_bmi_from_image(image, gender_idx)
        if bmi is None:
            st.warning("No face detected. Please try another image.")
        else:
            st.success(f"Predicted BMI: {bmi:.2f}")
else:
    class FaceAndBMIPredictor(VideoTransformerBase):
        def __init__(self, model, inference_transform, gender_idx):
            self.model = model
            self.inference_transform = inference_transform
            self.gender_idx = gender_idx
            self.last_prediction = None
            self.last_timestamp = 0
            self.prediction_interval = 2

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes, _ = mtcnn_detect.detect(pil_img)

            if boxes is None or len(boxes) == 0:
               cv2.putText(img, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
               return img

            x1, y1, x2, y2 = boxes[0].astype(int)
            h, w = pil_img.size[1], pil_img.size[0]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_crop = pil_img.crop((x1, y1, x2, y2))
            face_tensor = self.inference_transform(face_crop).unsqueeze(0).to(DEVICE)

            current_time = time.time()
            if current_time - self.last_timestamp > self.prediction_interval:
                with torch.no_grad():
                    bmi_pred, _ = self.model(face_tensor, 
                                         torch.tensor([self.gender_idx], device=DEVICE))
                self.last_prediction = bmi_pred.item()
                self.last_timestamp = current_time
            predicted_bmi = self.last_prediction

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Face Detected", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(img, f"BMI: {predicted_bmi:.2f}", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            return img

webrtc_streamer(
    key="bmi-webcam",
    video_processor_factory=lambda: FaceAndBMIPredictor(model, inference_transform, gender_idx),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)