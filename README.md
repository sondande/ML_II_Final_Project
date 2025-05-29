# Real-Time BMI Prediction – Final Project for Machine Learning II

This project demonstrates real-time Body Mass Index (BMI) prediction using facial images or webcam input. It combines face embeddings, demographic prediction, and facial landmark geometry into a multimodal regression pipeline powered by XGBoost.

---

## Setup Instructions

### Python Version (Recommended)
**Python 3.10.17**  
Using other versions may lead to compatibility issues with some packages (e.g., `dlib`, `torch`, `streamlit_webrtc`).

### Quick Use:

Run the following
```python
sh run_app.sh
```

### Create and Activate Virtual Environment

```bash
# Create environment
python3 -m venv streamlit-bmi

# Activate environment
source streamlit-bmi/bin/activate

# Install dependencies
pip install -r requirements.txt

### run app
streamlit run Final_App.py
```

### Features
Real-time webcam BMI prediction via facial analysis.

Upload mode for predicting BMI from static images.

Combines facial embeddings, demographic predictions, and geometric features.

Uses:

MTCNN: for face detection

InceptionResnetV1 (VGGFace2): for facial embeddings

FairFace ResNet34: for race, age, and gender inference

dlib: for 68-point landmark extraction

XGBoost: for regression modeling

Facial landmarks and bounding box displayed on all detected faces.

Prediction once every 5 seconds during live streaming.


### File Descriptions 

Final_App.py = Main Streamlit app with webcam + upload support for real-time BMI prediction.

requirements.txt = Lists all necessary packages to recreate the environment.

saved_models/best_bmi_model_xgboost.joblib	 = Trained XGBoost regressor model.

FairFace/fair_face_models/res34_fair_align_multi_4_20190809.pt = FairFace ResNet34 model weights for age, race, gender classification.


dlib_models/shape_predictor_68_face_landmarks.dat = Pretrained dlib model for extracting 68 facial landmarks.

## Information of Feature extraction used for final model, read BMI/Feature_Extracton_README.md