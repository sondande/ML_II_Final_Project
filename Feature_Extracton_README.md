### `Landmark_Feature_Extraction.py`

Extracts geometric facial features from images using dlib’s 68-point landmark predictor.

- Processes `.jpg` images in `FairFace/detected_faces`
- Computes features like jaw width, nose width, eye distance, etc.
- Saves:
  - `landmark_features.csv` – facial measurements
  - `landmark_overlay/` – images with numbered landmarks

---

### `BMI_Feature_Extraction.ipynb`

Combines facial features, demographics, and BMI for regression modeling.

- Merges:
  - Landmark features
  - FairFace race/gender/age labels
  - BMI data
- One-hot encodes categorical variables
- Trains XGBoost to predict BMI
- Evaluates with MSE, R², and correlation
- Visualizes:
  - Actual vs. predicted BMI
  - Feature importance
  - BMI vs. individual facial features
- Just feature extractions themselves.


### `Fairface_Feature_Extraction.py`

Detects faces and predicts race, gender, and age using a pre-trained FairFace ResNet34 model.

- Inputs:
  - CSV with image paths (`--csv`)
  - `dlib_models/shape_predictor_5_face_landmarks.dat`
- Outputs:
  - Cropped face images in `detected_faces/`
  - Demographic predictions in `fairface_race_gender_age.csv`