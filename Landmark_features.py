import os
import cv2
import dlib
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────
PREDICTOR_PATH = "dlib_models/shape_predictor_68_face_landmarks.dat"
IMAGE_FOLDER = "FairFace/detected_faces"
OUTPUT_CSV = "landmark_features.csv"
OVERLAY_FOLDER = "landmark_overlay"

# ─── SETUP ───────────────────────────────────────────────

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
os.makedirs(OVERLAY_FOLDER, exist_ok=True)

# ─── FEATURE EXTRACTION ─────────────────────────────────────────────

def get_landmark_features(shape):
    points = np.array([(p.x, p.y) for p in shape.parts()])

    def dist(p1, p2):
        return np.linalg.norm(points[p1] - points[p2])

    features = {
        "jaw_width": dist(0, 16),
        "face_height": dist(8, 27),
        "cheekbone_width": dist(1, 15),
        "nose_width": dist(31, 35),
        "mouth_width": dist(48, 54),
        "mouth_height": dist(51, 57),
        "eye_distance": dist(39, 42),
        "left_eye_width": dist(36, 39),
        "right_eye_width": dist(42, 45),
        "face_width_to_height": dist(0, 16) / dist(8, 27) if dist(8, 27) != 0 else 0,
        "mouth_to_nose_ratio": dist(48, 54) / dist(31, 35) if dist(31, 35) != 0 else 0,
    }
    return features

# ─── LOOP THROUGH IMAGES ────────────────────────────────────────────
features_list = []
image_paths = list(Path(IMAGE_FOLDER).glob("*.jpg"))

for img_path in image_paths:
    img = dlib.load_rgb_image(str(img_path))
    dets = detector(img, 1)
    if len(dets) == 0:
        print(f"No face found in {img_path.name}")
        continue

    shape = predictor(img, dets[0])
    feats = get_landmark_features(shape)
    feats["image"] = img_path.name
    features_list.append(feats)

    # Draw numbered landmarks and save overlay image
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(68):
        pt = shape.part(i)
        cv2.putText(img_bgr, str(i), (pt.x, pt.y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    out_path = Path(OVERLAY_FOLDER) / img_path.name
    cv2.imwrite(str(out_path), img_bgr)

# ─── SAVE TO CSV ────────────────────────────────────────────

df = pd.DataFrame(features_list)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved facial landmark features to {OUTPUT_CSV}")
print(f"Saved landmark overlay images to {OVERLAY_FOLDER}/")