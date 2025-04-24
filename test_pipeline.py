#!/usr/bin/env python3
"""
bmi_face_predictor.py

Pipeline:
 1. Detect & align faces via MTCNN
 2. Extract embeddings via ArcFace backbone
 3. Fine‑tune embeddings with Triplet Loss
 4. Train an XGBoost regressor
 5. Predict BMI on new images
"""

import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN
from pytorch_metric_learning import losses, miners
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 1. Face Detection & Alignment
mtcnn = MTCNN(image_size=112, margin=0, keep_all=False, device=device)

# Helper to safely load and align images
def load_and_align(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aligned = mtcnn(img_rgb)
    if aligned is None:
        # fallback tensor
        return torch.zeros(3,112,112)
    return aligned

# 2. ArcFace Embedding Model stub (flatten->projector)
class ArcFaceEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # Replace Flatten with actual ArcFace backbone
        self.backbone = nn.Flatten()
        in_features = 3 * 112 * 112
        self.projector = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        return nn.functional.normalize(x, p=2, dim=1)

# 3. Prepare BMI‑labeled dataset via CSV
class BMIDataset(Dataset):
    def __init__(self, df, images_dir):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.images_dir, row['name'])
        aligned = load_and_align(img_path).to(device)
        bmi = torch.tensor(row['bmi'], dtype=torch.float32).to(device)
        return aligned, bmi

# Utility: filter out entries whose image files are missing
def filter_existing_images(df, images_dir):
    existing = df['name'].apply(lambda fn: os.path.isfile(os.path.join(images_dir, fn)))
    missing = df.loc[~existing, 'name'].tolist()
    if missing:
        print(f"Filtering out {len(missing)} entries with missing images: {missing}")
    return df.loc[existing].reset_index(drop=True)

# Load annotations
df = pd.read_csv('data/data.csv')

# Ensure correct types
df['bmi'] = df['bmi'].astype(float)
df['is_training'] = df['is_training'].astype(int)
df['name'] = df['name'].astype(str)

# Filter out rows with missing image files
images_dir = os.path.join('data', 'Images')
df = filter_existing_images(df, images_dir)
print(f"Loaded {len(df)} entries with images from {images_dir}")

# Split train/validation based on 'is_training'
train_df = df[df['is_training'] == 1]
val_df   = df[df['is_training'] == 0]

# Create datasets
datasets = {
    'train': BMIDataset(train_df, images_dir),
    'val':   BMIDataset(val_df, images_dir)
}

# 4. Fine‑tune embeddings with Triplet Loss
miner     = miners.TripletMarginMiner(margin=0.2, type_of_triplets='semi-hard')
criterion = losses.TripletMarginLoss(margin=0.2)
embedder  = ArcFaceEmbedder().to(device)
optimizer = torch.optim.AdamW(embedder.parameters(), lr=1e-4, weight_decay=1e-5)

print("Training with Triplet Loss...")
train_loader = DataLoader(datasets['train'], batch_size=32, shuffle=True)
for epoch in range(10):
    embedder.train()
    total_loss = 0.0
    for imgs, bmis in train_loader:
        feats = embedder(imgs)
        labels = torch.round(bmis).long()
        hard_pairs = miner(feats, labels)
        loss = criterion(feats, labels, hard_pairs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} – Triplet Loss: {total_loss/len(train_loader):.4f}")

# 5. Extract embeddings for regression
def extract_embeddings(df, images_dir):
    embedder.eval()
    embeddings = []
    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, row['name'])
        aligned = load_and_align(img_path).to(device)
        with torch.no_grad():
            emb = embedder(aligned.unsqueeze(0)).cpu().numpy().ravel()
        embeddings.append(emb)
    return np.vstack(embeddings)

# Extract embeddings for training and validation sets
print("Extracting embeddings for training and validation sets...")
X_train = extract_embeddings(train_df, images_dir)
X_val   = extract_embeddings(val_df, images_dir)
y_train = train_df['bmi'].values
y_val   = val_df['bmi'].values

# 6. Train XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=200,
    objective='reg:squarederror',
)

print("Training XGBoost Regressor...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# 7. Evaluate
print("Evaluating on validation set...")
preds = xgb_model.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, preds))

# 8. Save models
torch.save(embedder.state_dict(), 'arcface_bmi_embedder.pth')
xgb_model.save_model('bmi_xgb.json')

# 9. Inference helper
def predict_bmi(image_path):
    aligned = load_and_align(image_path).to(device)
    with torch.no_grad():
        emb = embedder(aligned.unsqueeze(0)).cpu().numpy()
    return xgb_model.predict(emb)[0]

if __name__ == '__main__':
    sample_name = train_df.iloc[0]['name']
    sample_path = os.path.join(images_dir, sample_name)
    print("Example prediction:", predict_bmi(sample_path))
