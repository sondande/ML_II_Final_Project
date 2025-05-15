import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp_mediapipe  # Rename to avoid conflict
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from facenet_pytorch import InceptionResnetV1

from torchvision import transforms

# ── GLOBAL CONFIG ───────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps"  if torch.backends.mps.is_available() 
    else "cpu"
)

backbone = InceptionResnetV1(
    pretrained='vggface2',  # load VGGFace2 weights
    classify=False          # drop the final classification layer
).eval().to(DEVICE)

# ── DATASET ─────────────────────────────────────────────────────────────────────
class BMIImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.df               = pd.read_csv(csv_file)
        self.img_dir          = img_dir
        self.transform        = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.img_dir, row["name"])
        image = Image.open(path).convert("RGB")
        label = row["bmi"]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # ensure label is a float tensor of shape [1]
        return image, torch.tensor([label], dtype=torch.float32)


# ── MEDIAPIPE ALIGNMENT ─────────────────────────────────────────────────────────
class MediaPipeFaceAlignTransform:
    def __init__(self, output_size=(224, 224), margin=0.2,
                 min_detection_confidence=0.5, max_num_faces=1):
        self.output_size               = output_size
        self.margin                    = margin
        self.min_detection_confidence  = min_detection_confidence
        self.max_num_faces             = max_num_faces

    def __call__(self, img: Image.Image) -> Image.Image:
        mpfm = mp_mediapipe.solutions.face_mesh  # Use renamed import
        fm   = mpfm.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=self.max_num_faces,
            min_detection_confidence=self.min_detection_confidence
        )

        img_np = np.array(img)
        h, w   = img_np.shape[:2]

        results = fm.process(img_np)
        fm.close()

        if not results.multi_face_landmarks:
            # fallback to center crop
            return self._center_crop(img).resize(self.output_size, Image.BILINEAR)

        lm = results.multi_face_landmarks[0].landmark
        # eye corners
        L_idxs = [33, 133]
        R_idxs = [362,263]
        L_eye  = np.mean([(lm[i].x*w, lm[i].y*h) for i in L_idxs], axis=0)
        R_eye  = np.mean([(lm[i].x*w, lm[i].y*h) for i in R_idxs], axis=0)

        # compute rotation
        dx, dy = R_eye - L_eye
        angle  = np.degrees(np.arctan2(dy, dx))

        # rotate full image
        M       = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        rotated = cv2.warpAffine(img_np, M, (w,h), flags=cv2.INTER_LINEAR)

        # rotate all landmarks
        pts    = np.array([[p.x*w, p.y*h] for p in lm])
        pts_h  = np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1)
        pts_r  = pts_h @ M.T

        x_min, y_min = pts_r.min(axis=0)
        x_max, y_max = pts_r.max(axis=0)
        pad_w = (x_max - x_min) * self.margin
        pad_h = (y_max - y_min) * self.margin

        box = [
            max(0, int(x_min - pad_w)),
            max(0, int(y_min - pad_h)),
            min(w, int(x_max + pad_w)),
            min(h, int(y_max + pad_h)),
        ]

        face = rotated[box[1]:box[3], box[0]:box[2]]
        return Image.fromarray(face).resize(self.output_size, Image.BILINEAR)

    def _center_crop(self, pil: Image.Image) -> Image.Image:
        w, h = pil.size
        side = min(w, h)
        left = (w-side)//2
        top  = (h-side)//2
        return pil.crop((left, top, left+side, top+side))


# ── MODEL ───────────────────────────────────────────────────────────────────────
class BMIRegressor(nn.Module):
    def __init__(self, backbone, feat_dim=512):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)  
        return self.head(feats)


# ── TRAIN / EVAL FUNCS ──────────────────────────────────────────────────────────
def train_one_epoch(model, dl, optim, loss_fn, device):
    model.train()
    total = 0.0
    for imgs, labels in dl:
        imgs, labels = imgs.to(device), labels.to(device)
        optim.zero_grad()
        preds = model(imgs)
        loss  = loss_fn(preds, labels)
        loss.backward()
        optim.step()
        total += loss.item() * imgs.size(0)
    return total / len(dl.dataset)

def evaluate(model, dl, loss_fn, device):
    model.eval()
    total, preds, reals = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in dl:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = loss_fn(out, labels)
            total += loss.item() * imgs.size(0)
            preds.append(out.cpu())
            reals.append(labels.cpu())
    preds = torch.cat(preds).view(-1).numpy()
    reals = torch.cat(reals).view(-1).numpy()
    mae  = np.mean(np.abs(preds - reals))
    rmse = np.sqrt(np.mean((preds - reals)**2))
    return total / len(dl.dataset), mae, rmse


# ── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # spawn for DataLoader pickles
    mp.set_start_method("spawn", force=True)

    # init wandb
    wandb.init(project="bmi_prediction", config={
        "batch_size":    BATCH_SIZE,
        "lr":            LEARNING_RATE,
        "epochs":        NUM_EPOCHS,
        "weight_decay":  WEIGHT_DECAY,
    })

    # transforms
    train_tf = transforms.Compose([
        MediaPipeFaceAlignTransform(output_size=(224,224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.RandomAffine(
            degrees=5, translate=(0.02,0.02),
            scale=(0.95,1.05), shear=2
        ),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225]
        ),
    ])
    val_tf = transforms.Compose([
        MediaPipeFaceAlignTransform(output_size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225]
        ),
    ])

    # if you have computed mean/std, you can plug in a target_transform here:
    def standardize_bmi(x):
        return (x - bmi_mean) / bmi_std

    # datasets & loaders
    train_ds = BMIImageDataset(
        csv_file="data/train/train_data.csv",
        img_dir ="data/Images",
        transform=train_tf,
        target_transform=None,  # or standardize_bmi
    )
    val_ds = BMIImageDataset(
        csv_file="data/validation/validation_data.csv",
        img_dir ="data/Images",
        transform=val_tf,
        target_transform=None,  # or standardize_bmi
    )
    test_ds = BMIImageDataset(
        csv_file="data/test/test_data.csv",
        img_dir ="data/Images",
        transform=val_tf,
        target_transform=None,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # model, loss, optimizer, scheduler
    backbone = InceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE)
    model    = BMIRegressor(backbone, feat_dim=512).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    best_val = float("inf")
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"[{epoch}/{NUM_EPOCHS}] "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_mae={val_mae:.4f}  "
              f"val_rmse={val_rmse:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "saved_models/best_bmi_model.pth")
            print("Saved best model")

    # final test eval
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, criterion, DEVICE)
    print("Test ▶ loss:{:.4f}  mae:{:.4f}  rmse:{:.4f}"
          .format(test_loss, test_mae, test_rmse))
    wandb.log({
        "test_loss": test_loss,
        "test_mae":    test_mae,
        "test_rmse":   test_rmse,
    })
    wandb.finish()