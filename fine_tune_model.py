import os
from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp_mediapipe  # Rename to avoid conflict
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from facenet_pytorch import InceptionResnetV1

from torchvision import transforms

# ── GLOBAL CONFIG ───────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
NUM_EPOCHS    = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 1e-4

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
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Pre-compute normalized BMI values
        if self.target_transform is not None:
            self.normalized_bmis = np.array([self.target_transform(bmi) for bmi in self.df['bmi']])
        else:
            self.normalized_bmis = self.df['bmi'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["name"])
        image = Image.open(path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Use pre-computed normalized BMI
        label = self.normalized_bmis[idx]
        
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
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze last few layers for fine-tuning
        for param in list(self.backbone.parameters())[-4:]:
            param.requires_grad = True
            
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
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
        loss = loss_fn(preds, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()
        total += loss.item() * imgs.size(0)
    return total / len(dl.dataset)

def evaluate(model, dl, loss_fn, device):
    model.eval()
    total, preds, reals = 0.0, [], []
    with torch.no_grad():
        for imgs, labels in dl:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = loss_fn(out, labels)
            total += loss.item() * imgs.size(0)
            preds.append(out.cpu())
            reals.append(labels.cpu())
    preds = torch.cat(preds).view(-1).numpy()
    reals = torch.cat(reals).view(-1).numpy()
    mae = np.mean(np.abs(preds - reals))
    rmse = np.sqrt(np.mean((preds - reals)**2))
    r2 = r2_score(reals, preds)
    return total / len(dl.dataset), mae, rmse, r2


# ── MAIN ─────────────────────────────────────────────────────────────────────
def compute_dataset_stats(csv_file):
    """Compute BMI statistics and fit the scaler."""
    df = pd.read_csv(csv_file)
    # Extract BMI values
    bmi_values = df['bmi'].values.reshape(-1, 1)
    # Initialize BMI scaler
    bmi_scaler = MinMaxScaler()
    bmi_scaler.fit(bmi_values)
    # Save the fitted scaler
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(bmi_scaler, 'saved_models/bmi_scaler.joblib')
    return bmi_scaler

def load_bmi_scaler():
    """Load the fitted BMI scaler."""
    try:
        return joblib.load('saved_models/bmi_scaler.joblib')
    except (FileNotFoundError, EOFError):
        print("Warning: No saved scaler found. Computing from training data...")
        return compute_dataset_stats("data/train/train_data.csv")

def normalize_bmi(x):
    """Normalize BMI using fitted scaler."""
    return bmi_scaler.transform(np.array([[x]]))[0][0]

def denormalize_bmi(x):
    """Convert normalized BMI back to original scale."""
    return bmi_scaler.inverse_transform(np.array([[x]]))[0][0]

def evaluate_model(model_path, test_csv, img_dir, output_dir):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and scaler
    global bmi_scaler
    bmi_scaler = load_bmi_scaler()
    
    backbone = InceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE)
    model = BMIRegressor(backbone, feat_dim=512).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Debug: Print model parameters and scaler info
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
    
    print("\nBMI Scaler Info:")
    print(f"Min: {bmi_scaler.min_[0]:.2f}, Scale: {bmi_scaler.scale_[0]:.2f}")
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    
    # Setup transforms
    val_tf = transforms.Compose([
        MediaPipeFaceAlignTransform(output_size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Lists to store results
    actual_bmis = []
    predicted_bmis = []
    image_paths = []
    
    # Process each image
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            try:
                img_path = os.path.join(img_dir, row["name"])
                image = Image.open(img_path).convert("RGB")
                
                # Apply transforms and ensure proper tensor shape
                image_tensor = val_tf(image)
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                image_tensor = image_tensor.to(DEVICE)
                
                # Get prediction and denormalize
                pred = model(image_tensor).item()
                pred = denormalize_bmi(pred)  # Denormalize the prediction
                actual = row["bmi"]
                
                # Debug: Print intermediate values
                if idx == 0:  # Only for first image
                    with torch.no_grad():
                        features = model.backbone(image_tensor)
                        print("\nFirst image intermediate values:")
                        print(f"Backbone features: mean={features.mean().item():.4f}, std={features.std().item():.4f}")
                        print(f"Input tensor: mean={image_tensor.mean().item():.4f}, std={image_tensor.std().item():.4f}")
                
                # Store results
                actual_bmis.append(actual)
                predicted_bmis.append(pred)
                image_paths.append(img_path)
                
                # Save comparison image
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title(f'Actual BMI: {actual:.1f}')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(image)
                plt.title(f'Predicted BMI: {pred:.1f}')
                plt.axis('off')
                
                plt.savefig(os.path.join(output_dir, f'comparison_{idx}.png'))
                plt.close()
                
                print(f"Processed image {idx}: Actual BMI = {actual:.1f}, Predicted BMI = {pred:.1f}")
                
            except Exception as e:
                print(f"Error processing image {idx} ({img_path}): {str(e)}")
                continue
    
    # Calculate metrics
    mae = mean_absolute_error(actual_bmis, predicted_bmis)
    rmse = np.sqrt(mean_squared_error(actual_bmis, predicted_bmis))
    r2 = r2_score(actual_bmis, predicted_bmis)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actual_bmis, y=predicted_bmis)
    plt.plot([min(actual_bmis), max(actual_bmis)], 
             [min(actual_bmis), max(actual_bmis)], 
             'r--', label='Perfect Prediction')
    plt.xlabel('Actual BMI')
    plt.ylabel('Predicted BMI')
    plt.title('Actual vs Predicted BMI')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BMI Prediction Model')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], 
                       default='train', help='Mode: train or evaluate')
    parser.add_argument('--model_path', type=str, 
                       default='saved_models/best_bmi_model.pth',
                       help='Path to saved model for evaluation')
    parser.add_argument('--test_csv', type=str,
                       default='data/test/test_data.csv',
                       help='Path to test data CSV')
    parser.add_argument('--img_dir', type=str,
                       default='data/Images',
                       help='Path to image directory')
    parser.add_argument('--output_dir', type=str,
                       default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    if args.mode == 'eval':
        print("Evaluating model...")
        evaluate_model(args.model_path, args.test_csv, args.img_dir, args.output_dir)
    else:
        # Original training code
        mp.set_start_method("spawn", force=True)

        # init wandb
        wandb.init(project="bmi_prediction", config={
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
        })

        # Compute dataset statistics and fit BMI scaler BEFORE creating datasets
        print("Computing dataset statistics...")
        global bmi_scaler
        bmi_scaler = compute_dataset_stats("data/train/train_data.csv")
        print(f"BMI scaler fitted with min: {bmi_scaler.min_[0]:.2f}, scale: {bmi_scaler.scale_[0]:.2f}")
        
        # transforms
        train_tf = transforms.Compose([
            MediaPipeFaceAlignTransform(output_size=(224,224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1),
                scale=(0.9, 1.1), shear=5
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        val_tf = transforms.Compose([
            MediaPipeFaceAlignTransform(output_size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # datasets & loaders
        train_ds = BMIImageDataset(
            csv_file="data/train/train_data.csv",
            img_dir="data/Images",
            transform=train_tf,
            target_transform=normalize_bmi,
        )
        
        val_ds = BMIImageDataset(
            csv_file="data/validation/validation_data.csv",
            img_dir="data/Images",
            transform=val_tf,
            target_transform=normalize_bmi,
        )
        
        test_ds = BMIImageDataset(
            csv_file="data/test/test_data.csv",
            img_dir="data/Images",
            transform=val_tf,
            target_transform=normalize_bmi,
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
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(),
                              lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)
        
        # Learning rate scheduler with warmup
        num_warmup_steps = len(train_loader) * 5  # 5 epochs of warmup
        num_training_steps = len(train_loader) * NUM_EPOCHS
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / 
                      float(max(1, num_training_steps - num_warmup_steps)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping setup
        best_val = float("inf")
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_mae, val_rmse, val_r2 = evaluate(model, val_loader, criterion, DEVICE)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "lr": optimizer.param_groups[0]["lr"],
            })

            print(f"[{epoch}/{NUM_EPOCHS}] "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_mae={val_mae:.4f}  "
                  f"val_rmse={val_rmse:.4f}  "
                  f"val_r2={val_r2:.4f}")

            # Early stopping check
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                torch.save(model.state_dict(), "saved_models/best_bmi_model.pth")
                print("Saved best model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        test_loss, test_mae, test_rmse, test_r2 = evaluate(model, test_loader, criterion, DEVICE)
        print("Test ▶ loss:{:.4f}  mae:{:.4f}  rmse:{:.4f}  r2:{:.4f}"
              .format(test_loss, test_mae, test_rmse, test_r2))
        
        wandb.log({
            "test_loss": test_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        })
        wandb.finish()