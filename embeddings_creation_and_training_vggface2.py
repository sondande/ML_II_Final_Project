"""
BMI Prediction Pipeline (Embeddings + Optional Deep-Head + XGBoost)
=================================================================
This single script supports three independent yet **compatible** stages:

1. **embed**   – Create and cache *512-D VGGFace2 embeddings* for every face image.  
2. **train**   – Option A: Train an **XGBoost regressor** on the embeddings (+ optional tabular
   features from *extra_features.csv*).  
   Option B: Fine-tune a *small fully-connected head* on top of the frozen embedder.
3. **realtime** – Serve predictions from a webcam or image file with <100 ms latency.

The pipeline now uses a **unified transformation helper** that applies
augmentation during training and minimal, deterministic preprocessing for
validation / inference.

Run ``python bmi_real_time_pipeline.py --help`` for CLI details.
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

# ---------------------------------------------------------------------------
# 0) Transformation factory --------------------------------------------------
# ---------------------------------------------------------------------------

def get_transform(split: str = "train") -> transforms.Compose:
    """Return torchvision transforms for a given split.
    Training split includes augmentations, while validation/test splits
    use minimal deterministic preprocessing.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(
            #     degrees=5,
            #     translate=(0.02, 0.02),
            #     scale=(0.95, 1.05),
            #     shear=2
            # ),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    # val / test / inference
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ---------------------------------------------------------------------------
# 1) Dataset – returns aligned & transformed Tensor + BMI + filename
# ---------------------------------------------------------------------------
class BMIDatasetEmbed(Dataset):
    """PyTorch dataset that yields (image_tensor, bmi_float, filename_str)."""

    def __init__(self, csv_file: str | Path, image_dir: str | Path,
                 transform: transforms.Compose, mtcnn: MTCNN,
                 split: str = "train"):
        self.image_dir = Path(image_dir)
        df = pd.read_csv(csv_file)
        df = df[df["is_training"].eq(1 if split == "train" else 0)]
        mask = df["name"].apply(lambda p: (self.image_dir / p).exists())
        self.df = df[mask].reset_index(drop=True)
        self.transform, self.mtcnn = transform, mtcnn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["name"]
        img = Image.open(img_path).convert("RGB")

        face = self.mtcnn(img)
        if face is None:
            # Fallback: use the same transforms as the current split
            print(f"[Warning] No face detected in {img_path}. Using fallback preprocessing.")
            x = self.transform(img)
        else:
            # MTCNN returns a tensor in [0,1], convert to PIL Image first
            face_pil = transforms.ToPILImage()(face.squeeze(0))
            x = self.transform(face_pil)
        bmi = torch.tensor(row["bmi"], dtype=torch.float32)
        return x, bmi, row["name"]

# ---------------------------------------------------------------------------
# 2) Embedding extractor – pre-trained VGGFace2 (512-D) ----------------------
# ---------------------------------------------------------------------------

def load_embedder(device: torch.device) -> nn.Module:
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

# ---------------------------------------------------------------------------
# 3) Utility: DataLoader -> compressed .npz ----------------------------------
# ---------------------------------------------------------------------------

def extract_embeddings(loader: DataLoader, embedder: nn.Module, device: torch.device,
                       outfile: str | Path):
    zs, bmis, ids = [], [], []
    with torch.no_grad():
        for imgs, bmi, names in tqdm(loader, desc="Embedding"):
            z = embedder(imgs.to(device))
            zs.append(z.cpu())
            bmis.append(bmi)
            ids.extend(names)
    np.savez_compressed(outfile,
                        id=np.array(ids),
                        emb=torch.cat(zs).numpy(),
                        bmi=torch.cat(bmis).numpy())
    print(f"Saved {len(ids)} embeddings ➜ {outfile}")

# ---------------------------------------------------------------------------
# 4) Small dense head (optional PyTorch path) --------------------------------
# ---------------------------------------------------------------------------
class BMIHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 256, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(1)

class VGGFaceBMI(nn.Module):
    def __init__(self, embedder: nn.Module, head: nn.Module):
        super().__init__()
        self.embedder, self.head = embedder, head

    def forward(self, x):
        with torch.no_grad():
            z = self.embedder(x)
        return self.head(z)

# ---------------------------------------------------------------------------
# 5) Tabular helper for XGBoost ---------------------------------------------
# ---------------------------------------------------------------------------

def build_dataset(npz_path: str | Path, extra_csv: str | Path | None = None) -> Tuple[np.ndarray, np.ndarray]:
    blob = np.load(npz_path, allow_pickle=True)
    X, y, ids = blob["emb"], blob["bmi"], blob["id"]
    if extra_csv:
        df_extra = pd.read_csv(extra_csv).set_index("id")
        X = np.hstack([X, df_extra.loc[ids].to_numpy()])
    return X, y

# ---------------------------------------------------------------------------
# 6) Single-image / webcam prediction ---------------------------------------
# ---------------------------------------------------------------------------

def predict_bmi(image: Image.Image, mtcnn: MTCNN, embedder: nn.Module, regressor):
    face = mtcnn(image)
    if face is None:
        raise RuntimeError("No face detected")
    x = get_transform("val")(face.squeeze(0)).unsqueeze(0)
    with torch.no_grad():
        z = embedder(x.to(next(embedder.parameters()).device)).cpu().numpy()
    return float(regressor.predict(z)[0])

def normalize_bmi(bmi_values):
    """Apply log transformation to BMI values."""
    return np.log1p(bmi_values)

def denormalize_bmi(normalized_values):
    """Convert normalized BMI values back to original scale."""
    return np.expm1(normalized_values)

# ---------------------------------------------------------------------------
# --------------------------- CLI ENTRY -------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("BMI Embedding / Training / Realtime Pipeline")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_embed = sub.add_parser("embed");      p_embed.add_argument("--csv", required=True)
    p_embed.add_argument("--imgdir", required=True); p_embed.add_argument("--split", choices=["train","test"], default="train")
    p_embed.add_argument("--out", default="embeddings.npz"); p_embed.add_argument("--batch", type=int, default=64)
    p_embed.add_argument("--num_workers", type=int, default=0)

    p_xgb = sub.add_parser("train_xgb");    p_xgb.add_argument("--npz", required=True)
    p_xgb.add_argument("--extra_csv");      p_xgb.add_argument("--model_out", default="xgb_bmi.json")

    p_head = sub.add_parser("train_head");  p_head.add_argument("--csv", required=True)
    p_head.add_argument("--imgdir", required=True); p_head.add_argument("--epochs", type=int, default=20)

    p_rt = sub.add_parser("realtime");      p_rt.add_argument("--model", required=True)
    p_rt.add_argument("--mode", choices=["webcam","image"], default="image"); p_rt.add_argument("--img")

    p_eval = sub.add_parser("evaluate_xgb"); p_eval.add_argument("--csv", required=True)
    p_eval.add_argument("--imgdir", required=True); p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--out", default="test_predictions.csv")

    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage == "embed":
        import numpy as np
        import pandas as pd
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_absolute_error
        from facenet_pytorch import MTCNN
        from tqdm import tqdm
        mtcnn = MTCNN(image_size=224, device=DEVICE)
        tf = get_transform("val")  # deterministic for embedding cache
        ds = BMIDatasetEmbed(args.csv, args.imgdir, tf, mtcnn, args.split)
        ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
        embedder = load_embedder(DEVICE)
        extract_embeddings(ld, embedder, DEVICE, args.out)

    elif args.stage == "train_xgb":
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split, RandomizedSearchCV
        from sklearn.metrics import mean_absolute_error, make_scorer
        import numpy as np
        import gc  # For garbage collection
        
        try:
            # Load and validate data
            print("Loading data...")
            X, y = build_dataset(args.npz, args.extra_csv)
            print(f"Loaded data shape: X={X.shape}, y={y.shape}")
            
            # Normalize BMI values
            print("Normalizing BMI values...")
            y_normalized = normalize_bmi(y)
            
            # Ensure data is float32 and contiguous
            X = np.ascontiguousarray(X.astype(np.float32))
            y_normalized = np.ascontiguousarray(y_normalized.astype(np.float32))
            
            # Check for NaN values
            if np.isnan(X).any() or np.isnan(y_normalized).any():
                raise ValueError("Data contains NaN values")
            
            # Split data
            print("Splitting data...")
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_normalized, test_size=0.2, random_state=42
            )
            
            # Clear memory
            del X, y_normalized
            gc.collect()
            
            # Define parameter grid for random search
            param_dist = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5, 7],
                'gamma': [0, 0.1, 0.2, 0.3],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.1, 0.5, 1.0, 5.0]
            }
            
            # Initialize base model
            base_model = XGBRegressor(
                objective="reg:squarederror",
                tree_method='hist',
                n_jobs=1,
                random_state=42
            )
            
            # Create MAE scorer
            mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
            
            # Initialize RandomizedSearchCV
            print("Starting hyperparameter tuning with RandomizedSearchCV...")
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                n_iter=50,  # Number of parameter settings to try
                cv=5,       # 5-fold cross-validation
                scoring=mae_scorer,
                n_jobs=1,   # Single thread to avoid memory issues
                random_state=42,
                verbose=2
            )
            
            # Fit the random search
            random_search.fit(X_tr, y_tr)
            
            # Get best model
            model = random_search.best_estimator_
            print("\nBest parameters found:")
            for param, value in random_search.best_params_.items():
                print(f"{param}: {value}")
            
            # Evaluate on validation set
            print("\nEvaluating best model...")
            val_preds_normalized = model.predict(X_val)
            val_preds = denormalize_bmi(val_preds_normalized)
            y_val_original = denormalize_bmi(y_val)
            mae = mean_absolute_error(y_val_original, val_preds)
            print(f"Validation MAE = {mae:.2f} kg/m²")
            
            # Save model
            print(f"\nSaving model to {args.model_out}...")
            model.save_model(args.model_out)
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        finally:
            # Clean up
            gc.collect()

    elif args.stage == "train_head":
        from torch.optim import AdamW; from torch.nn import L1Loss
        mtcnn, tf = MTCNN(image_size=224, device=DEVICE), get_transform("train")
        ds = BMIDatasetEmbed(args.csv, args.imgdir, tf, mtcnn, "train")
        ld = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
        embedder, head = load_embedder(DEVICE), BMIHead().to(DEVICE)
        model, opt, loss_fn = VGGFaceBMI(embedder, head).to(DEVICE), AdamW(head.parameters(), lr=3e-4), L1Loss()
        for ep in range(1, args.epochs+1):
            model.train(); running=0.
            for imgs, bmi, _ in ld:
                opt.zero_grad(); preds = model(imgs.to(DEVICE)); loss = loss_fn(preds, bmi.to(DEVICE)); loss.backward(); opt.step(); running += loss.item()*imgs.size(0)
            print(f"Epoch {ep} – MAE {running/len(ds):.2f}")
        torch.save(head.state_dict(), "bmi_head.pt"); print("Head weights saved ➜ bmi_head.pt")

    elif args.stage == "evaluate_xgb":
        import numpy as np
        import pandas as pd
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_absolute_error
        from facenet_pytorch import MTCNN
        from tqdm import tqdm
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load test set
        print("Loading test set...")
        mtcnn = MTCNN(image_size=224, device=DEVICE)
        tf = get_transform("val")
        ds = BMIDatasetEmbed(args.csv, args.imgdir, tf, mtcnn, split="test")
        ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        embedder = load_embedder(DEVICE)

        # Extract embeddings for test set
        print("Extracting embeddings for test set...")
        zs, bmis, ids = [], [], []
        with torch.no_grad():
            for imgs, bmi, names in tqdm(ld, desc="Embedding (test)"):
                z = embedder(imgs.to(DEVICE))
                zs.append(z.cpu())
                bmis.append(bmi)
                ids.extend(names)
        X_test = torch.cat(zs).numpy()
        y_test = torch.cat(bmis).numpy()
        id_test = np.array(ids)

        # Load XGBoost model
        print("Loading XGBoost model...")
        model = XGBRegressor()
        model.load_model(args.model)

        # Predict
        print("Predicting on test set...")
        y_test_normalized = normalize_bmi(y_test)
        y_pred_normalized = model.predict(X_test)
        y_pred = denormalize_bmi(y_pred_normalized)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Test MAE = {mae:.2f} kg/m²")

        # Save predictions
        df_out = pd.DataFrame({
            "name": id_test,
            "bmi_true": y_test,
            "bmi_pred": y_pred,
            "bmi_true_normalized": y_test_normalized,
            "bmi_pred_normalized": y_pred_normalized
        })
        df_out.to_csv(args.out, index=False)
        print(f"Predictions saved to {args.out}")
        exit(1)

if __name__ == "__main__":
    main()
