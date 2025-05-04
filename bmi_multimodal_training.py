import os
import ssl
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from torch.utils.tensorboard import SummaryWriter
import wandb

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

IMAGE_DIR = "data/Images"
existing_image_files = os.listdir(IMAGE_DIR)

# ------- Dataset Definition -------
class BMIDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, mtcnn=None, split='train'):
        self.df = pd.read_csv(csv_file).drop(columns=["Unnamed: 0"])
        self.df = self.df[self.df["name"].isin(existing_image_files)]
        if split == 'train':
            self.df = self.df[self.df['is_training'] == 1].reset_index(drop=True)
        else:
            self.df = self.df[self.df['is_training'] == 0].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.mtcnn = mtcnn
        self.gender_map = {'Male': 0, 'Female': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['name'])
        img = Image.open(img_path).convert('RGB')
        if self.mtcnn:
            face = self.mtcnn(img)
            if face is not None:
                img = transforms.ToPILImage()(face.squeeze(0).cpu())
        if self.transform:
            img = self.transform(img)
        bmi = torch.tensor(row['bmi'], dtype=torch.float32)
        gender = torch.tensor(self.gender_map.get(row['gender'], 0), dtype=torch.long)
        return img, gender, bmi

# ------- Model Definition -------
class BMIModel(nn.Module):
    def __init__(self, num_gender_classes=2, dropout_p=0.5):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.gender_head = nn.Linear(num_ftrs, num_gender_classes)
        self.dropout = nn.Dropout(dropout_p)
        self.bmi_head = nn.Linear(num_ftrs + num_gender_classes, 1)

    def forward(self, x, gender):
        features = self.backbone(x)
        gender_logits = self.gender_head(features)
        gender_onehot = torch.nn.functional.one_hot(gender, num_classes=gender_logits.size(1)).float()
        combined = torch.cat([self.dropout(features), gender_onehot.to(features.device)], dim=1)
        bmi_pred = self.bmi_head(combined).squeeze(1)
        return bmi_pred, gender_logits

# ------- Training & Evaluation Utilities -------
def train_one_epoch(model, dataloader, optimizer, criterion_reg, criterion_cls, device, lambda_cls=0.5):
    model.train()
    total_loss = 0.0
    for imgs, genders, bmis in dataloader:
        imgs, genders, bmis = imgs.to(device), genders.to(device), bmis.to(device)
        optimizer.zero_grad()
        pred_bmi, pred_gender = model(imgs, genders)
        loss_reg = criterion_reg(pred_bmi, bmis)
        loss_cls = criterion_cls(pred_gender, genders)
        loss = loss_reg + lambda_cls * loss_cls
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    preds, trues, genders_list = [], [], []
    with torch.no_grad():
        for imgs, genders, bmis in dataloader:
            imgs, genders, bmis = imgs.to(device), genders.to(device), bmis.to(device)
            pred_bmi, _ = model(imgs, genders)
            preds.append(pred_bmi.cpu().numpy())
            trues.append(bmis.cpu().numpy())
            genders_list.append(genders.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    genders_arr = np.concatenate(genders_list)
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    return mae, rmse, preds, trues, genders_arr

# ------- Main Training Script -------
if __name__ == "__main__":
    # Configuration
    CSV_FILE = "data/data.csv"
    IMAGE_DIR = "data/Images"
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize TensorBoard & Weights & Biases
    writer = SummaryWriter(log_dir="runs/bmi_experiment")
    wandb.init(project="bmi_multimodal", config={
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": NUM_EPOCHS
    })

    # Setup face detector
    mtcnn = MTCNN(image_size=224, margin=0, keep_all=False, device=DEVICE)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Datasets & loaders
    train_dataset = BMIDataset(CSV_FILE, IMAGE_DIR, transform, mtcnn, split='train')
    val_dataset = BMIDataset(CSV_FILE, IMAGE_DIR, transform, mtcnn, split='test')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model, losses, optimizer, scheduler
    model = BMIModel().to(DEVICE)
    wandb.watch(model, log="all")
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Training loop with logging
    best_mae = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_reg, criterion_cls, DEVICE)
        val_mae, val_rmse, preds, trues, genders = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_rmse)

        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        })

        # Bias/Fairness checks by gender
        for gender_label, gender_name in enumerate(['Male', 'Female']):
            mask = genders == gender_label
            if mask.sum() > 0:
                g_mae = np.mean(np.abs(preds[mask] - trues[mask]))
                g_rmse = np.sqrt(np.mean((preds[mask] - trues[mask]) ** 2))
                writer.add_scalar(f'MAE/{gender_name}', g_mae, epoch)
                writer.add_scalar(f'RMSE/{gender_name}', g_rmse, epoch)
                wandb.log({
                    f'{gender_name}_mae': g_mae,
                    f'{gender_name}_rmse': g_rmse,
                    'epoch': epoch
                })

        print(f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}  Val MAE: {val_mae:.4f}  Val RMSE: {val_rmse:.4f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_bmi_multimodal_model.pth")
            print("Best model saved.")

    writer.close()
    wandb.finish()
    print("Training complete.")
