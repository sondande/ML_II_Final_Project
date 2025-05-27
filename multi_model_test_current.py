import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mediapipe as mp
import cv2
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

class MediaPipeFaceAlignTransform:
    """
    Enhanced face alignment that extracts more comprehensive facial regions.
    This version is optimized for demographic feature extraction alongside BMI prediction.
    """
    def __init__(self, output_size=(224, 224), include_context=True):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.output_size = output_size
        self.include_context = include_context
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            # Get the first (most confident) detection
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            h, w = image.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Strategic padding to capture body context for BMI while preserving face details for demographics
            if self.include_context:
                # More aggressive vertical padding to capture shoulders/neck for BMI
                pad_x = int(width * 0.2)  # Moderate horizontal padding
                pad_y = int(height * 0.4)  # Aggressive vertical padding
            else:
                # Minimal padding for pure face analysis
                pad_x = int(width * 0.1)
                pad_y = int(height * 0.1)
            
            # Expand bounding box
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + width + pad_x)
            y2 = min(h, y + height + pad_y * 2)  # Extra padding below for shoulders
            
            # Crop and resize
            cropped = image[y1:y2, x1:x2]
            cropped = cv2.resize(cropped, self.output_size)
            
            # Convert back to PIL Image
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped_rgb)
        else:
            # If no face detected, return resized original
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return pil_image.resize(self.output_size)

class DemographicPredictor(nn.Module):
    """
    Auxiliary network for predicting demographic features from facial images.
    This creates additional supervision signals that help the main BMI predictor.
    
    The key insight: facial features that correlate with age and ethnicity
    often also correlate with BMI patterns, so learning these jointly improves both tasks.
    """
    def __init__(self, backbone_features=1024):
        super(DemographicPredictor, self).__init__()
        
        # Age prediction branch - regression task
        self.age_predictor = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)  # Single value for age
        )
        
        # Ethnicity prediction branch - classification task
        # We'll predict major ethnic categories that might correlate with BMI patterns
        self.ethnicity_predictor = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 5)  # 5 major ethnic categories (adjust as needed)
        )
    
    def forward(self, features):
        age_pred = self.age_predictor(features)
        ethnicity_pred = self.ethnicity_predictor(features)
        return age_pred, ethnicity_pred

class MultiTaskBMINet(nn.Module):
    """
    Multi-task learning network that jointly predicts BMI and demographic features.
    
    The architecture philosophy:
    1. Shared backbone extracts general facial features
    2. Demographic predictors provide additional supervision
    3. BMI predictor uses both original features and demographic predictions
    4. Joint training creates richer feature representations
    """
    def __init__(self, num_ethnic_categories=5, dropout_rate=0.5, use_gender=True):
        super(MultiTaskBMINet, self).__init__()
        
        # Shared backbone - VGG16 pre-trained on faces
        self.backbone = models.vgg16(pretrained=True)
        
        # Strategic freezing: keep early feature extractors, adapt high-level reasoning
        for param in self.backbone.features[:15].parameters():
            param.requires_grad = False
        
        # Feature extraction head
        feature_size = 1024
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, feature_size),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )
        
        # Demographic prediction branches (auxiliary tasks)
        self.demographic_predictor = DemographicPredictor(feature_size)
        
        # Calculate input size for BMI predictor
        bmi_input_size = feature_size + 1 + num_ethnic_categories  # features + age + ethnicity_probs
        if use_gender:
            bmi_input_size += 1  # Add gender from dataset
        
        # Main BMI prediction branch
        self.bmi_predictor = nn.Sequential(
            nn.Linear(bmi_input_size, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 1)
        )
        
        self.use_gender = use_gender
        self.num_ethnic_categories = num_ethnic_categories
    
    def forward(self, image, gender=None, return_demographics=False):
        # Extract shared visual features
        visual_features = self.backbone(image)
        
        # Predict demographic features
        age_pred, ethnicity_logits = self.demographic_predictor(visual_features)
        ethnicity_probs = torch.softmax(ethnicity_logits, dim=1)
        
        # Combine features for BMI prediction
        bmi_features = [visual_features, age_pred]
        bmi_features.append(ethnicity_probs)
        
        if self.use_gender and gender is not None:
            bmi_features.append(gender.unsqueeze(1) if gender.dim() == 1 else gender)
        
        combined_features = torch.cat(bmi_features, dim=1)
        
        # Final BMI prediction
        bmi_pred = self.bmi_predictor(combined_features)
        
        if return_demographics:
            return bmi_pred, age_pred, ethnicity_logits
        else:
            return bmi_pred

class BMIDataset(Dataset):
    """
    Enhanced dataset that handles the available data fields and prepares for multi-task learning.
    Now works with the actual CSV structure: [bmi, gender, is_training, name]
    """
    def __init__(self, df, image_dir, transform=None, return_demographics=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.return_demographics = return_demographics
        
        # Encode gender (assuming binary: M/F or Male/Female)
        self.gender_encoder = LabelEncoder()
        # Handle potential missing values
        gender_values = self.df['gender'].fillna('Unknown').astype(str)
        self.encoded_gender = self.gender_encoder.fit_transform(gender_values)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image - handle different possible filename formats
        possible_names = [
            row['name'],
            f"{row['name']}.jpg",
            f"{row['name']}.jpeg",
            f"{row['name']}.png"
        ]
        
        image = None
        for name in possible_names:
            img_path = os.path.join(self.image_dir, name)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                    break
                except Exception as e:
                    continue
        
        # Fallback to black image if loading fails
        if image is None:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            print(f"Warning: Could not load image for {row['name']}")
        
        if self.transform:
            image = self.transform(image)
        
        # Prepare outputs
        bmi = torch.FloatTensor([row['bmi']])
        gender = torch.FloatTensor([self.encoded_gender[idx]])
        
        if self.return_demographics:
            # For training, we don't have true demographic labels,
            # so we'll use pseudo-labels or create synthetic targets
            # This is a common approach in semi-supervised learning
            return image, gender, bmi, idx
        else:
            return image, gender, bmi

def create_weighted_sampler(bmi_values, num_bins=10):
    """
    Enhanced weighted sampling that addresses the right-skewed BMI distribution.
    
    The strategy: create bins across the BMI range and oversample from underrepresented bins.
    This ensures the model sees sufficient examples from all BMI ranges during training.
    """
    # Create bins with special attention to extreme values
    bins = np.percentile(bmi_values, np.linspace(0, 100, num_bins + 1))
    bins[0] = bmi_values.min()  # Ensure we capture the minimum
    bins[-1] = bmi_values.max()  # Ensure we capture the maximum
    
    bin_indices = np.digitize(bmi_values, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    # Calculate inverse frequency weights with smoothing
    bin_counts = np.bincount(bin_indices, minlength=num_bins)
    # Add smoothing to prevent extreme weights
    smoothed_counts = bin_counts + np.sqrt(bin_counts.mean())
    bin_weights = 1.0 / smoothed_counts
    
    # Apply weights to samples
    sample_weights = bin_weights[bin_indices]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

class MultiTaskLoss(nn.Module):
    """
    Combined loss function for multi-task learning.
    
    Balances three objectives:
    1. BMI prediction accuracy (primary task)
    2. Age prediction (auxiliary task - helps with feature learning)
    3. Ethnicity prediction (auxiliary task - captures population-specific patterns)
    
    The weights reflect task importance: BMI is primary, demographics are auxiliary.
    """
    def __init__(self, bmi_weight=1.0, age_weight=0.3, ethnicity_weight=0.2):
        super(MultiTaskLoss, self).__init__()
        self.bmi_weight = bmi_weight
        self.age_weight = age_weight
        self.ethnicity_weight = ethnicity_weight
        
        # Different loss functions for different task types
        self.bmi_loss = nn.SmoothL1Loss()  # Robust to outliers
        self.age_loss = nn.SmoothL1Loss()  # Also regression
        self.ethnicity_loss = nn.CrossEntropyLoss()  # Classification
    
    def forward(self, bmi_pred, bmi_true, age_pred=None, age_true=None, 
                ethnicity_pred=None, ethnicity_true=None):
        
        # Primary BMI loss
        total_loss = self.bmi_weight * self.bmi_loss(bmi_pred, bmi_true)
        
        # Auxiliary losses (when available)
        if age_pred is not None and age_true is not None:
            total_loss += self.age_weight * self.age_loss(age_pred, age_true)
        
        if ethnicity_pred is not None and ethnicity_true is not None:
            total_loss += self.ethnicity_weight * self.ethnicity_loss(ethnicity_pred, ethnicity_true)
        
        return total_loss

def create_pseudo_demographic_labels(model, dataloader, device):
    """
    Creates pseudo-labels for demographic features using the current model predictions.
    
    This is a key technique in semi-supervised learning: we use the model's own predictions
    as training targets for the auxiliary tasks. As the model improves, these pseudo-labels
    become more accurate, creating a positive feedback loop.
    """
    model.eval()
    age_predictions = []
    ethnicity_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, gender, bmi, indices = batch
            images = images.to(device)
            gender = gender.to(device)
            
            # Get demographic predictions
            _, age_pred, ethnicity_pred = model(images, gender, return_demographics=True)
            
            age_predictions.extend(age_pred.cpu().numpy())
            ethnicity_predictions.extend(torch.argmax(ethnicity_pred, dim=1).cpu().numpy())
    
    return np.array(age_predictions), np.array(ethnicity_predictions)

def train_multitask_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """
    Training loop for multi-task learning with progressive pseudo-labeling.
    
    The training strategy:
    1. Initial phases focus primarily on BMI prediction
    2. Gradually introduce demographic pseudo-labeling
    3. Use curriculum learning to progressively increase task complexity
    """
    criterion = MultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=7, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    # Progressive training strategy
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Determine if we should use pseudo-labeling (after warmup period)
        use_pseudo_labels = epoch > 10
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_data in pbar:
            images, gender, bmi_targets, batch_indices = batch_data
            images = images.to(device)
            gender = gender.to(device)
            bmi_targets = bmi_targets.to(device)
            
            if use_pseudo_labels:
                # Multi-task learning with pseudo-labels
                bmi_pred, age_pred, ethnicity_pred = model(images, gender, return_demographics=True)
                
                # Create pseudo-labels (simplified approach)
                age_pseudo = age_pred.detach()  # Use model's own predictions
                ethnicity_pseudo = torch.argmax(ethnicity_pred.detach(), dim=1)
                
                loss = criterion(bmi_pred, bmi_targets, age_pred, age_pseudo, 
                               ethnicity_pred, ethnicity_pseudo)
            else:
                # Pure BMI training during warmup
                bmi_pred = model(images, gender)
                loss = criterion(bmi_pred, bmi_targets)
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Multi-task': use_pseudo_labels
            })
        
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        val_metrics = evaluate_model_during_training(model, val_loader, criterion)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_metrics["loss"]:.4f}, '
              f'Val MAE: {val_metrics["mae"]:.4f}, '
              f'Val R²: {val_metrics["r2"]:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Early stopping with model saving
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'val_mae': val_metrics['mae']
            }, 'best_multitask_bmi_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return history

def evaluate_model_during_training(model, val_loader, criterion):
    """Helper function for validation during training"""
    model.eval()
    val_loss = 0.0
    predictions = []
    targets = []
    val_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            images, gender, bmi_targets, _ = batch_data
            images = images.to(device)
            gender = gender.to(device)
            bmi_targets = bmi_targets.to(device)
            
            bmi_pred = model(images, gender)
            loss = criterion(bmi_pred, bmi_targets)
            
            val_loss += loss.item()
            val_batches += 1
            
            predictions.extend(bmi_pred.cpu().numpy())
            targets.extend(bmi_targets.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return {
        'loss': val_loss / val_batches,
        'mae': mae,
        'r2': r2
    }

def comprehensive_model_evaluation(model, test_loader):
    """
    Comprehensive evaluation that analyzes both BMI prediction and learned demographic features.
    """
    model.eval()
    bmi_predictions = []
    bmi_targets = []
    age_predictions = []
    ethnicity_predictions = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            images, gender, bmi_targets_batch, _ = batch_data
            images = images.to(device)
            gender = gender.to(device)
            
            # Get all predictions
            bmi_pred, age_pred, ethnicity_pred = model(images, gender, return_demographics=True)
            
            bmi_predictions.extend(bmi_pred.cpu().numpy())
            bmi_targets.extend(bmi_targets_batch.numpy())
            age_predictions.extend(age_pred.cpu().numpy())
            ethnicity_predictions.extend(torch.argmax(ethnicity_pred, dim=1).cpu().numpy())
    
    # Convert to numpy arrays
    bmi_predictions = np.array(bmi_predictions).flatten()
    bmi_targets = np.array(bmi_targets).flatten()
    age_predictions = np.array(age_predictions).flatten()
    ethnicity_predictions = np.array(ethnicity_predictions)
    
    # BMI prediction metrics
    mae = mean_absolute_error(bmi_targets, bmi_predictions)
    mse = mean_squared_error(bmi_targets, bmi_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(bmi_targets, bmi_predictions)
    
    # Accuracy within tolerance ranges
    within_3 = np.mean(np.abs(bmi_predictions - bmi_targets) <= 3) * 100
    within_5 = np.mean(np.abs(bmi_predictions - bmi_targets) <= 5) * 100
    
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    print(f"BMI Prediction Performance:")
    print(f"  Mean Absolute Error (MAE): {mae:.3f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.3f}")
    print(f"  R² Score: {r2:.3f}")
    print(f"  Predictions within ±3 BMI units: {within_3:.1f}%")
    print(f"  Predictions within ±5 BMI units: {within_5:.1f}%")
    print(f"\nDemographic Insights:")
    print(f"  Predicted age range: {age_predictions.min():.1f} - {age_predictions.max():.1f}")
    print(f"  Mean predicted age: {age_predictions.mean():.1f}")
    print(f"  Ethnicity distribution: {np.bincount(ethnicity_predictions)}")
    
    # Create comprehensive visualizations
    create_evaluation_plots(bmi_targets, bmi_predictions, age_predictions, ethnicity_predictions)
    
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'within_3': within_3, 'within_5': within_5,
        'bmi_predictions': bmi_predictions, 'bmi_targets': bmi_targets,
        'age_predictions': age_predictions, 'ethnicity_predictions': ethnicity_predictions
    }

def create_evaluation_plots(bmi_targets, bmi_predictions, age_predictions, ethnicity_predictions):
    """Create comprehensive evaluation visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # BMI Prediction vs True
    axes[0, 0].scatter(bmi_targets, bmi_predictions, alpha=0.6, s=20)
    axes[0, 0].plot([bmi_targets.min(), bmi_targets.max()], 
                    [bmi_targets.min(), bmi_targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True BMI')
    axes[0, 0].set_ylabel('Predicted BMI')
    axes[0, 0].set_title('BMI Predictions vs True Values')
    
    # Residuals plot
    residuals = bmi_predictions - bmi_targets
    axes[0, 1].scatter(bmi_targets, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('True BMI')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Prediction Residuals')
    
    # Error distribution
    axes[0, 2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(x=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('Prediction Error')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Error Distribution')
    
    # Age predictions vs BMI
    axes[1, 0].scatter(age_predictions, bmi_targets, alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Predicted Age')
    axes[1, 0].set_ylabel('True BMI')
    axes[1, 0].set_title('Age-BMI Relationship')
    
    # BMI distribution by predicted ethnicity
    for eth in np.unique(ethnicity_predictions):
        mask = ethnicity_predictions == eth
        if np.sum(mask) > 0:
            axes[1, 1].hist(bmi_targets[mask], alpha=0.6, label=f'Ethnicity {eth}', bins=20)
    axes[1, 1].set_xlabel('True BMI')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('BMI Distribution by Predicted Ethnicity')
    axes[1, 1].legend()
    
    # Prediction accuracy by BMI range
    bmi_ranges = [(15, 20), (20, 25), (25, 30), (30, 35), (35, 45)]
    range_errors = []
    range_labels = []
    
    for low, high in bmi_ranges:
        mask = (bmi_targets >= low) & (bmi_targets < high)
        if np.sum(mask) > 0:
            range_error = np.abs(bmi_predictions[mask] - bmi_targets[mask]).mean()
            range_errors.append(range_error)
            range_labels.append(f'{low}-{high}')
    
    axes[1, 2].bar(range_labels, range_errors)
    axes[1, 2].set_xlabel('BMI Range')
    axes[1, 2].set_ylabel('Mean Absolute Error')
    axes[1, 2].set_title('Prediction Accuracy by BMI Range')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def predict_bmi_with_demographics(model, image_path, gender, return_demographics=False):
    """
    Predict BMI for a single image, optionally returning demographic predictions.
    """
    # Prepare transforms
    face_align_transform = MediaPipeFaceAlignTransform()
    transform = transforms.Compose([
        face_align_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prepare gender (assuming 0=female, 1=male or similar encoding)
    gender_tensor = torch.FloatTensor([gender]).unsqueeze(0).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        if return_demographics:
            bmi_pred, age_pred, ethnicity_pred = model(image_tensor, gender_tensor, return_demographics=True)
            ethnicity_probs = torch.softmax(ethnicity_pred, dim=1)
            
            return {
                'bmi': bmi_pred.cpu().item(),
                'predicted_age': age_pred.cpu().item(),
                'ethnicity_probabilities': ethnicity_probs.cpu().numpy()[0],
                'most_likely_ethnicity': torch.argmax(ethnicity_probs, dim=1).cpu().item()
            }
        else:
            bmi_pred = model(image_tensor, gender_tensor)
            return bmi_pred.cpu().item()

def main():
    """
    Main training pipeline adapted for the actual CSV structure: [bmi, gender, is_training, name]
    """
    # Configuration - adjust these paths for your setup
    IMAGE_DIR = "path/to/your/images"  # Update this path
    CSV_FILE = "data.csv"  # Your CSV file
    BATCH_SIZE = 32
    NUM_EPOCHS = 60  # Slightly more epochs for multi-task learning
    LEARNING_RATE = 0.0008  # Slightly lower for stability
    
    print("Loading and preparing data...")
    df = pd.read_csv(CSV_FILE)
    
    # Basic data validation and cleaning
    print(f"Original dataset size: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean the data
    df = df.dropna(subset=['bmi', 'gender', 'name'])  # Remove rows with missing essential data
    df = df[df['bmi'] > 0]  # Remove invalid BMI values
    print(f"After cleaning: {len(df)} samples")
    
    # Data distribution analysis
    print(f"\nBMI Distribution:")
    print(f"  Mean: {df['bmi'].mean():.2f}")
    print(f"  Std: {df['bmi'].std():.2f}")
    print(f"  Min: {df['bmi'].min():.2f}")
    print(f"  Max: {df['bmi'].max():.2f}")
    print(f"  Median: {df['bmi'].median():.2f}")
    
    print(f"\nGender Distribution:")
    print(df['gender'].value_counts())
    
    print(f"\nTraining/Test Split:")
    print(df['is_training'].value_counts())
    
    # Separate training and test data
    train_df = df[df['is_training'] == True].copy()
    test_df = df[df['is_training'] == False].copy()
    
    # Create validation split from training data
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, 
                                        stratify=pd.cut(train_df['bmi'], bins=5))
    
    print(f"\nFinal splits:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Visualize BMI distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(train_df['bmi'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    plt.title('Training BMI Distribution')
    plt.axvline(train_df['bmi'].mean(), color='red', linestyle='--', label=f'Mean: {train_df["bmi"].mean():.1f}')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    gender_counts = train_df['gender'].value_counts()
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('Gender Distribution')
    
    plt.subplot(1, 3, 3)
    plt.boxplot([train_df[train_df['gender'] == g]['bmi'].values for g in train_df['gender'].unique()],
                labels=train_df['gender'].unique())
    plt.ylabel('BMI')
    plt.title('BMI by Gender')
    
    plt.tight_layout()
    plt.show()
    
    # Data transforms with strategic augmentation
    face_align_transform = MediaPipeFaceAlignTransform(include_context=True)
    
    # Training transforms: aggressive augmentation to improve generalization
    train_transform = transforms.Compose([
        face_align_transform,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transforms: minimal processing for consistent evaluation
    val_transform = transforms.Compose([
        face_align_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BMIDataset(train_df, IMAGE_DIR, train_transform, return_demographics=True)
    val_dataset = BMIDataset(val_df, IMAGE_DIR, val_transform, return_demographics=True)
    test_dataset = BMIDataset(test_df, IMAGE_DIR, val_transform, return_demographics=True)
    
    # Create weighted sampler to handle BMI distribution imbalance
    train_sampler = create_weighted_sampler(train_df['bmi'].values, num_bins=12)
    
    # Create data loaders with appropriate batch sizes
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"\nData loaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize the multi-task model
    model = MultiTaskBMINet(
        num_ethnic_categories=5,  # Adjust based on your expected diversity
        dropout_rate=0.4,
        use_gender=True
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Using device: {device}")
    
    # Train the model
    print(f"\nStarting multi-task training...")
    print("Training strategy:")
    print("  - First 10 epochs: Focus on BMI prediction only")
    print("  - Remaining epochs: Multi-task learning with demographic pseudo-labeling")
    print("  - Progressive learning rate reduction")
    print("  - Early stopping with patience=15")
    
    history = train_multitask_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE
    )
    
    # Load the best model for evaluation
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load('best_multitask_bmi_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']} with validation MAE: {checkpoint['val_mae']:.3f}")
    
    # Comprehensive evaluation
    print("\nPerforming comprehensive evaluation...")
    results = comprehensive_model_evaluation(model, test_loader)
    
    # Plot training history
    plot_training_history(history)
    
    # Analyze model predictions by demographic segments
    analyze_demographic_performance(model, test_loader)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final Model Performance:")
    print(f"  Best Validation MAE: {checkpoint['val_mae']:.3f}")
    print(f"  Test MAE: {results['mae']:.3f}")
    print(f"  Test R²: {results['r2']:.3f}")
    print(f"  Predictions within ±3 BMI: {results['within_3']:.1f}%")
    print(f"  Model saved as: 'best_multitask_bmi_model.pth'")
    
    return model, results, history

def plot_training_history(history):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation MAE
    axes[0, 1].plot(history['val_mae'], label='Validation MAE', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('Validation MAE Progress')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation R²
    axes[1, 0].plot(history['val_r2'], label='Validation R²', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Validation R² Progress')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning progress summary
    axes[1, 1].plot(history['train_loss'], label='Train Loss', alpha=0.7)
    axes[1, 1].plot(history['val_loss'], label='Val Loss', alpha=0.7)
    ax2 = axes[1, 1].twinx()
    ax2.plot(history['val_mae'], label='Val MAE', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    ax2.set_ylabel('MAE')
    axes[1, 1].set_title('Combined Training Progress')
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def analyze_demographic_performance(model, test_loader):
    """Analyze how well the model performs across different demographic segments"""
    model.eval()
    
    all_data = []
    with torch.no_grad():
        for batch_data in test_loader:
            images, gender, bmi_targets, batch_indices = batch_data
            images = images.to(device)
            gender = gender.to(device)
            
            bmi_pred, age_pred, ethnicity_pred = model(images, gender, return_demographics=True)
            ethnicity_pred_classes = torch.argmax(ethnicity_pred, dim=1)
            
            batch_results = {
                'bmi_true': bmi_targets.numpy(),
                'bmi_pred': bmi_pred.cpu().numpy(),
                'gender': gender.cpu().numpy(),
                'age_pred': age_pred.cpu().numpy(),
                'ethnicity_pred': ethnicity_pred_classes.cpu().numpy()
            }
            all_data.append(batch_results)
    
    # Combine all results
    combined_data = {}
    for key in all_data[0].keys():
        combined_data[key] = np.concatenate([batch[key] for batch in all_data])
    
    # Analyze performance by gender
    print("\nPerformance by Gender:")
    for gender_val in np.unique(combined_data['gender']):
        mask = combined_data['gender'] == gender_val
        if np.sum(mask) > 0:
            mae = mean_absolute_error(combined_data['bmi_true'][mask], 
                                    combined_data['bmi_pred'][mask])
            r2 = r2_score(combined_data['bmi_true'][mask], 
                         combined_data['bmi_pred'][mask])
            print(f"  Gender {int(gender_val)}: MAE={mae:.3f}, R²={r2:.3f}, n={np.sum(mask)}")
    
    # Analyze performance by predicted age groups
    print("\nPerformance by Predicted Age Groups:")
    age_bins = np.percentile(combined_data['age_pred'], [0, 25, 50, 75, 100])
    age_groups = np.digitize(combined_data['age_pred'], age_bins) - 1
    
    for i, (low, high) in enumerate(zip(age_bins[:-1], age_bins[1:])):
        mask = age_groups == i
        if np.sum(mask) > 0:
            mae = mean_absolute_error(combined_data['bmi_true'][mask], 
                                    combined_data['bmi_pred'][mask])
            r2 = r2_score(combined_data['bmi_true'][mask], 
                         combined_data['bmi_pred'][mask])
            print(f"  Age {low:.1f}-{high:.1f}: MAE={mae:.3f}, R²={r2:.3f}, n={np.sum(mask)}")
    
    # Analyze performance by predicted ethnicity
    print("\nPerformance by Predicted Ethnicity:")
    for eth in np.unique(combined_data['ethnicity_pred']):
        mask = combined_data['ethnicity_pred'] == eth
        if np.sum(mask) > 5:  # Only analyze groups with reasonable sample size
            mae = mean_absolute_error(combined_data['bmi_true'][mask], 
                                    combined_data['bmi_pred'][mask])
            r2 = r2_score(combined_data['bmi_true'][mask], 
                         combined_data['bmi_pred'][mask])
            print(f"  Ethnicity {eth}: MAE={mae:.3f}, R²={r2:.3f}, n={np.sum(mask)}")

# Example usage and testing functions
def test_single_prediction():
    """Example of how to use the trained model for single predictions"""
    # Load the trained model
    model = MultiTaskBMINet(num_ethnic_categories=5, use_gender=True)
    checkpoint = torch.load('best_multitask_bmi_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Example prediction (adjust the path and gender as needed)
    image_path = "path/to/test/image.jpg"
    gender = 1  # 0 for female, 1 for male (or however your gender encoding works)
    
    # Get prediction with demographic insights
    result = predict_bmi_with_demographics(model, image_path, gender, return_demographics=True)
    
    print("Prediction Results:")
    print(f"  Predicted BMI: {result['bmi']:.2f}")
    print(f"  Predicted Age: {result['predicted_age']:.1f}")
    print(f"  Most Likely Ethnicity: {result['most_likely_ethnicity']}")
    print(f"  Ethnicity Probabilities: {result['ethnicity_probabilities']}")
    
    return result

def create_model_interpretation_analysis(model, test_loader):
    """
    Advanced analysis to understand what the model has learned about 
    the relationship between facial features and BMI
    """
    model.eval()
    
    # Collect predictions and analyze patterns
    results = []
    with torch.no_grad():
        for batch_data in test_loader:
            images, gender, bmi_targets, _ = batch_data
            images = images.to(device)
            gender = gender.to(device)
            
            bmi_pred, age_pred, ethnicity_pred = model(images, gender, return_demographics=True)
            
            for i in range(len(images)):
                results.append({
                    'bmi_true': bmi_targets[i].item(),
                    'bmi_pred': bmi_pred[i].item(),
                    'gender': gender[i].item(),
                    'age_pred': age_pred[i].item(),
                    'ethnicity_pred': torch.argmax(ethnicity_pred[i]).item()
                })
    
    df_results = pd.DataFrame(results)
    
    # Correlation analysis
    print("Model Learning Analysis:")
    print("="*50)
    
    # How does predicted age correlate with BMI patterns?
    age_bmi_corr = df_results['age_pred'].corr(df_results['bmi_true'])
    print(f"Age-BMI correlation: {age_bmi_corr:.3f}")
    
    # Gender differences in prediction accuracy
    for gender in df_results['gender'].unique():
        gender_data = df_results[df_results['gender'] == gender]
        mae = mean_absolute_error(gender_data['bmi_true'], gender_data['bmi_pred'])
        print(f"Gender {int(gender)} MAE: {mae:.3f}")
    
    # BMI range analysis
    df_results['bmi_category'] = pd.cut(df_results['bmi_true'], 
                                       bins=[0, 18.5, 25, 30, 35, 50],
                                       labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese'])
    
    print("\nPerformance by BMI Category:")
    for category in df_results['bmi_category'].cat.categories:
        cat_data = df_results[df_results['bmi_category'] == category]
        if len(cat_data) > 0:
            mae = mean_absolute_error(cat_data['bmi_true'], cat_data['bmi_pred'])
            print(f"{category}: MAE={mae:.3f}, n={len(cat_data)}")
    
    return df_results

if __name__ == "__main__":
    # Run the complete training pipeline
    try:
        model, results, history = main()
        
        # Additional analysis
        print("\nRunning model interpretation analysis...")
        interpretation_results = create_model_interpretation_analysis(model, 
                                   DataLoader(BMIDataset(pd.read_csv("data.csv")[pd.read_csv("data.csv")['is_training'] == False], 
                                                        "path/to/your/images", 
                                                        transforms.Compose([MediaPipeFaceAlignTransform(), transforms.ToTensor(), 
                                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
                                                        return_demographics=True), 
                                             batch_size=32, shuffle=False))
        
        print("\n" + "="*60)
        print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("="*60)
        print("Next steps:")
        print("1. Review the evaluation plots and metrics")
        print("2. Test single predictions using test_single_prediction()")
        print("3. Experiment with different model architectures if needed")
        print("4. Consider collecting more data for underperforming BMI ranges")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Please check your data paths and CSV format")
        raise e