import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

class RetinaFaceAlignTransform:
    """
    Custom PyTorch Transform Class using RetinaFace for face detection and alignment
    """
    def __init__(self, output_size=(224, 224), device='cpu'):
        self.output_size = output_size
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=-1)
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = v2.ToPILImage()(img)
        img = np.array(img)

        faces = self.face_app.get(img)
        if not faces:
            print(f"Warning: No face detected in image. Using original image.")
            # Return resized original image if no face detected
            return Image.fromarray(img).resize(self.output_size)

        face = faces[0]  # Take the first detected face
        aligned_face = face_align.norm_crop(img, face.kps)  # Align using 5 landmarks

        # Convert to PIL for further transforms
        aligned_face = Image.fromarray(aligned_face).resize(self.output_size)

        return aligned_face

class BMIImageDataset(Dataset):
    """
    Custom PyTorch Dataset for BMI Images
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = v2.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels.iloc[idx]
        img_path = os.path.join(self.img_dir, row['name'])
        image = Image.open(img_path).convert('RGB')
        label = row['bmi']
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label 