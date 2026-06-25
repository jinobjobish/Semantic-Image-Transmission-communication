# datasets.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VOC2012PersonDataset(Dataset):
    def __init__(self, root_dir='dataset/processed', split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        split_file = os.path.join(root_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f]
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.images_dir, f'{img_name}.jpg')).convert('RGB')
        mask = Image.open(os.path.join(self.masks_dir, f'{img_name}.png')).convert('L')
        image = np.array(image)
        mask = np.array(mask) // 255  # convert 0/255 to 0/1

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {'image': image, 'mask': mask, 'name': img_name}


class ROIDataset(Dataset):
    """Dataset that returns only the ROI region (masked image)."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        image = item['image']          # tensor (C,H,W) normalized
        mask = item['mask']            # tensor (H,W) long, 0/1
        # Create ROI: keep only where mask == 1
        roi = image * mask.unsqueeze(0).float()
        # Target for autoencoder is the ROI image itself (reconstruction)
        return roi, roi


class RONIDataset(Dataset):
    """Dataset that returns only the RONI region (background)."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        image = item['image']
        mask = item['mask']
        # RONI: keep where mask == 0
        roni = image * (1 - mask.unsqueeze(0).float())
        return roni, roni


def create_dataloaders(batch_size=8, data_dir='dataset/processed', num_workers=0):
    """
    Create dataloaders for training and validation.
    Note: num_workers=0 avoids multiprocessing issues on Windows.
    """
    # Enhanced augmentation for training
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),               # new
        A.RandomBrightnessContrast(p=0.2),      # new
        A.HueSaturationValue(p=0.2),            # new
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = VOC2012PersonDataset(root_dir=data_dir, split='train', transform=train_transform)
    val_dataset = VOC2012PersonDataset(root_dir=data_dir, split='test', transform=val_transform)

    # For autoencoder training
    roi_train = ROIDataset(train_dataset)
    roni_train = RONIDataset(train_dataset)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'roi_train': DataLoader(roi_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'roni_train': DataLoader(roni_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    }
    return dataloaders