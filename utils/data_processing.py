# utils/data_processing.py
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess_image(image, target_size=(512, 512)):
    """
    Preprocess an image for model input.
    
    Args:
        image (PIL.Image or np.ndarray): Input image.
        target_size (tuple): Desired output size (H, W).
    
    Returns:
        torch.Tensor: Normalized image tensor (C, H, W).
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()  # avoid modifying original
    
    transform = A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image_np)
    return transformed['image']


def postprocess_image(tensor):
    """
    Convert a normalized model output tensor back to a float image in [0, 1].
    
    Args:
        tensor (torch.Tensor or np.ndarray): Image tensor (C, H, W) or (H, W, C) in normalized range.
    
    Returns:
        np.ndarray: float32 image (H, W, C) with values in [0, 1].
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        # Ensure channel dimension first
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3]:
            # Already (C, H, W)
            pass
        elif tensor.dim() == 3 and tensor.shape[-1] in [1, 3]:
            # (H, W, C) → (C, H, W)
            tensor = tensor.permute(2, 0, 1)
        # Denormalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        # Convert to (H, W, C) numpy
        image_np = tensor.permute(1, 2, 0).numpy()
    else:
        image_np = tensor.copy()
        # Assume already in [0,1] range
        image_np = np.clip(image_np, 0, 1)
    
    return image_np.astype(np.float32)


def split_roi_roni(image, segmentation_mask):
    """
    Split an image into ROI (Region of Interest) and RONI (Region of Non-Interest)
    based on a binary segmentation mask (1 = ROI, 0 = RONI).
    
    Args:
        image (PIL.Image, np.ndarray, or torch.Tensor): Input image.
        segmentation_mask (np.ndarray): Binary mask of shape (H, W) with values 0 or 1.
    
    Returns:
        tuple: (roi_image, roni_image, roi_mask, roni_mask)
            - roi_image: image with background zeroed (uint8)
            - roni_image: image with ROI zeroed (uint8)
            - roi_mask: binary mask as uint8
            - roni_mask: binary mask as uint8
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    elif isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        # If tensor was normalized, denormalize approximately (better to use preprocessed image)
        # We'll assume input is already uint8 (0-255) for this function
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image.copy()
    
    # Ensure mask is uint8 and has same spatial dimensions
    if isinstance(segmentation_mask, torch.Tensor):
        mask_np = segmentation_mask.cpu().numpy()
    else:
        mask_np = segmentation_mask.copy()
    
    # If mask is floating (0-1), convert to uint8
    if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
        mask_np = (mask_np > 0.5).astype(np.uint8)
    
    roi_mask = mask_np.astype(np.uint8)
    roni_mask = 1 - roi_mask
    
    # Apply masks
    roi_image = image_np * roi_mask[..., np.newaxis]
    roni_image = image_np * roni_mask[..., np.newaxis]
    
    return roi_image, roni_image, roi_mask, roni_mask