# utils/metrics.py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(original, reconstructed, max_val=1.0):
    """Calculate PSNR between original and reconstructed images"""
    # Ensure images are in range [0, 1]
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Calculate MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def calculate_theta_psnr(original, reconstructed, roi_mask, theta=0.8):
    """Calculate θ-PSNR as defined in the paper"""
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Calculate MSE for ROI
    roi_mse = np.mean((original[roi_mask > 0] - reconstructed[roi_mask > 0]) ** 2)
    
    # Calculate MSE for RONI
    roni_mask = 1 - roi_mask
    roni_mse = np.mean((original[roni_mask > 0] - reconstructed[roni_mask > 0]) ** 2)
    
    # Weighted MSE
    weighted_mse = theta * roi_mse + (1 - theta) * roni_mse
    
    if weighted_mse == 0:
        return float('inf')
    
    # Calculate θ-PSNR
    max_val = 1.0
    theta_psnr = 10 * np.log10(max_val**2 / weighted_mse)
    return theta_psnr

def calculate_compression_ratio(original_size, compressed_size):
    """Calculate compression ratio"""
    return original_size / compressed_size

def calculate_ssim(original, reconstructed):
    """Calculate Structural Similarity Index"""
    # Convert to grayscale if needed
    if original.ndim == 3:
        original_gray = np.mean(original, axis=2)
        reconstructed_gray = np.mean(reconstructed, axis=2)
    else:
        original_gray = original
        reconstructed_gray = reconstructed
    
    # Calculate SSIM
    data_range = original_gray.max() - original_gray.min()
    ssim_value = structural_similarity(
        original_gray, 
        reconstructed_gray, 
        data_range=data_range
    )
    return ssim_value