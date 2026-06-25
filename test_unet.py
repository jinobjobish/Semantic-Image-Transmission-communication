import torch
import numpy as np
from PIL import Image
from models.unet import UNet
from utils.data_processing import preprocess_image
import sys
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load U‑Net
unet = UNet(n_channels=3, n_classes=2).to(device).eval()
unet.load_state_dict(torch.load('checkpoints/unet_final.pth', map_location=device))
print("U‑Net loaded successfully.")

# Get image path from command line or use default
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = 'dataset/processed/images/2007_000032.jpg'  # fallback

# Check if file exists
if not os.path.exists(image_path):
    print(f"Error: File not found: {image_path}")
    sys.exit(1)

# Open image
img = Image.open(image_path).convert('RGB')

# Preprocess
tensor = preprocess_image(img, (512, 512)).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    seg = unet(tensor)
    mask = torch.argmax(seg, dim=1).squeeze().cpu().numpy()

# Calculate ROI percentage
roi_pct = np.sum(mask == 1) / mask.size * 100
print(f"ROI percentage in this image: {roi_pct:.2f}%")