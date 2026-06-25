import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNet
from models.semantic_encoder import SemanticEncoder
from models.semantic_decoder import SemanticDecoder
from utils.data_processing import preprocess_image, postprocess_image
from utils.compression import compress_sparse_matrix, restore_sparse_matrix
from utils.channel_simulation import AWGNChannel, RayleighChannel
from utils.metrics import calculate_psnr, calculate_theta_psnr
from evaluate_result import VOCSegmentationDataset

# ------------------------- Configuration -------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (512, 512)
DATA_DIR = r"D:\mini4\semantic_comm_flask\dataset\VOC2012_train_val\VOC2012_train_val"  # adjust if needed
CHECKPOINT_DIR = 'checkpoints'
CHANNEL_TYPE = 'awgn'   # or 'rayleigh'
SNR_DB = 10.0
THETA = 0.8

# ------------------------- Load Models -------------------------
def load_models():
    unet = UNet(n_channels=3, n_classes=2).to(DEVICE)
    unet_path = os.path.join(CHECKPOINT_DIR, 'unet_final.pth')
    if os.path.exists(unet_path):
        unet.load_state_dict(torch.load(unet_path, map_location=DEVICE, weights_only=True))
        print("✅ U‑Net loaded")
    else:
        raise FileNotFoundError(f"U‑Net checkpoint not found: {unet_path}")

    # SC1 (low bandwidth)
    encoder_sc1 = SemanticEncoder(bandwidth='low').to(DEVICE)
    decoder_sc1 = SemanticDecoder(bandwidth='low').to(DEVICE)
    sc1_path = os.path.join(CHECKPOINT_DIR, 'sc1_final.pth')
    if os.path.exists(sc1_path):
        checkpoint = torch.load(sc1_path, map_location=DEVICE, weights_only=False)
        encoder_sc1.load_state_dict(checkpoint['encoder'])
        decoder_sc1.load_state_dict(checkpoint['decoder'])
        print("✅ SC1 loaded")
    else:
        raise FileNotFoundError(f"SC1 checkpoint not found: {sc1_path}")

    # SC2 (high bandwidth)
    encoder_sc2 = SemanticEncoder(bandwidth='high').to(DEVICE)
    decoder_sc2 = SemanticDecoder(bandwidth='high').to(DEVICE)
    sc2_path = os.path.join(CHECKPOINT_DIR, 'sc2_final.pth')
    if os.path.exists(sc2_path):
        checkpoint = torch.load(sc2_path, map_location=DEVICE, weights_only=False)
        encoder_sc2.load_state_dict(checkpoint['encoder'])
        decoder_sc2.load_state_dict(checkpoint['decoder'])
        print("✅ SC2 loaded")
    else:
        raise FileNotFoundError(f"SC2 checkpoint not found: {sc2_path}")

    unet.eval()
    encoder_sc1.eval()
    decoder_sc1.eval()
    encoder_sc2.eval()
    decoder_sc2.eval()
    return unet, encoder_sc1, decoder_sc1, encoder_sc2, decoder_sc2

# ------------------------- Evaluation Function (No IoU) -------------------------
def evaluate_one_image(image_tensor, gt_mask, models, channel_type, snr_db, theta):
    """
    gt_mask is still passed but not used for IoU.
    Returns dict with psnr, theta_psnr, ssim (%), comp_ratio.
    """
    unet, enc1, dec1, enc2, dec2 = models

    with torch.no_grad():
        seg_logits = unet(image_tensor.unsqueeze(0))
        seg_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        roi_mask = (seg_mask == 1).astype(np.uint8)
        roni_mask = (seg_mask == 0).astype(np.uint8)

        image_np = postprocess_image(image_tensor)
        roi_image = image_np * roi_mask[..., np.newaxis]
        roni_image = image_np * roni_mask[..., np.newaxis]

        roi_tensor = preprocess_image(Image.fromarray((roi_image*255).astype(np.uint8)), IMAGE_SIZE).to(DEVICE)
        roni_tensor = preprocess_image(Image.fromarray((roni_image*255).astype(np.uint8)), IMAGE_SIZE).to(DEVICE)

        enc_roi = enc2(roi_tensor.unsqueeze(0))
        enc_roni = enc1(roni_tensor.unsqueeze(0))

    # Black baselines for compression
    black = torch.zeros(1, 3, *IMAGE_SIZE).to(DEVICE)
    with torch.no_grad():
        n_roi = torch.mean(enc2(black)).item()
        n_roni = torch.mean(enc1(black)).item()

    # Adaptive tolerance
    roi_std = enc_roi.std().item()
    roni_std = enc_roni.std().item()
    tol_roi = min(max(roi_std * 0.5, 0.002), 0.02)
    tol_roni = min(max(roni_std * 0.3, 0.002), 0.015)

    comp_roi = compress_sparse_matrix(enc_roi.squeeze().cpu().numpy(), n_roi, tol=tol_roi)
    comp_roni = compress_sparse_matrix(enc_roni.squeeze().cpu().numpy(), n_roni, tol=tol_roni)

    # Channel simulation
    channel = AWGNChannel(snr_db) if channel_type == 'awgn' else RayleighChannel(snr_db)
    tx_roi = channel.transmit(comp_roi)
    tx_roni = channel.transmit(comp_roni)

    # Restoration
    rest_roi = restore_sparse_matrix(tx_roi, n_roi, enc_roi.shape[1:])
    rest_roni = restore_sparse_matrix(tx_roni, n_roni, enc_roni.shape[1:])

    rest_roi_t = torch.from_numpy(rest_roi.astype(np.float32)).unsqueeze(0).to(DEVICE)
    rest_roni_t = torch.from_numpy(rest_roni.astype(np.float32)).unsqueeze(0).to(DEVICE)

    # Decoding
    with torch.no_grad():
        dec_roi = dec2(rest_roi_t)
        dec_roni = dec1(rest_roni_t)

    # Combine
    dec_roi_np = dec_roi.squeeze().permute(1,2,0).cpu().numpy()
    dec_roni_np = dec_roni.squeeze().permute(1,2,0).cpu().numpy()
    combined = roi_mask[..., np.newaxis] * dec_roi_np + roni_mask[..., np.newaxis] * dec_roni_np
    combined = np.clip(combined, 0, 1)

    # Metrics
    psnr_val = calculate_psnr(image_np, combined)
    theta_psnr_val = calculate_theta_psnr(image_np, combined, roi_mask, theta)

    # SSIM (percentage) – fixed for new skimage
    try:
        ssim_val = ssim(image_np, combined, channel_axis=-1, data_range=1.0)
    except TypeError:
        ssim_val = ssim(image_np, combined, multichannel=True, data_range=1.0)
    ssim_pct = ssim_val * 100

    # Compression ratio
    orig_size = image_np.size * image_np.itemsize
    comp_size = (len(comp_roi[0]) + len(comp_roni[0])) * 4 + (len(comp_roi[1]) + len(comp_roni[1])) * 4
    comp_ratio = orig_size / comp_size if comp_size > 0 else 0

    return {
        'psnr': psnr_val,
        'theta_psnr': theta_psnr_val,
        'ssim': ssim_pct,
        'comp_ratio': comp_ratio
    }

# ------------------------- Main -------------------------
def main():
    models = load_models()

    dataset = VOCSegmentationDataset(DATA_DIR, split='val', image_size=IMAGE_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialise results (no IoU)
    results = {
        'psnr': [],
        'theta_psnr': [],
        'ssim': [],
        'comp_ratio': []
    }

    for images, masks, _ in tqdm(dataloader, desc="Evaluating"):
        img_tensor = images[0].to(DEVICE)
        gt_mask = masks[0].numpy()  # not used for metrics anymore
        out = evaluate_one_image(img_tensor, gt_mask, models, CHANNEL_TYPE, SNR_DB, THETA)
        for key in results:
            results[key].append(out[key])

    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    # Plot 2×2 histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Evaluation Metrics (Channel: {CHANNEL_TYPE.upper()}, SNR={SNR_DB}dB, θ={THETA})')

    metrics = [
        ('psnr', 'PSNR (dB)', axes[0,0]),
        ('theta_psnr', 'θ-PSNR (dB)', axes[0,1]),
        ('ssim', 'SSIM (%)', axes[1,0]),
        ('comp_ratio', 'Compression Ratio', axes[1,1])
    ]

    for key, xlabel, ax in metrics:
        ax.hist(results[key], bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{xlabel} Distribution')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_histograms.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Histograms saved as evaluation_histograms.png")

    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    for key in results:
        mean = results[key].mean()
        std = results[key].std()
        if key == 'ssim':
            print(f"{key:12s}: {mean:6.2f} ± {std:5.2f} %")
        elif key == 'comp_ratio':
            print(f"{key:12s}: {mean:6.2f} ± {std:5.2f}")
        else:
            print(f"{key:12s}: {mean:6.3f} ± {std:5.3f}")

if __name__ == "__main__":
    main()