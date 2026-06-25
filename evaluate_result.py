# evaluate_result.py (updated)
import os
import argparse
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys
from skimage.metrics import structural_similarity as ssim

# Add project root to path (adjust if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNet
from models.semantic_encoder import SemanticEncoder
from models.semantic_decoder import SemanticDecoder
from utils.data_processing import preprocess_image, postprocess_image
from utils.compression import compress_sparse_matrix, restore_sparse_matrix
from utils.channel_simulation import AWGNChannel, RayleighChannel
from utils.metrics import calculate_psnr, calculate_theta_psnr

# Local IoU function
def calculate_iou(pred_mask, gt_mask):
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 1.0

# ------------------------- Configuration -------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (512, 512)

class SemanticSystem:
    """Wrapper for the semantic communication models (same as in Flask app)"""
    def __init__(self, unet_path='checkpoints/unet_final.pth',
                 sc1_path='checkpoints/sc1_final.pth',
                 sc2_path='checkpoints/sc2_final.pth'):
        self.device = DEVICE
        # U-Net
        self.unet = UNet(n_channels=3, n_classes=2).to(self.device)
        if os.path.exists(unet_path):
            self.unet.load_state_dict(torch.load(unet_path, map_location=self.device, weights_only=True))
            print(f"✅ Loaded U-Net from {unet_path}")
        else:
            print(f"⚠️ U-Net checkpoint not found: {unet_path} – using random weights")
        self.unet.eval()

        # SC1 (low bandwidth)
        self.encoder_sc1 = SemanticEncoder(bandwidth='low').to(self.device)
        self.decoder_sc1 = SemanticDecoder(bandwidth='low').to(self.device)
        if os.path.exists(sc1_path):
            checkpoint = torch.load(sc1_path, map_location=self.device, weights_only=False)
            self.encoder_sc1.load_state_dict(checkpoint['encoder'])
            self.decoder_sc1.load_state_dict(checkpoint['decoder'])
            print(f"✅ Loaded SC1 from {sc1_path}")
        else:
            print(f"⚠️ SC1 checkpoint not found: {sc1_path} – using random weights")
        self.encoder_sc1.eval()
        self.decoder_sc1.eval()

        # SC2 (high bandwidth)
        self.encoder_sc2 = SemanticEncoder(bandwidth='high').to(self.device)
        self.decoder_sc2 = SemanticDecoder(bandwidth='high').to(self.device)
        if os.path.exists(sc2_path):
            checkpoint = torch.load(sc2_path, map_location=self.device, weights_only=False)
            self.encoder_sc2.load_state_dict(checkpoint['encoder'])
            self.decoder_sc2.load_state_dict(checkpoint['decoder'])
            print(f"✅ Loaded SC2 from {sc2_path}")
        else:
            print(f"⚠️ SC2 checkpoint not found: {sc2_path} – using random weights")
        self.encoder_sc2.eval()
        self.decoder_sc2.eval()

        # Compute black encoding baselines for compression
        with torch.no_grad():
            black = torch.zeros(1, 3, *IMAGE_SIZE).to(self.device)
            self.n_value_roni = torch.mean(self.encoder_sc1(black)).item()
            self.n_value_roi  = torch.mean(self.encoder_sc2(black)).item()
        print(f"[Baseline] SC1 black mean = {self.n_value_roni:.4f}, SC2 black mean = {self.n_value_roi:.4f}")

    @torch.no_grad()
    def process(self, image_tensor, channel_type='awgn', snr_db=10, theta=0.8):
        """Run the full semantic communication pipeline on a single image tensor."""
        # Segmentation
        seg_logits = self.unet(image_tensor.unsqueeze(0))
        seg_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()  # shape [H,W]
        roi_mask = (seg_mask == 1).astype(np.uint8)
        roni_mask = (seg_mask == 0).astype(np.uint8)

        # Convert image to numpy for masking
        image_np = image_tensor.permute(1,2,0).cpu().numpy()  # [H,W,3]
        roi_image_np = image_np * roi_mask[..., np.newaxis]
        roni_image_np = image_np * roni_mask[..., np.newaxis]

        # Create tensors for encoding
        roi_tensor = torch.from_numpy(roi_image_np).permute(2,0,1).float().unsqueeze(0).to(self.device)
        roni_tensor = torch.from_numpy(roni_image_np).permute(2,0,1).float().unsqueeze(0).to(self.device)

        # Encode
        enc_roi  = self.encoder_sc2(roi_tensor)
        enc_roni = self.encoder_sc1(roni_tensor)

        # Compression (using black baselines)
        comp_roni = compress_sparse_matrix(enc_roni.squeeze().cpu().numpy(),
                                           self.n_value_roni, tol=0.01)
        comp_roi = compress_sparse_matrix(enc_roi.squeeze().cpu().numpy(),
                                          self.n_value_roi, tol=0.01)

        # Channel simulation
        channel = AWGNChannel(snr_db) if channel_type == 'awgn' else RayleighChannel(snr_db)
        tx_roni = channel.transmit(comp_roni)
        tx_roi  = channel.transmit(comp_roi)

        # Restoration
        rest_roni = restore_sparse_matrix(tx_roni, self.n_value_roni, enc_roni.shape[1:])
        rest_roi  = restore_sparse_matrix(tx_roi,  self.n_value_roi, enc_roi.shape[1:])

        rest_roni_t = torch.from_numpy(rest_roni.astype(np.float32)).unsqueeze(0).to(self.device)
        rest_roi_t  = torch.from_numpy(rest_roi.astype(np.float32)).unsqueeze(0).to(self.device)

        # Decode
        dec_roni = self.decoder_sc1(rest_roni_t)
        dec_roi  = self.decoder_sc2(rest_roi_t)

        # Combine
        dec_roni_np = dec_roni.squeeze().permute(1,2,0).cpu().numpy()
        dec_roi_np  = dec_roi.squeeze().permute(1,2,0).cpu().numpy()
        roi_mask_3d = roi_mask[..., np.newaxis]
        roni_mask_3d = roni_mask[..., np.newaxis]
        combined = roi_mask_3d * dec_roi_np + roni_mask_3d * dec_roni_np
        combined = np.clip(combined, 0, 1)

        # Compute metrics
        psnr_val = calculate_psnr(image_np, combined)
        theta_psnr_val = calculate_theta_psnr(image_np, combined, roi_mask, theta)

        # SSIM as percentage
        ssim_val = ssim(image_np, combined, channel_axis=-1, data_range=1.0)
        ssim_percent = ssim_val * 100

        # Compression ratio
        orig_size = image_np.size * image_np.itemsize
        comp_size = (len(comp_roni[0]) + len(comp_roi[0])) * 4 + (len(comp_roni[1]) + len(comp_roi[1])) * 4
        comp_ratio = orig_size / comp_size if comp_size > 0 else 0

        return {
            'psnr': psnr_val,
            'theta_psnr': theta_psnr_val,
            'ssim_percent': ssim_percent,
            'compression_ratio': comp_ratio,
            'seg_mask': seg_mask,
            'reconstructed': combined,
            'roi_percentage': np.sum(roi_mask) / roi_mask.size * 100
        }

# ------------------------- Dataset Loader (VOC2012 without split files) -------------------------
class VOCSegmentationDataset(Dataset):
    """Load VOC2012 validation images and segmentation masks without requiring split files."""
    def __init__(self, root_dir, split='val', image_size=IMAGE_SIZE, val_fraction=0.2, seed=42):
        self.root_dir = root_dir
        self.image_size = image_size
        self.split = split

        # Paths
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClass')

        # Get all image IDs (without extension) from JPEGImages
        all_ids = []
        if os.path.exists(self.image_dir):
            for f in os.listdir(self.image_dir):
                if f.lower().endswith('.jpg'):
                    all_ids.append(os.path.splitext(f)[0])

        # Filter IDs that have a corresponding mask
        self.valid_ids = []
        for img_id in all_ids:
            mask_path = os.path.join(self.mask_dir, f'{img_id}.png')
            if os.path.exists(mask_path):
                self.valid_ids.append(img_id)

        print(f"Found {len(self.valid_ids)} images with masks")

        # If no valid images found, raise error
        if len(self.valid_ids) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {root_dir}. Check paths.")

        # Create train/val split
        random.seed(seed)
        random.shuffle(self.valid_ids)
        split_idx = int(len(self.valid_ids) * (1 - val_fraction))
        if split == 'train':
            self.valid_ids = self.valid_ids[:split_idx]
        elif split == 'val':
            self.valid_ids = self.valid_ids[split_idx:]
        else:
            # If split is something else, use all (e.g., 'trainval')
            pass

        print(f"Using {len(self.valid_ids)} images for {split} split")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        mask_path = os.path.join(self.mask_dir, f'{img_id}.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_np).long()
        return image_tensor, mask_tensor, img_id

# ------------------------- Evaluation Function -------------------------
def evaluate(system, dataloader, channel_type='awgn', snr_db=10, theta=0.8):
    results = []
    for images, masks, img_ids in tqdm(dataloader, desc="Evaluating"):
        images = images.to(DEVICE)
        for i in range(images.size(0)):
            img_tensor = images[i]
            gt_mask = masks[i].numpy()
            img_id = img_ids[i]

            out = system.process(img_tensor, channel_type, snr_db, theta)

            # IoU (person class = 15 in VOC)
            pred_mask = out['seg_mask']
            gt_binary = (gt_mask == 15).astype(np.uint8)
            iou = calculate_iou(pred_mask, gt_binary)

            results.append({
                'image_id': img_id,
                'psnr': out['psnr'],
                'theta_psnr': out['theta_psnr'],
                'ssim_percent': out['ssim_percent'],
                'compression_ratio': out['compression_ratio'],
                'roi_percentage': out['roi_percentage'],
                'iou': iou
            })
    return pd.DataFrame(results)

# ------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./VOC2012',
                        help='Path to VOC2012 dataset')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'trainval'], help='Dataset split')
    parser.add_argument('--channel', type=str, default='awgn',
                        choices=['awgn', 'rayleigh'], help='Channel type')
    parser.add_argument('--snr', type=float, default=10.0, help='SNR in dB')
    parser.add_argument('--theta', type=float, default=0.8, help='θ for θ-PSNR')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--unet', type=str, default='checkpoints/unet_final.pth',
                        help='Path to U-Net checkpoint')
    parser.add_argument('--sc1', type=str, default='checkpoints/sc1_final.pth',
                        help='Path to SC1 combined checkpoint')
    parser.add_argument('--sc2', type=str, default='checkpoints/sc2_final.pth',
                        help='Path to SC2 combined checkpoint')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                        help='Output CSV file')
    args = parser.parse_args()

    print("="*60)
    print("Semantic Communication System Evaluation")
    print(f"Device: {DEVICE}")
    print(f"Data dir: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Channel: {args.channel}, SNR={args.snr} dB, θ={args.theta}")
    print("="*60)

    system = SemanticSystem(unet_path=args.unet, sc1_path=args.sc1, sc2_path=args.sc2)
    dataset = VOCSegmentationDataset(args.data_dir, split=args.split, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    df = evaluate(system, dataloader, args.channel, args.snr, args.theta)

    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Number of images: {len(df)}")
    print(f"PSNR (mean ± std): {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
    print(f"θ-PSNR (mean ± std): {df['theta_psnr'].mean():.2f} ± {df['theta_psnr'].std():.2f} dB")
    print(f"SSIM (mean ± std): {df['ssim_percent'].mean():.2f} ± {df['ssim_percent'].std():.2f} %")
    print(f"Compression ratio (mean ± std): {df['compression_ratio'].mean():.2f} ± {df['compression_ratio'].std():.2f}")
    print(f"IoU (mean ± std): {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
    print(f"ROI percentage (mean): {df['roi_percentage'].mean():.2f}%")

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Plot histograms including SSIM
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0,0].hist(df['psnr'], bins=20, color='blue', alpha=0.7)
    axes[0,0].set_title('PSNR Distribution')
    axes[0,0].set_xlabel('PSNR (dB)')
    axes[0,1].hist(df['theta_psnr'], bins=20, color='red', alpha=0.7)
    axes[0,1].set_title('θ-PSNR Distribution')
    axes[0,1].set_xlabel('θ-PSNR (dB)')
    axes[0,2].hist(df['ssim_percent'], bins=20, color='purple', alpha=0.7)
    axes[0,2].set_title('SSIM Distribution (%)')
    axes[0,2].set_xlabel('SSIM (%)')
    axes[1,0].hist(df['compression_ratio'], bins=20, color='green', alpha=0.7)
    axes[1,0].set_title('Compression Ratio Distribution')
    axes[1,0].set_xlabel('Ratio')
    axes[1,1].hist(df['iou'], bins=20, color='orange', alpha=0.7)
    axes[1,1].set_title('IoU Distribution')
    axes[1,1].set_xlabel('IoU')
    axes[1,2].axis('off')
    plt.tight_layout()
    plt.savefig('evaluation_histograms.png', dpi=150)
    plt.show()
    print("Histograms saved to evaluation_histograms.png")

if __name__ == '__main__':
    main()