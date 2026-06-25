# train.py (Step 1: Improved U‑Net training)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import glob
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.unet import UNet
from models.semantic_encoder import SemanticEncoder
from models.semantic_decoder import SemanticDecoder
from datasets import create_dataloaders

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.writer = SummaryWriter(log_dir=config['log_dir'])

        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)

        print("Loading datasets...")
        self.dataloaders = create_dataloaders(batch_size=config['batch_size'])

        self.init_models()
        self.init_optimizers()

        self.best_val_loss = float('inf')
        self.train_history = {
            'unet': {'train_loss': [], 'val_loss': []},
            'sc1': {'train_loss': [], 'val_loss': []},
            'sc2': {'train_loss': [], 'val_loss': []}
        }
        self.accumulation_steps = config.get('accumulation_steps', 4)
        self.start_epoch_unet = 0
        self.start_epoch_sc1 = 0
        self.start_epoch_sc2 = 0

    def init_models(self):
        self.unet = UNet(n_channels=3, n_classes=2).to(self.device)
        self.encoder_sc1 = SemanticEncoder(bandwidth='low').to(self.device)
        self.decoder_sc1 = SemanticDecoder(bandwidth='low').to(self.device)
        self.encoder_sc2 = SemanticEncoder(bandwidth='high').to(self.device)
        self.decoder_sc2 = SemanticDecoder(bandwidth='high').to(self.device)

    def init_optimizers(self):
        self.unet_optimizer = optim.Adam(self.unet.parameters(), lr=self.config['lr_unet'])
        self.unet_scheduler = optim.lr_scheduler.StepLR(self.unet_optimizer, step_size=20, gamma=0.5)

        sc1_params = list(self.encoder_sc1.parameters()) + list(self.decoder_sc1.parameters())
        self.sc1_optimizer = optim.Adam(sc1_params, lr=self.config['lr_sc'])
        self.sc1_scheduler = optim.lr_scheduler.StepLR(self.sc1_optimizer, step_size=100, gamma=0.5)

        sc2_params = list(self.encoder_sc2.parameters()) + list(self.decoder_sc2.parameters())
        self.sc2_optimizer = optim.Adam(sc2_params, lr=self.config['lr_sc'])
        self.sc2_scheduler = optim.lr_scheduler.StepLR(self.sc2_optimizer, step_size=100, gamma=0.5)

        # Class weights for U‑Net (background=1.0, person=config['unet_person_weight'])
        person_weight = self.config.get('unet_person_weight', 2.0)
        self.segmentation_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, person_weight]).to(self.device)
        )
        weight_tensor = torch.tensor([1.0, float(person_weight)]).to(self.device)
        self.segmentation_loss = nn.CrossEntropyLoss(weight=weight_tensor)
        
        self.reconstruction_loss = nn.MSELoss()

    # ---------- Helper to find latest checkpoint ----------
    def find_latest_checkpoint(self, model_type='unet'):
        pattern = os.path.join(self.config['checkpoint_dir'], f'{model_type}_epoch_*_loss_*.pth')
        files = glob.glob(pattern)
        if not files:
            return None
        # Extract epoch numbers
        epochs = []
        for f in files:
            match = re.search(rf'{model_type}_epoch_(\d+)_loss_', f)
            if match:
                epochs.append(int(match.group(1)))
            else:
                epochs.append(0)
        max_idx = np.argmax(epochs)
        return files[max_idx]

    def load_unet_checkpoint(self, checkpoint_path):
        print(f"Loading U‑Net checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.unet.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer' in checkpoint:
                self.unet_optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.unet_scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                self.start_epoch_unet = checkpoint['epoch'] + 1
            if 'loss' in checkpoint:
                print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        else:
            self.unet.load_state_dict(checkpoint)
            print("Loaded U‑Net state_dict (no optimizer state).")
        print(f"Resuming U‑Net from epoch {self.start_epoch_unet}")

    def load_sc1_checkpoint(self, checkpoint_path):
        print(f"Loading SC1 checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.encoder_sc1.load_state_dict(checkpoint['encoder'])
        self.decoder_sc1.load_state_dict(checkpoint['decoder'])
        if 'optimizer' in checkpoint:
            self.sc1_optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.sc1_scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            self.start_epoch_sc1 = checkpoint['epoch'] + 1
        if 'loss' in checkpoint:
            print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        print(f"Resuming SC1 from epoch {self.start_epoch_sc1}")

    def load_sc2_checkpoint(self, checkpoint_path):
        print(f"Loading SC2 checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.encoder_sc2.load_state_dict(checkpoint['encoder'])
        self.decoder_sc2.load_state_dict(checkpoint['decoder'])
        if 'optimizer' in checkpoint:
            self.sc2_optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.sc2_scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            self.start_epoch_sc2 = checkpoint['epoch'] + 1
        if 'loss' in checkpoint:
            print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        print(f"Resuming SC2 from epoch {self.start_epoch_sc2}")

    # ---------- Training methods ----------
    def train_unet(self):
        print("\n" + "="*50)
        print("Stage 1: Training U‑Net for Semantic Segmentation")
        print("="*50)

        latest = self.find_latest_checkpoint('unet')
        if latest:
            self.load_unet_checkpoint(latest)
        else:
            print("No U‑Net checkpoint found, starting from scratch.")

        self.unet.train()

        for epoch in range(self.start_epoch_unet, self.config['epochs_unet']):
            train_loss = 0.0
            progress_bar = tqdm(self.dataloaders['train'], desc=f'U‑Net Epoch {epoch+1}/{self.config["epochs_unet"]}')

            for batch in progress_bar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                self.unet_optimizer.zero_grad()
                outputs = self.unet(images)
                loss = self.segmentation_loss(outputs, masks)
                loss.backward()
                self.unet_optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            val_loss = self.validate_unet()
            self.unet_scheduler.step()

            avg_train_loss = train_loss / len(self.dataloaders['train'])
            self.train_history['unet']['train_loss'].append(avg_train_loss)
            self.train_history['unet']['val_loss'].append(val_loss)

            self.writer.add_scalar('UNet/Train_Loss', avg_train_loss, epoch)
            self.writer.add_scalar('UNet/Val_Loss', val_loss, epoch)

            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('unet', epoch, val_loss, is_best=True)
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint('unet', epoch, val_loss)

        print("U‑Net training complete!")

    def validate_unet(self):
        self.unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                outputs = self.unet(images)
                loss = self.segmentation_loss(outputs, masks)
                val_loss += loss.item()
        self.unet.train()
        torch.cuda.empty_cache()
        return val_loss / len(self.dataloaders['val'])

    def train_sc1(self):
        print("\n" + "="*50)
        print("Stage 2: Training SC1 (Low Bandwidth for RONI)")
        print("="*50)

        latest = self.find_latest_checkpoint('sc1')
        if latest:
            self.load_sc1_checkpoint(latest)
        else:
            print("No SC1 checkpoint found, starting from scratch.")

        self.encoder_sc1.train()
        self.decoder_sc1.train()

        l1_lambda = self.config.get('l1_lambda_sc1', 1e-3)
        print(f"Using L1 lambda = {l1_lambda}")

        for epoch in range(self.start_epoch_sc1, self.config['epochs_sc']):
            train_loss = 0.0
            progress_bar = tqdm(self.dataloaders['roni_train'], desc=f'SC1 Epoch {epoch+1}/{self.config["epochs_sc"]}')

            for roni_images, targets in progress_bar:
                roni_images = roni_images.to(self.device)
                targets = targets.to(self.device)

                self.sc1_optimizer.zero_grad()
                encoded = self.encoder_sc1(roni_images)
                decoded = self.decoder_sc1(encoded)

                recon_loss = self.reconstruction_loss(decoded, targets)
                l1_loss = l1_lambda * torch.mean(torch.abs(encoded))
                loss = recon_loss + l1_loss
                loss.backward()
                self.sc1_optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            self.sc1_scheduler.step()
            avg_train_loss = train_loss / len(self.dataloaders['roni_train'])
            self.train_history['sc1']['train_loss'].append(avg_train_loss)
            self.writer.add_scalar('SC1/Train_Loss', avg_train_loss, epoch)

            if epoch % 10 == 0:
                with torch.no_grad():
                    black = torch.zeros(1, 3, 512, 512).to(self.device)
                    black_enc = self.encoder_sc1(black)
                    print(f"\n[Black encoding] epoch {epoch+1}: min={black_enc.min():.3f}, max={black_enc.max():.3f}, mean={black_enc.mean():.3f}, range={black_enc.max()-black_enc.min():.3f}")

            print(f'SC1 Epoch {epoch+1}: Loss = {avg_train_loss:.6f}')

            if (epoch + 1) % 50 == 0 or (epoch + 1) == self.config['epochs_sc']:
                self.save_checkpoint('sc1', epoch, avg_train_loss)

        print("SC1 training complete!")

    def train_sc2(self):
        print("\n" + "="*50)
        print("Stage 3: Training SC2 (High Bandwidth for ROI)")
        print("="*50)

        latest = self.find_latest_checkpoint('sc2')
        if latest:
            self.load_sc2_checkpoint(latest)
        else:
            print("No SC2 checkpoint found, starting from scratch.")

        self.encoder_sc2.train()
        self.decoder_sc2.train()

        l1_lambda = self.config.get('l1_lambda_sc2', 0.0)
        if l1_lambda > 0:
            print(f"Using L1 lambda for SC2 = {l1_lambda}")

        for epoch in range(self.start_epoch_sc2, self.config['epochs_sc']):
            train_loss = 0.0
            progress_bar = tqdm(self.dataloaders['roi_train'], desc=f'SC2 Epoch {epoch+1}/{self.config["epochs_sc"]}')

            for roi_images, targets in progress_bar:
                roi_images = roi_images.to(self.device)
                targets = targets.to(self.device)

                self.sc2_optimizer.zero_grad()
                encoded = self.encoder_sc2(roi_images)
                decoded = self.decoder_sc2(encoded)

                recon_loss = self.reconstruction_loss(decoded, targets)
                if l1_lambda > 0:
                    l1_loss = l1_lambda * torch.mean(torch.abs(encoded))
                    loss = recon_loss + l1_loss
                else:
                    loss = recon_loss
                loss.backward()
                self.sc2_optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            self.sc2_scheduler.step()
            avg_train_loss = train_loss / len(self.dataloaders['roi_train'])
            self.train_history['sc2']['train_loss'].append(avg_train_loss)
            self.writer.add_scalar('SC2/Train_Loss', avg_train_loss, epoch)

            print(f'SC2 Epoch {epoch+1}: Loss = {avg_train_loss:.6f}')

            if (epoch + 1) % 50 == 0 or (epoch + 1) == self.config['epochs_sc']:
                self.save_checkpoint('sc2', epoch, avg_train_loss)

        print("SC2 training complete!")

    def save_checkpoint(self, model_type, epoch, loss, is_best=False):
        suffix = '_best' if is_best else ''
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            f'{model_type}_epoch_{epoch+1}_loss_{loss:.4f}{suffix}.pth'
        )
        if model_type == 'unet':
            torch.save({
                'model_state_dict': self.unet.state_dict(),
                'optimizer': self.unet_optimizer.state_dict(),
                'scheduler': self.unet_scheduler.state_dict(),
                'epoch': epoch,
                'loss': loss,
            }, checkpoint_path)
        elif model_type == 'sc1':
            torch.save({
                'encoder': self.encoder_sc1.state_dict(),
                'decoder': self.decoder_sc1.state_dict(),
                'optimizer': self.sc1_optimizer.state_dict(),
                'scheduler': self.sc1_scheduler.state_dict(),
                'epoch': epoch,
                'loss': loss,
            }, checkpoint_path)
        elif model_type == 'sc2':
            torch.save({
                'encoder': self.encoder_sc2.state_dict(),
                'decoder': self.decoder_sc2.state_dict(),
                'optimizer': self.sc2_optimizer.state_dict(),
                'scheduler': self.sc2_scheduler.state_dict(),
                'epoch': epoch,
                'loss': loss,
            }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def save_final_models(self):
        torch.save(self.unet.state_dict(), 'checkpoints/unet_final.pth')
        torch.save(self.encoder_sc1.state_dict(), 'checkpoints/encoder_sc1_final.pth')
        torch.save(self.decoder_sc1.state_dict(), 'checkpoints/decoder_sc1_final.pth')
        torch.save(self.encoder_sc2.state_dict(), 'checkpoints/encoder_sc2_final.pth')
        torch.save(self.decoder_sc2.state_dict(), 'checkpoints/decoder_sc2_final.pth')

        history_path = os.path.join(self.config['log_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print("All models saved to checkpoints/")
        print(f"Training history saved to {history_path}")

    def plot_training_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(self.train_history['unet']['train_loss'], label='Train')
        axes[0].plot(self.train_history['unet']['val_loss'], label='Validation')
        axes[0].set_title('U-Net Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.train_history['sc1']['train_loss'], label='Train', color='orange')
        axes[1].set_title('SC1 Training Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(self.train_history['sc2']['train_loss'], label='Train', color='green')
        axes[2].set_title('SC2 Training Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['log_dir'], 'training_history.png'), dpi=150)
        plt.close()
        print(f"Training plot saved to {self.config['log_dir']}/training_history.png")

def main():
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 2,
        'accumulation_steps': 4,
        'epochs_unet': 150,                # Increased to 150
        'epochs_sc': 800,                   # Keep for now, will update later
        'lr_unet': 0.0005,                  # Lower learning rate for fine-tuning
        'lr_sc': 0.01,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'data_dir': 'dataset/processed',
        'l1_lambda_sc1': 0.001,
        'l1_lambda_sc2': 0.0,
        'unet_person_weight': 10.0,         # Much higher weight for person class
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    trainer = Trainer(config)

    # Uncomment only U‑Net training for this step
    trainer.train_unet()
    # trainer.train_sc1()
    # trainer.train_sc2()

    trainer.save_final_models()
    trainer.plot_training_history()

if __name__ == "__main__":
    main()