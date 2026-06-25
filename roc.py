import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from models.unet import UNet
from evaluate_result import VOCSegmentationDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (512, 512)
CHECKPOINT_PATH = 'checkpoints/unet_final.pth'
DATA_DIR = r"D:\mini4\semantic_comm_flask\dataset\VOC2012_train_val\VOC2012_train_val"

def load_model():
    model = UNet(n_channels=3, n_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def get_predictions(model, dataloader):
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1, :, :].cpu().numpy()
            for i in range(images.size(0)):
                prob = probs[i].flatten()
                gt = (masks[i].numpy() == 15).astype(np.uint8).flatten()
                # subsample to avoid memory issues (optional)
                idx = np.random.choice(len(prob), int(len(prob)*0.05), replace=False)
                all_probs.extend(prob[idx])
                all_labels.extend(gt[idx])
    return np.array(all_probs), np.array(all_labels)

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("ROC curve saved as roc_curve.png")

def main():
    dataset = VOCSegmentationDataset(DATA_DIR, split='val', image_size=IMAGE_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = load_model()
    probs, labels = get_predictions(model, dataloader)
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC = {roc_auc:.4f}")
    plot_roc_curve(fpr, tpr, roc_auc)

if __name__ == "__main__":
    main()