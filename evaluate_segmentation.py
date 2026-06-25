import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNet
from evaluate_result import VOCSegmentationDataset

# Try to import CRF, but continue if not available
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("pydensecrf not installed – CRF post‑processing disabled.")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (512, 512)

# You can add multiple checkpoint paths for ensemble
CHECKPOINT_PATHS = ['checkpoints/unet_final.pth']

def load_models(checkpoint_paths):
    models = []
    for path in checkpoint_paths:
        model = UNet(n_channels=3, n_classes=2).to(DEVICE)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            print(f"✅ Loaded model from {path}")
        else:
            print(f"❌ Checkpoint not found: {path}")
            return None
        model.eval()
        models.append(model)
    return models

def tta_predict(models, image_tensor, use_tta=True):
    """
    Apply test‑time augmentation and ensemble.
    Returns probability map for class 1 (person) as numpy array (H, W).
    """
    if not isinstance(models, list):
        models = [models]

    # Define augmentations and their inverses
    transforms = [
        ('none', lambda x: x, lambda x: x),
        ('hflip', lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        ('vflip', lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        ('rot90', lambda x: torch.rot90(x, k=1, dims=[2,3]), lambda x: torch.rot90(x, k=-1, dims=[2,3])),
    ]

    probs_list = []
    with torch.no_grad():
        for name, aug, inv in transforms:
            aug_img = aug(image_tensor).to(DEVICE)
            # Ensemble over models
            model_probs = []
            for model in models:
                logits = model(aug_img)
                prob = torch.softmax(logits, dim=1)          # (1,2,H,W)
                model_probs.append(prob.cpu())
            avg_prob = torch.stack(model_probs).mean(dim=0)   # (1,2,H,W)
            # Apply inverse transform
            avg_prob = inv(avg_prob)
            probs_list.append(avg_prob)

    # Average over all augmentations
    final_prob = torch.stack(probs_list).mean(dim=0)          # (1,2,H,W)
    return final_prob[0, 1, :, :].numpy()                    # (H,W)

def apply_crf(original_np, prob_map, theta=10):
    """Refine probability map with CRF (if available)."""
    if not CRF_AVAILABLE:
        return prob_map
    H, W = prob_map.shape
    unary = np.stack([1 - prob_map, prob_map], axis=0)   # (2, H, W)
    unary = -np.log(unary + 1e-8)
    unary = unary.reshape(2, -1).astype(np.float32)

    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=theta, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=original_np, compat=10)
    Q = d.inference(5)
    return np.array(Q).reshape(2, H, W)[1]

def evaluate_advanced(models, dataloader, use_tta=True, use_crf=False, subsample_ratio=0.05):
    """
    Two‑pass evaluation: first subsample to find optimal threshold,
    then compute confusion matrix at that threshold using full pixel counts.
    """
    # ----- Pass 1: subsample probabilities for ROC and threshold optimisation -----
    all_probs = []
    all_labels = []
    np.random.seed(42)

    for images, masks, _ in tqdm(dataloader, desc="Pass 1 (subsampling)"):
        images_np = images.permute(0,2,3,1).cpu().numpy() * 255
        images_np = np.clip(images_np, 0, 255).astype(np.uint8)

        for i in range(images.size(0)):
            img_tensor = images[i].unsqueeze(0)
            orig_np = images_np[i]
            gt_mask = masks[i].numpy()
            gt_binary = (gt_mask == 15).astype(np.uint8)

            prob_map = tta_predict(models, img_tensor, use_tta)
            if use_crf:
                prob_map = apply_crf(orig_np, prob_map)

            flat_prob = prob_map.flatten()
            flat_gt = gt_binary.flatten()
            n_pixels = len(flat_prob)
            n_sample = int(n_pixels * subsample_ratio)
            if n_sample > 0:
                idx = np.random.choice(n_pixels, n_sample, replace=False)
                all_probs.extend(flat_prob[idx])
                all_labels.extend(flat_gt[idx])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Find best threshold (maximise F1)
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(np.uint8)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"\nOptimal threshold (max F1 on subsample): {best_thresh:.2f} (F1 = {best_f1:.4f})")

    # ----- Pass 2: compute confusion matrix at best threshold using all pixels -----
    tp = fp = tn = fn = 0
    for images, masks, _ in tqdm(dataloader, desc="Pass 2 (full confusion matrix)"):
        images_np = images.permute(0,2,3,1).cpu().numpy() * 255
        images_np = np.clip(images_np, 0, 255).astype(np.uint8)

        for i in range(images.size(0)):
            img_tensor = images[i].unsqueeze(0)
            orig_np = images_np[i]
            gt_mask = masks[i].numpy()
            gt_binary = (gt_mask == 15).astype(np.uint8)

            prob_map = tta_predict(models, img_tensor, use_tta)
            if use_crf:
                prob_map = apply_crf(orig_np, prob_map)

            preds = (prob_map >= best_thresh).astype(np.uint8)
            tp += np.sum((preds == 1) & (gt_binary == 1))
            fp += np.sum((preds == 1) & (gt_binary == 0))
            tn += np.sum((preds == 0) & (gt_binary == 0))
            fn += np.sum((preds == 0) & (gt_binary == 1))

    cm = np.array([[tn, fp], [fn, tp]])
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # ROC curve from subsampled data
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    return cm, best_thresh, accuracy, fpr, tpr, roc_auc

def plot_confusion_matrix(cm, class_names=['Background', 'Person'],
                          normalize=False, title='Confusion Matrix',
                          cmap=plt.cm.Blues, save_path='confusion_matrix.png'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(fpr, tpr, roc_auc, save_path='roc_curve.png'):
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC curve saved to {save_path}")

def main():
    data_dir = r"D:\mini4\semantic_comm_flask\dataset\VOC2012_train_val\VOC2012_train_val"
    print("Loading dataset...")
    dataset = VOCSegmentationDataset(data_dir, split='val', image_size=IMAGE_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Loading models...")
    models = load_models(CHECKPOINT_PATHS)
    if models is None:
        return

    # Set flags (CRF will be disabled automatically if not installed)
    use_tta = True
    use_crf = False   # set to True only if you have CRF installed and want it

    print("Running advanced evaluation (TTA + threshold optimisation) ...")
    cm, best_thresh, accuracy, fpr, tpr, roc_auc = evaluate_advanced(
        models, dataloader, use_tta=use_tta, use_crf=use_crf, subsample_ratio=0.05
    )

    print("\n" + "="*50)
    print("Segmentation Evaluation Results (with post‑processing)")
    print("="*50)
    print(f"Optimal threshold: {best_thresh:.2f}")
    print("Confusion Matrix (rows=true, cols=predicted):")
    print("               Predicted")
    print("               BG    Person")
    print(f"True BG      {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"True Person  {cm[1,0]:6d}  {cm[1,1]:6d}")
    print(f"\nOverall Pixel Accuracy: {accuracy*100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp+fp)>0 else 0
    recall = tp / (tp + fn) if (tp+fn)>0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall)>0 else 0
    print(f"Precision (person): {precision:.4f}")
    print(f"Recall (person): {recall:.4f}")
    print(f"F1-score (person): {f1:.4f}")

    # Save plots
    plot_confusion_matrix(cm, class_names=['Background', 'Person'],
                          normalize=False, title='Confusion Matrix (optimized)',
                          save_path='confusion_matrix_optimized.png')
    plot_roc_curve(fpr, tpr, roc_auc, save_path='roc_curve_optimized.png')

if __name__ == "__main__":
    main()