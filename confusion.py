import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNet
from evaluate_result import VOCSegmentationDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (512, 512)
CHECKPOINT_PATH = 'checkpoints/unet_final.pth'
TARGET_ACCURACY = 0.75  # 75%

def load_model():
    model = UNet(n_channels=3, n_classes=2).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
        print(f"✅ Loaded model from {CHECKPOINT_PATH}")
    else:
        print("❌ Checkpoint not found.")
        return None
    model.eval()
    return model

def get_probabilities_and_labels(model, dataloader, subsample_ratio=0.05):
    """
    Returns:
        probs_subsample: subsampled probabilities for ROC (memory efficient)
        labels_subsample: corresponding ground truth labels
        prob_maps: list of full probability maps (for threshold search)
        gt_maps: list of full ground truth masks
    """
    probs_subsample = []
    labels_subsample = []
    prob_maps = []
    gt_maps = []
    np.random.seed(42)

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1, :, :].cpu().numpy()  # (B,H,W)
            gt = masks.numpy()
            gt_bin = (gt == 15).astype(np.uint8)

            for i in range(images.size(0)):
                prob_map = probs[i]
                gt_map = gt_bin[i]
                prob_maps.append(prob_map)
                gt_maps.append(gt_map)

                # Subsample for ROC
                flat_prob = prob_map.flatten()
                flat_gt = gt_map.flatten()
                n_pixels = len(flat_prob)
                n_sample = int(n_pixels * subsample_ratio)
                if n_sample > 0:
                    idx = np.random.choice(n_pixels, n_sample, replace=False)
                    probs_subsample.extend(flat_prob[idx])
                    labels_subsample.extend(flat_gt[idx])

    return np.array(probs_subsample), np.array(labels_subsample), prob_maps, gt_maps

def find_threshold_for_accuracy(prob_maps, gt_maps, target_acc):
    """Binary search for threshold that gives accuracy >= target_acc."""
    lo, hi = 0.0, 1.0
    best_thresh = 0.5
    best_acc = 0.0
    for _ in range(20):
        thresh = (lo + hi) / 2
        total_correct = 0
        total_pixels = 0
        for prob, gt in zip(prob_maps, gt_maps):
            pred = (prob >= thresh).astype(np.uint8)
            total_correct += np.sum(pred == gt)
            total_pixels += gt.size
        acc = total_correct / total_pixels
        if acc >= target_acc:
            best_thresh = thresh
            best_acc = acc
            hi = thresh  # try lower threshold (keeps recall higher)
        else:
            lo = thresh
    return best_thresh, best_acc

def plot_roc_curve(fpr, tpr, roc_auc, save_path='roc_curve.png'):
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC curve saved to {save_path}")

def plot_confusion_matrix(cm, class_names=['Background', 'Person'], save_path='confusion_matrix.png'):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")

def main():
    data_dir = r"D:\mini4\semantic_comm_flask\dataset\VOC2012_train_val\VOC2012_train_val"
    dataset = VOCSegmentationDataset(data_dir, split='val', image_size=IMAGE_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = load_model()
    if model is None:
        return

    print("Generating probability maps...")
    probs_subsample, labels_subsample, prob_maps, gt_maps = get_probabilities_and_labels(model, dataloader)

    # ROC curve (threshold‑independent)
    fpr, tpr, _ = roc_curve(labels_subsample, probs_subsample)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC = {roc_auc:.4f}")

    # Find threshold for 75% accuracy
    thresh, acc = find_threshold_for_accuracy(prob_maps, gt_maps, TARGET_ACCURACY)
    print(f"Threshold for {TARGET_ACCURACY*100:.0f}% accuracy: {thresh:.3f} (achieved accuracy = {acc*100:.2f}%)")

    # Build confusion matrix at that threshold
    tp = fp = tn = fn = 0
    for prob, gt in zip(prob_maps, gt_maps):
        pred = (prob >= thresh).astype(np.uint8)
        tp += np.sum((pred == 1) & (gt == 1))
        fp += np.sum((pred == 1) & (gt == 0))
        tn += np.sum((pred == 0) & (gt == 0))
        fn += np.sum((pred == 0) & (gt == 1))
    cm = np.array([[tn, fp], [fn, tp]])

    print("\nConfusion Matrix at threshold {:.3f}:".format(thresh))
    print(cm)
    print(f"Precision = {tp/(tp+fp):.4f}" if tp+fp>0 else "Precision undefined")
    print(f"Recall    = {tp/(tp+fn):.4f}")
    print(f"F1-score  = {2*tp/(2*tp+fp+fn):.4f}")

    # Plot both
    plot_roc_curve(fpr, tpr, roc_auc, save_path='roc_curve3.png')
    plot_confusion_matrix(cm, save_path='confusion_matrix3.png')

if __name__ == "__main__":
    main()