# preprocess_voc2012.py
import os
import shutil
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET

class VOC2012Preprocessor:
    def __init__(self, data_root="dataset/VOC2012_train_val/VOC2012_train_val"):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, "JPEGImages")
        self.masks_dir = os.path.join(data_root, "SegmentationClass")
        self.trainval_txt = os.path.join(data_root, "ImageSets", "Segmentation", "trainval.txt")
        self.train_txt = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
        self.val_txt = os.path.join(data_root, "ImageSets", "Segmentation", "val.txt")
        
    def get_person_images(self):
        """Get images containing person class (class 15)"""
        print("Identifying images with person class...")
        
        # Read all image names
        with open(self.trainval_txt, 'r') as f:
            all_images = [line.strip() for line in f]
        
        person_images = []
        
        for img_name in tqdm(all_images):
            mask_path = os.path.join(self.masks_dir, f"{img_name}.png")
            if not os.path.exists(mask_path):
                continue
            
            # Load mask
            mask = np.array(Image.open(mask_path))
            
            # Check if person class (15) exists in mask
            if 15 in mask:
                person_images.append(img_name)
        
        print(f"Found {len(person_images)} images with person class out of {len(all_images)} total images")
        return person_images
    
    def create_binary_masks(self, output_dir="dataset/processed"):
        """Create binary masks (person = 1, background = 0)"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "roi_masks"), exist_ok=True)
        
        person_images = self.get_person_images()
        
        print("Creating binary masks...")
        for img_name in tqdm(person_images):
            # Load and resize image
            img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
            image = Image.open(img_path).convert('RGB')
            image_resized = image.resize((512, 512), Image.BILINEAR)
            
            # Load and resize mask
            mask_path = os.path.join(self.masks_dir, f"{img_name}.png")
            mask = np.array(Image.open(mask_path))
            mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            
            # Create binary mask (person = 1, else = 0)
            binary_mask = (mask_resized == 15).astype(np.uint8) * 255
            
            # Save processed data
            image_resized.save(os.path.join(output_dir, "images", f"{img_name}.jpg"))
            Image.fromarray(binary_mask).save(os.path.join(output_dir, "masks", f"{img_name}.png"))
            
            # Create ROI mask (white for person, black for background)
            roi_mask = binary_mask.astype(np.float32) / 255.0
            np.save(os.path.join(output_dir, "roi_masks", f"{img_name}.npy"), roi_mask)
        
        print(f"Processed {len(person_images)} images saved to {output_dir}")
        
        # Create train/test split (90/10 as in paper)
        self.create_train_test_split(person_images, output_dir)
    
    def create_train_test_split(self, image_list, output_dir, test_ratio=0.1):
        """Create train/test split files"""
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(image_list)
        
        split_idx = int(len(image_list) * (1 - test_ratio))
        train_images = image_list[:split_idx]
        test_images = image_list[split_idx:]
        
        # Save split files
        with open(os.path.join(output_dir, "train.txt"), 'w') as f:
            for img in train_images:
                f.write(f"{img}\n")
        
        with open(os.path.join(output_dir, "test.txt"), 'w') as f:
            for img in test_images:
                f.write(f"{img}\n")
        
        print(f"Train: {len(train_images)} images")
        print(f"Test: {len(test_images)} images")
        print(f"Split files saved to {output_dir}")
    
    def create_dataset_info(self):
        """Create dataset information file"""
        info = {
            "dataset": "VOC2012",
            "num_classes": 2,
            "classes": ["background", "person"],
            "image_size": [512, 512],
            "roi_class_id": 15,
            "total_images": len(self.get_person_images()),
            "notes": "Preprocessed for semantic communication system"
        }
        
        import json
        with open("data/dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print("Dataset info saved to data/dataset_info.json")

if __name__ == "__main__":
    preprocessor = VOC2012Preprocessor()
    preprocessor.create_binary_masks()
    preprocessor.create_dataset_info()