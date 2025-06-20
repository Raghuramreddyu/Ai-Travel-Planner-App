import os
import shutil
from pathlib import Path
import random

def prepare_dataset():
    # Source directories
    src_images = Path('dataset/annotations/images')
    src_labels = Path('dataset/annotations/labels')
    
    # Create train and val directories
    dataset_root = Path('dataset')
    for split in ['train', 'val']:
        (dataset_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get all image files that have corresponding labels
    image_files = []
    for img_path in src_images.glob('*.jpg'):
        label_path = src_labels / (img_path.stem + '.txt')
        if label_path.exists():
            image_files.append(img_path)
    
    if not image_files:
        print("Error: No valid image-label pairs found!")
        return
        
    print(f"Found {len(image_files)} valid image-label pairs")
    random.shuffle(image_files)
    
    # Split into train (80%) and val (20%)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Clean existing files
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            folder_path = dataset_root / split / folder
            if folder_path.exists():
                for file in folder_path.glob('*'):
                    file.unlink()
    
    # Copy files to respective directories
    def copy_files(files, split):
        for img_path in files:
            # Copy image
            shutil.copy2(img_path, dataset_root / split / 'images' / img_path.name)
            
            # Copy corresponding label
            label_path = src_labels / (img_path.stem + '.txt')
            shutil.copy2(label_path, dataset_root / split / 'labels' / label_path.name)
    
    print("Preparing dataset...")
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    print(f"Dataset prepared: {len(train_files)} training images, {len(val_files)} validation images")

if __name__ == "__main__":
    prepare_dataset()
