from ultralytics import YOLO
import os

def train_model():
    # Print dataset information
    print("\nChecking dataset structure:")
    for split in ['train', 'val']:
        img_dir = os.path.join('dataset', split, 'images')
        label_dir = os.path.join('dataset', split, 'labels')
        if os.path.exists(img_dir):
            print(f"{split} images:", len(os.listdir(img_dir)))
        if os.path.exists(label_dir):
            print(f"{split} labels:", len(os.listdir(label_dir)))
    
    # Load a pretrained YOLOv8n model
    print("\nLoading model...")
    model = YOLO('yolov8n.pt')
    
    # Train the model on our custom dataset
    print("\nStarting training...")
    try:
        results = model.train(
            data='dataset.yaml',      # Path to data config file
            epochs=100,              # Increased epochs for small dataset
            imgsz=640,              # Image size
            batch=4,                # Smaller batch size for better generalization
            name='weed_detector',   # Name of the experiment
            verbose=True,           # Print training progress
            patience=50,            # Early stopping patience
            # Augmentation parameters for small dataset
            mosaic=1.0,            # Mosaic augmentation
            scale=0.5,             # Scale images
            fliplr=0.5,            # Horizontal flip
            flipud=0.3,            # Vertical flip
            hsv_h=0.015,           # HSV hue augmentation
            hsv_s=0.7,             # HSV saturation augmentation
            hsv_v=0.4,             # HSV value augmentation
            degrees=10.0,          # Rotation
            translate=0.2,         # Translation
            perspective=0.001,     # Perspective
            shear=2.0,            # Shear
            mixup=0.1,            # Mixup
            copy_paste=0.1,       # Copy-paste augmentation
        )
        
        # Validate the model
        results = model.val()
        
        print("\nTraining completed. Model saved in 'runs/detect/weed_detector'")
    except Exception as e:
        print("\nError during training:", str(e))

if __name__ == "__main__":
    train_model()
