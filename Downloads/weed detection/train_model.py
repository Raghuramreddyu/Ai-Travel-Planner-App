from ultralytics import YOLO
import os
import cv2
import shutil
import random
from pathlib import Path

def create_dataset_structure():
    """Create train/val/test splits and prepare dataset structure."""
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f'dataset/{split}/images', exist_ok=True)
        os.makedirs(f'dataset/{split}/labels', exist_ok=True)

def extract_frames_and_create_labels():
    """Extract frames from videos and create corresponding labels."""
    classes = {
        'only-weed': 0,  # weed class
        'weed with crop': [0, 1],  # both weed and crop
        'no-plants': None  # no labels needed
    }
    
    # Clear existing images and labels
    for split in ['train', 'val', 'test']:
        for folder in ['images', 'labels']:
            folder_path = f'dataset/{split}/{folder}'
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
    
    frame_count = 0
    for class_name, class_id in classes.items():
        video_dir = f'dataset/{class_name}'
        if not os.path.exists(video_dir):
            continue
            
        for video_file in os.listdir(video_dir):
            if not video_file.endswith(('.mp4', '.avi')):
                continue
                
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % 30 == 0:  # Extract every 30th frame
                    # Determine split (80% train, 10% val, 10% test)
                    split = 'train' if frame_count % 10 < 8 else ('val' if frame_count % 10 < 9 else 'test')
                    
                    # Save frame
                    frame_name = f"{os.path.splitext(video_file)[0]}_{frame_count}.jpg"
                    frame_path = f"dataset/{split}/images/{frame_name}"
                    cv2.imwrite(frame_path, frame)
                    
                    # Create label if class_id is not None
                    if class_id is not None:
                        height, width = frame.shape[:2]
                        label_path = f"dataset/{split}/labels/{os.path.splitext(frame_name)[0]}.txt"
                        
                        with open(label_path, 'w') as f:
                            if isinstance(class_id, list):
                                # For 'weed with crop', create multiple boxes
                                # Detect green areas for weed
                                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                                lower_green = (35, 30, 30)
                                upper_green = (85, 255, 255)
                                mask = cv2.inRange(hsv, lower_green, upper_green)
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                for contour in contours:
                                    if cv2.contourArea(contour) > 100:  # Filter small contours
                                        x, y, w, h = cv2.boundingRect(contour)
                                        # Convert to YOLO format (x_center, y_center, width, height)
                                        x_center = (x + w/2) / width
                                        y_center = (y + h/2) / height
                                        w = w / width
                                        h = h / height
                                        f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                                
                                # Add some crop boxes
                                for _ in range(2):
                                    x_center = random.uniform(0.2, 0.8)
                                    y_center = random.uniform(0.2, 0.8)
                                    w = random.uniform(0.1, 0.3)
                                    h = random.uniform(0.1, 0.3)
                                    f.write(f"1 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                            else:
                                # For 'only-weed', detect green areas
                                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                                lower_green = (35, 30, 30)
                                upper_green = (85, 255, 255)
                                mask = cv2.inRange(hsv, lower_green, upper_green)
                                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                for contour in contours:
                                    if cv2.contourArea(contour) > 100:  # Filter small contours
                                        x, y, w, h = cv2.boundingRect(contour)
                                        # Convert to YOLO format (x_center, y_center, width, height)
                                        x_center = (x + w/2) / width
                                        y_center = (y + h/2) / height
                                        w = w / width
                                        h = h / height
                                        f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
                frame_count += 1
            cap.release()

def train_model():
    """Train the YOLOv8 model on the prepared dataset."""
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    model.train(
        data='data.yaml',
        epochs=50,  # reduced epochs for faster training
        imgsz=640,
        batch=16,
        name='weed_detection'
    )

def main():
    print("Creating dataset structure...")
    create_dataset_structure()
    
    print("Extracting frames and creating labels...")
    extract_frames_and_create_labels()
    
    print("Training model...")
    train_model()

if __name__ == "__main__":
    main()
