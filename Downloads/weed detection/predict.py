from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class WeedDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize the weed detector with a trained model."""
        self.model = YOLO(model_path)
        self.model = YOLO(model_path)
        # Get class names from the model
        self.class_names = self.model.names
        # Generate random colors for each class
        self.colors = {}
        import random
        for class_id in range(len(self.class_names)):
            self.colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def draw_predictions(self, image, results):
        """Draw bounding boxes and confidence scores on the image."""
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get class id and confidence
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Draw bounding box
                color = self.colors[class_id]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class name and confidence
                label = f'{self.class_names[class_id]}: {confidence:.2%}'
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1-label_h-5), (x1+label_w, y1), color, -1)
                
                # Draw label text
                cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return image

    def predict_image(self, image_path, save_path=None):
        """Predict on a single image and show/save results."""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Run prediction
        results = self.model(image)
        
        # Draw predictions
        annotated_image = self.draw_predictions(image.copy(), results)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, annotated_image)
        
        # Display image
        cv2.imshow('Prediction', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predict_video(self, video_path, save_path=None):
        """Predict on video and show/save results."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer if save path provided
        if save_path:
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                   fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run prediction
            results = self.model(frame)
            
            # Draw predictions
            annotated_frame = self.draw_predictions(frame.copy(), results)
            
            # Save frame if writer exists
            if save_path:
                writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow('Prediction', annotated_frame)
            
            # Break if 'q' pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        if save_path:
            writer.release()
        cv2.destroyAllWindows()

def main():
    # Initialize detector
    detector = WeedDetector()
    
    # Test on an image
    print('Testing image prediction...')
    image_path = 'dataset/train/images/only-weed-1_0.jpg'
    detector.predict_image(image_path, 'output_image.jpg')
    
    # Test on a video
    print('\nTesting video prediction...')
    video_path = 'dataset/only-weed/only-weed-1.mp4'
    detector.predict_video(video_path, 'output_video.mp4')

if __name__ == "__main__":
    main()
