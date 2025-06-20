import cv2
import numpy as np
import os
from pathlib import Path

class VideoAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.current_frame = None
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.boxes = []
        self.current_class = 0  # 0 for weed, 1 for crop
        self.class_colors = {
            0: (0, 255, 0),  # Green for weeds
            1: (255, 0, 0)   # Blue for crops
        }
        self.frame_annotations = {}
        self.current_frame_num = 0
        self.paused = True
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directories
        self.output_dir = Path('dataset/annotations')
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.current_frame.copy()
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), 
                            self.class_colors[self.current_class], 2)
                self.draw_info(img_copy)
                cv2.imshow('Annotation', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            if x1 != x2 and y1 != y2:  # Ensure box has area
                self.boxes.append((self.current_class, x1, y1, x2, y2))
                self.draw_boxes()

    def draw_info(self, img):
        # Draw frame information and controls
        info_text = [
            f"Frame: {self.current_frame_num}/{self.total_frames}",
            f"Current class: {'Crop' if self.current_class == 1 else 'Weed'}",
            "Controls:",
            "SPACE: Play/Pause",
            "->: Next frame",
            "<-: Previous frame",
            "c: Toggle class",
            "s: Save annotations",
            "d: Delete last box",
            "q: Quit and save"
        ]
        
        y = 30
        for text in info_text:
            cv2.putText(img, text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25

    def draw_boxes(self):
        img_copy = self.current_frame.copy()
        for class_id, x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), 
                        self.class_colors[class_id], 2)
            label = 'Weed' if class_id == 0 else 'Crop'
            cv2.putText(img_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       self.class_colors[class_id], 2)
        self.draw_info(img_copy)
        cv2.imshow('Annotation', img_copy)

    def save_annotations(self):
        if not self.boxes:
            return
            
        # Save image
        img_path = self.images_dir / f'frame_{self.current_frame_num:06d}.jpg'
        cv2.imwrite(str(img_path), self.current_frame)
        
        # Save annotations in YOLO format
        h, w = self.current_frame.shape[:2]
        label_path = self.labels_dir / f'frame_{self.current_frame_num:06d}.txt'
        
        with open(label_path, 'w') as f:
            for class_id, x1, y1, x2, y2 in self.boxes:
                # Convert to YOLO format (center_x, center_y, width, height)
                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                f.write(f'{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n')
        print(f'Saved annotations for frame {self.current_frame_num}')

    def save_all_annotations(self):
        print("Saving all annotations...")
        for frame_num, boxes in self.frame_annotations.items():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.boxes = boxes
                self.current_frame_num = frame_num
                self.save_annotations()
        print(f"Saved annotations for {len(self.frame_annotations)} frames")

    def run(self):
        cv2.namedWindow('Annotation')
        cv2.setMouseCallback('Annotation', self.mouse_callback)
        
        # Read first frame
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.draw_boxes()
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame
                    self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    self.boxes = self.frame_annotations.get(self.current_frame_num, [])
                    self.draw_boxes()
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                self.save_all_annotations()
                break
            elif key == ord('s'):  # Save current frame and annotations
                self.save_annotations()
            elif key == ord('c'):  # Toggle class
                self.current_class = 1 - self.current_class
                print(f'Current class: {"Crop" if self.current_class == 1 else "Weed"}')
            elif key == ord('d'):  # Delete last box
                if self.boxes:
                    self.boxes.pop()
                    self.draw_boxes()
            elif key == 32:  # Spacebar - toggle pause
                self.paused = not self.paused
            elif key == 83:  # Right arrow - next frame
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame
                    self.current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    self.boxes = self.frame_annotations.get(self.current_frame_num, [])
                    self.draw_boxes()
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
            elif key == 81:  # Left arrow - previous frame
                prev_frame = self.current_frame_num - 1
                if prev_frame >= 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
                    ret, frame = self.cap.read()
                    if ret:
                        self.current_frame = frame
                        self.current_frame_num = prev_frame
                        self.boxes = self.frame_annotations.get(self.current_frame_num, [])
                        self.draw_boxes()
            
            self.frame_annotations[self.current_frame_num] = self.boxes
            
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your video path
    video_path = r'C:\Users\91630\OneDrive\Desktop\drone-weed\dataset\weed with crop\weed with crop-2.mp4'
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
    else:
        annotator = VideoAnnotator(video_path)
        annotator.run()
