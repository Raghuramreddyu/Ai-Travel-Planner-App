import cv2
import numpy as np
import os

# Load your dataset video
video_path = r"C:\Users\91630\OneDrive\Desktop\drone-weed\dataset\weed with crop\weed with crop-1.mp4"
if not os.path.exists(video_path):
    print("Video file not found!")
    exit()

# Open the video
cap = cv2.VideoCapture(video_path)

# Grid configuration for 16 cells (4x4)
GRID_COLS = 8
GRID_ROWS = 8

# HSV range for detecting green (weed)
LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

# Desired display width (adjust as needed)
DISPLAY_WIDTH = 800

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for better viewing
    h, w, _ = frame.shape
    aspect_ratio = h / w
    resized_frame = cv2.resize(frame, (DISPLAY_WIDTH, int(DISPLAY_WIDTH * aspect_ratio)))
    resized_h, resized_w, _ = resized_frame.shape

    cell_width = resized_w // GRID_COLS
    cell_height = resized_h // GRID_ROWS

    hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    total_weed_pixels = 0
    total_pixels = 0

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height

            cell_mask = mask[y1:y2, x1:x2]
            cell_weed = cv2.countNonZero(cell_mask)
            cell_total = cell_mask.size

            weed_percent = (cell_weed / cell_total) * 100
            total_weed_pixels += cell_weed
            total_pixels += cell_total

            # Draw grid and labels
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cell_id = row * GRID_COLS + col + 1
            cv2.putText(resized_frame, f"Cell {cell_id}", (x1 + 5, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(resized_frame, f"{weed_percent:.1f}%", (x1 + 5, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show overall weed coverage
    overall_coverage = (total_weed_pixels / total_pixels) * 100
    cv2.putText(resized_frame, f"Overall Weed Coverage: {overall_coverage:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video
    cv2.imshow("Weed Detection", resized_frame)
    if cv2.waitKey(25) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
