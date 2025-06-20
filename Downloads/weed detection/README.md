# Weed Detection System

This project uses YOLOv8 to detect and classify weeds and crops in videos.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your videos in the dataset folder with the following structure:
```
dataset/
├── only-weed/
├── no-plants/
└── weed with crop/
```

3. Train the model:
```bash
python train_model.py
```

4. Run detection on new videos:
```bash
python detect_weed.py
```

## Output
- The trained model will be saved in `runs/detect/weed_detection/`
- Detected videos will be saved with '_detected' suffix
- The model shows bounding boxes around detected weeds and crops with confidence scores
