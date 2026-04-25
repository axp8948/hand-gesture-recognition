# Real-Time Hand Gesture Recognition Using CNN

This project implements a real-time hand gesture recognition system using a Convolutional Neural Network (CNN). It captures hand images from a webcam, processes them, trains a deep learning model, and performs live inference to classify gestures in real time.

## Project Overview

The system recognizes five hand gestures:
- Thumb  
- Fist  
- Palm  
- Peace  
- Call  

Pipeline:
Data Collection → Preprocessing → CNN Training → Evaluation → Real-Time Inference

## Model Details

- Architecture: Custom CNN (3 convolutional blocks)
- Input size: 128 × 128 × 3
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Dropout: 0.5
- Output: Softmax (5 classes)

Pretrained model:
models/hand_gesture_cnn_5sign.h5

## Project Structure

```text
src/
├── dataCollection
│   ├── dataCollection.py
│   └── saveImage.py
├── inference
│   ├── ImageCapture.py
│   ├── featureVisualization.ipynb
│   └── test.py
├── models
│   ├── evaluation.py
│   ├── hand_gesture_cnn.h5
│   ├── hand_gesture_cnn_5sign.h5
│   ├── hand_gesture_cnn_old.h5
│   └── train.py
├── preprocessing
├── requirements_capture.txt
├── requirements_train.txt
└── utils

## Requirements

Two separate requirement files are used:

- src/requirements_capture.txt → data collection + inference
- src/requirements_train.txt → training

Install dependencies:
pip install -r src/requirements_train.txt

or

pip install -r src/requirements_capture.txt

## How to Run

### 1. Collect Data
python src/dataCollection/saveImage.py

### 2. Train Model
python src/models/train.py

### 3. Evaluate Model
python src/models/evaluation.py

### 4. Run Real-Time Prediction
python src/inference/test.py

## Dataset

- Captured using webcam
- Hand detection via MediaPipe / cvzone
- Each image:
  - Cropped around hand region
  - Normalized on white background
  - Resized to 128×128


## Future Work

- Larger dataset collection
- Better segmentation
- Temporal smoothing (majority voting)
- Advanced models (MobileNet / ResNet)
- Real-world gesture applications

## Author

Anmol Pandey
