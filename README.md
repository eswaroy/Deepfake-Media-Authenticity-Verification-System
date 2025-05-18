# Deepfake Detection System Using Deep Learning
This project detects deepfake videos using deep learning models on the UADFV Dataset. It uses facial features extracted from videos and trains CNN-based models to classify videos as real or fake.

## Features
Automatic Metadata Generation: Generates metadata from the dataset directory containing real and fake videos.

Face Extraction from Videos: Extracts up to 20 facial frames from each video using OpenCV Haar Cascades.

Multiple Model Support: Offers three model types – Simple CNN, MobileNetV2, and EfficientNetB0.

Transfer Learning: EfficientNetB0 and MobileNetV2 use pretrained ImageNet weights to improve performance.

Training Pipeline: Includes callbacks like early stopping, learning rate reduction, and best model checkpointing.

Model Evaluation: Outputs accuracy, classification report, confusion matrix, and training graphs.

Video Prediction with Annotation: Predicts new video files and saves annotated versions with bounding boxes and prediction scores.

Model Saving & Loading: Trained models are saved and can be reloaded for future predictions.
UADFV/
## Dataset Structure
Download the dataset from Kaggle and organize it like this:
├── real/
│   ├── real_001.mp4
│   ├── ...
├── fake/
│   ├── fake_001.mp4
│   ├── ...

## Installation
Install the required packages using pip:



Edit
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn

## How to Use
Update the base_path variable in main.py to point to your dataset folder.

Run the script:


Edit
python main.py

This will:

Load and process metadata

Extract face frames

Train the selected model

Evaluate and save results

## Output Files
metadata.csv: Contains video paths and labels (real/fake).

faces/: Contains extracted facial images.

checkpoints/: Stores the best model during training.

results/: Confusion matrix and training history plots.

models/deepfake_detector.h5: Final saved model.

annotated_video.mp4: (Optional) Video with face boxes and fake probabilities.
