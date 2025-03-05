# Qatari License Plate Detection with YOLOv8

This project implements a custom object detection model to detect Qatari license plates using YOLOv8. It leverages the Roboflow API to download and manage datasets and uses the Ultralytics YOLOv8 library for training and evaluation.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

---

## Overview

The objective of this project is to train a YOLOv8 model to detect Qatari license plates with high accuracy. The dataset is managed using Roboflow, and training is performed using Ultralytics' YOLOv8 implementation.

### Key Features:
- **Custom Dataset**: A dataset for Qatari license plates sourced and managed via Roboflow, consisting of 900 original images taken in the local marketplace. The images capture license plates under various conditions, including daylight and nighttime, as well as in different lighting scenarios such as dark and sunny environments, and from various angles to enhance accuracy during model training.
- **YOLOv8 Training**: Detection model trained and evaluated with YOLOv8.
- **Visualization**: Confusion matrix, detection results, and validation predictions are displayed after training.

---

## Setup

Follow the steps below to set up and run the project:

1. **Check NVIDIA GPU**:
   Ensure you have GPU access for faster training.

   ```bash
   !nvidia-smi
   ```
2. **Create Datasets Folder**:
Create a directory to store the dataset.
   ```bash
   !mkdir -p {HOME}/datasets
   %cd {HOME}/datasets
   ```
3. **Install Dependencies**:
Install the required Python packages.
```bash
  !pip install roboflow==1.1.48 --quiet
  !pip install ultralytics
   ```
4. **Login to Roboflow**:
Authenticate with Roboflow to access the dataset.
Visit the following link: [Download Dataset](https://app.roboflow.com/husham-eina/qatari_license_plate_detection/2)

```bash
import roboflow

# Log in to Roboflow
roboflow.login()

# Visit the following link to download the dataset:
# https://app.roboflow.com/husham-eina/qatari_license_plate_detection/2
# Select YOLOv8 as the download format. 
# Click on "Show download code," then copy the generated code and paste it below.
#it will be as below .....

rf = roboflow.Roboflow()
project = rf.workspace("husham-eina").project("qatari_license_plate_detection")
version = project.version(2)
dataset = version.download("yolov8")
```

## Dataset

The dataset for this project is hosted on Roboflow. It contains images and annotations for Qatari license plates. 
The dataset is automatically downloaded and prepared for YOLOv8 training using the following:

```bash
dataset = version.download("yolov8")
```

## Training
To train the YOLOv8 model, use the following command:
```bash
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800 plots=True
```

Training Hyperparameters:
- **Model**: yolov8s.pt (YOLOv8 small model).
- **Epochs**: 25.
-**Image Size**: 800x800
-**Data Path**: Path to the downloaded dataset YAML file.

