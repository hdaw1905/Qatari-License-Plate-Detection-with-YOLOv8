# Qatari License Plate Detection with YOLOv8
![image](https://github.com/user-attachments/assets/e67a7bea-8785-4ac6-90f2-b6dac6f560c0)

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
-**Model**: yolov8s.pt (YOLOv8 small model).
-**Epochs**: 25.
-**Image Size**: 800x800
-**Data Path**: Path to the downloaded dataset YAML file.

## Results
After training, the following results are visualized:
1.**Confusion Matrix**:
Displays the confusion matrix for the model's predictions.
```bash
conf_matrix_path = f"/content/{HOME}/datasets/{HOME}/datasets/{HOME}/datasets/runs/detect/train/confusion_matrix.png"
if os.path.exists(conf_matrix_path):
    display(Image(filename=conf_matrix_path, width=600))
```
![image](https://github.com/user-attachments/assets/606014c9-e0e4-4e3f-ba41-d0d0cde0785b)

2.**Training Results**:
Shows a graphical summary of the training process.
```bash
conf_matrix_path = f"/content/{HOME}/datasets/{HOME}/datasets/{HOME}/datasets/runs/detect/train/results.png"
if os.path.exists(conf_matrix_path):
    display(Image(filename=conf_matrix_path, width=600))
```
![image](https://github.com/user-attachments/assets/f9be455c-b95c-4e8c-bcae-34d4a7c8aa55)

3.**Validation Batch Predictions**:
Displays predictions on a validation batch.

```bash
conf_matrix_path = f"/content/{HOME}/datasets/{HOME}/datasets/{HOME}/datasets/runs/detect/train/val_batch0_pred.jpg"
if os.path.exists(conf_matrix_path):
    display(Image(filename=conf_matrix_path, width=600))
```
![image](https://github.com/user-attachments/assets/5d3d1763-359b-431f-a809-e18ba74cc49c)

## Directory Structure  :
```bash
.
├── Qatari_License_Plate_Detection-2
│   ├── train
│   │   ├── images
│   │   └── labels
│   ├── test
│   │   ├── images
│   │   └── labels
│   └── valid
│       ├── images
│       └── labels
├── runs
│   └── detect
│       └── train
│           ├── weights
│           ├── confusion_matrix.png
│           ├── results.png
│           └── val_batch0_pred.jpg
└── ...
```

** Key Directories:
1- **Qatari_License_Plate_Detection-2/train**: Contains training images and labels.

2- **Qatari_License_Plate_Detection-2/valid**: Contains validation images and labels.

3- **Qatari_License_Plate_Detection-2/test**: Contains testing images and labels.

4- **runs/detect/train**: Stores results from YOLOv8 training, including weights, confusion matrix, and validation predictions.

## Dependencies :
The following dependencies are required to run the project:

- **Python 3.8+**

- **Roboflow**: pip install roboflow==1.1.48

- **Ultralytics YOLOv8**: pip install ultralytics

- **Jupyter Notebook** for running and visualizing the training process.

## Acknowledgements: 
- **Roboflow**: For providing the tools to manage and preprocess datasets.

- **Ultralytics**: For the YOLOv8 implementation.

- **Google Colab**: For providing a free environment for training and evaluation.

# Author

**Husham Eina Abdalla**  
Email: [heashm.eina@gmail.com](mailto:heashm.eina@gmail.com)  
LinkedIn: [Husham E. Abdalla](https://www.linkedin.com/in/husham-e-abdalla/)

Feel free to reach out with any questions or contributions!

### Changes Made:
1. Incorporated the provided directory structure explicitly under the **Directory Structure** section.
2. Highlighted the excluded files and directories (`images`, `labels`, etc.) in the relevant paths.
3. Adjusted the formatting to align with the provided structure and clarified the role of each directory.

Replace placeholders like `[Your Name]` with your actual details.
