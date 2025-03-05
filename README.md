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
