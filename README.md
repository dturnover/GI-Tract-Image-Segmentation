# GI Tract Image Segmentation

## Overview
This project implements a **U-Net deep learning model** for **medical image segmentation**, specifically targeting segmentation of the gastrointestinal (GI) tract from medical scans. The pipeline includes custom data preprocessing, a modular training process, and exportable predictions in **Run-Length Encoding (RLE)** format.

---

## Key Features
- **Data Pipeline**:
  - Custom data generator to handle large datasets with RLE-encoded masks.
  - Dynamic resizing of images and masks to a configurable target size.
  - Flexible test mode for visualizing individual predictions and ground truths.

- **Model Architecture**:
  - Based on **TransUNet**, combining convolutional layers with transformer-based features for superior performance.
  - Option to load pre-trained weights for transfer learning or train from scratch.

- **Evaluation**:
  - Metrics include **Dice coefficient**, **accuracy**, and **visual analysis** of predictions.
  - Visualization overlays for ground truths and predictions on original images.

- **Export**:
  - Saves predictions in RLE format compatible with Kaggle competitions or downstream pipelines.

---

## Requirements
To set up and run the project, ensure the following dependencies are installed:
- TensorFlow 2.8+
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

---

## Usage

### **1. Running the Pipeline**
Run the main script to train, evaluate, or export predictions:
python GI-Tract-Image-Segmentation.py

---

### **2. Training the Model**
If training from scratch:
- Automatically splits the data into training, validation, and test sets.
- Implements early stopping, learning rate scheduling, and model checkpointing.
- Saves the best model weights to the `output/` directory.

### **3. Evaluating the Model**
During evaluation, the script:
- Computes **Dice coefficient** and loss for each test sample.
- Visualizes predictions and overlays with ground truths.

---

### **Data Pipeline**
The project includes a highly modular pipeline:
- **Custom Generator**:
  - Decodes RLE masks into binary masks dynamically.
  - Handles resizing, augmentation, and batch generation.
- **Training Pipeline**:
  - Modularized for scalability and customization.
  - Includes checkpoints and CSV logging of training metrics.
## Model Architecture
- **Base Model**: TransUNet
  - Combines transformer layers for long-range dependency capture with CNNs for spatial feature extraction.
- **Custom Modifications**:
  - Configurable input size.
  - Optional pre-trained weights for transfer learning.
