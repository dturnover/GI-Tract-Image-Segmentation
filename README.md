# U-Net for Medical Image Segmentation

This project implements a U-Net deep learning model for medical image segmentation. The primary goal is to accurately segment target structures from medical scans, using a highly efficient and modular pipeline for both training and evaluation.

## Features

- **Data Pipeline**: Custom data generator for handling large datasets with Run-Length Encoded (RLE) masks.
- **Model Architecture**: U-Net with configurable parameters for medical image segmentation tasks.
- **Evaluation**: Visualization of predictions and computation of Dice loss for performance metrics.
- **Export**: Automatic saving of predictions in CSV format using RLE encoding.

## File Structure

```plaintext
.
├── train.py                 # Main script for training, evaluation, and exporting predictions
├── requirements.txt         # List of dependencies
├── README.md                # Project documentation
├── data/
│   ├── train.csv            # CSV with image IDs and RLE-encoded masks
│   ├── train/               # Directory containing training images
│   └── test/                # Directory containing test images
└── output/
    ├── unet_model.h5        # Saved U-Net model
    └── predictions.csv      # Predictions in RLE format
