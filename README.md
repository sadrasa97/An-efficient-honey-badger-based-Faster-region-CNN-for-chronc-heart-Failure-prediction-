
# ECG Signal Classification and Analysis

This project is focused on classifying ECG signals using deep learning and machine learning techniques. It involves signal filtering, feature extraction (DCT, FFT), adaptive filtering using the DNLMS algorithm, CNN-based classification, and performance evaluation. Additionally, an SVM classifier is trained on extracted features to enhance classification performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Project Overview

The primary objective of this project is to classify ECG signals into normal and abnormal categories. The project includes various stages such as:
- Data preprocessing with bandpass, highpass, and lowpass filters.
- Feature extraction using Discrete Cosine Transform (DCT) and Fast Fourier Transform (FFT).
- Adaptive filtering using the DNLMS (Dynamic Normalized Least Mean Square) algorithm.
- Model training with CNN and optimization using a custom function (inspired by honey badger optimization).
- SVM classification for additional performance assessment.

## Requirements

To run this project, you need to install the following libraries:

```bash
pip install wfdb tensorflow scipy scikit-learn matplotlib
```

## Dataset

The project uses ECG data from the MIT-BIH Normal Sinus Rhythm Database. You can download it from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/). Place the data files in the `data_directory` specified in the code.

## Data Preprocessing

- **Signal Filtering**: ECG signals undergo bandpass, highpass, and lowpass filtering to remove noise and baseline wander.
- **Segmentation**: The ECG signals are segmented into fixed-length images, each representing a specific portion of the signal for model input.

## Feature Extraction

- **DCT (Discrete Cosine Transform)**: Used to extract compact and efficient features from ECG signals.
- **FFT (Fast Fourier Transform)**: Provides frequency-domain features from the ECG signals.

## Model Architecture

- **CNN Model**: The CNN model architecture comprises convolutional layers followed by pooling and fully connected layers. This model is designed for binary classification, outputting probabilities for normal and abnormal ECG signals.
- **SVM Classifier**: An SVM model with an RBF kernel is used to classify ECG signals based on extracted features.

## Training and Evaluation

- **CNN Optimization**: A custom optimization function perturbs model weights to improve performance over multiple iterations.
- **Training**: The model is trained on the processed ECG images with labels for normal and abnormal signals.
- **Evaluation Metrics**: We evaluate model performance using accuracy, precision, recall, and F1 score.

## Results

The CNN and SVM models were evaluated on accuracy, precision, recall, and F1 score. Below are sample results (adjust based on your actual outcomes):

- **SVM Accuracy**: `90.5%`
- **SVM Precision**: `89.2%`
- **SVM Recall**: `91.0%`
- **SVM F1 Score**: `90.1%`

## Usage

1. **Preprocess the Data**: Run the code section for data loading, filtering, and segmentation.
2. **Train the Model**: Execute the training code for CNN and optionally the SVM classifier.
3. **Evaluate Performance**: Run the evaluation code to check accuracy, precision, recall, and F1 score.

