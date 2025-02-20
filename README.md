# Plant Health and Type Detection using Neural Networks

This repository contains a PyTorch-based implementation for detecting plant health status (healthy or unhealthy) and classifying plant types from leaf images. The project leverages custom datasets and models for predictions.

## Table of Contents

- [Plant Health and Type Detection using Neural Networks](#plant-health-and-type-detection-using-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training Models](#training-models)
    - [Predicting Plant Health and Type](#predicting-plant-health-and-type)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
    - [Health Detection](#health-detection)
    - [Plant Type Classification](#plant-type-classification)
  - [Training and Validation](#training-and-validation)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

---

## Overview

This project uses Artificial Neural Networks (ANN) for plant leaf analysis. It comprises two primary tasks:
1. **Health Detection:** Classify whether a leaf is healthy or unhealthy.
2. **Plant Type Classification:** Identify the type of plant based on leaf images.

The implementation includes data preprocessing, custom datasets, model training, and interactive prediction functionality.

---

## Features

- **Custom Models:** Specialized ANN models for health detection and plant type classification.
- **Custom Datasets:** Flexible dataset handling with PyTorch `Dataset` class.
- **Image Augmentation:** Enhanced training with data augmentation.
- **Interactive Prediction:** Select images for on-the-fly prediction using trained models.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/plant-health-classification.git
   cd plant-health-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure GPU support for PyTorch (optional but recommended).

---

## Usage

### Training Models

Run the `main.py` script to train the models:
```bash
python main.py
```
- You will be prompted to specify the number of epochs for training each model.
- Ensure that the dataset is downloaded and accessible at the correct paths.

### Predicting Plant Health and Type

1. you can repare a folder of test images or used the ones provided.
2. Follow the interactive prompts in `main.py` to analyze an image.
3. The program outputs the predicted health status and plant type.

---

## Dataset

The project uses the **[New Plant Diseases Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)** available on Kaggle. The dataset includes:
- **Health Status:** Healthy and unhealthy labels.
- **Plant Types:** 14 plant species and a "Not A Plant" category.

---

## Model Architecture

### Health Detection
- Input size: 64x64 grayscale images (flattened to 4,096 features).
- Architecture:
  - Fully connected layers: 4,096 → 1,024 → 256 → 64 → 2.
  - Activation: ReLU.
  - Output: Binary classification (healthy/unhealthy).

### Plant Type Classification
- Input size: 64x64 grayscale images (flattened to 4,096 features).
- Architecture:
  - Fully connected layers: 4,096 → 1,024 → 256 → 64 → 15.
  - Activation: ReLU.
  - Output: Multi-class classification (15 plant types).

---

## Training and Validation

The `train.py` script handles model training and validation:
- Loss: Cross-Entropy Loss.
- Optimizer: Adam.
- Metrics: Training loss, validation loss, and validation accuracy.

---

## Results

Include sample results and performance metrics:
- Health Detection Accuracy: up to 80.00%
- Plant Type Classification Accuracy: up to 60.00%

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
