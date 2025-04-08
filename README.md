# TwoStageOpenMax: Open Set Recognition with Class-Mean Based OOD Detection

## Project Overview
This repository contains the implementation of TwoStageOpenMax, a two-stage approach for open set recognition (OSR) that can classify known classes accurately while detecting samples from unknown classes. Unlike traditional classification tasks limited to a closed set of classes, OSR mirrors real-world scenarios where novel classes may appear during inference.

This project was developed as part of the "Applied Deep Learning" course at Ben-Gurion University, Beer Sheva, Israel.

## Authors
- Nissim Brami
- Amnon Abaev

## Approach
Our TwoStageOpenMax implementation uses a class-mean based out-of-distribution (OOD) detection mechanism. The model consists of:

1. A CNN-based feature extractor and classifier for MNIST digits
2. A distance-based detector that identifies unknown samples based on their distance from class means in the embedding space

By computing class means during training and establishing a statistical threshold on distances, our model achieves high accuracy in both classifying MNIST digits (94.37%) and detecting out-of-distribution samples (98.75%), with an overall accuracy of 95.10%.

## Model Architecture
The TwoStageOpenMax model architecture includes:

- **Feature Extraction Layers**: A CNN that extracts relevant features from input images
- **Embedding Layer**: Maps extracted features to a lower-dimensional embedding space
- **Classification Layer**: Produces class probabilities from the embedding
- **Unknown Detection Mechanism**: Uses distances in the embedding space to identify OOD samples

## Key Features
- Effective unknown detection without exposure to OOD samples during training
- Distance-based detection mechanism that leverages the geometric properties of the embedding space
- Comprehensive visualization tools for understanding model behavior
- Robust performance across diverse OOD datasets

## Results
Our model achieved the following performance:

- MNIST Classification Accuracy: 94.37%
- OOD Detection Accuracy: 98.75%
- Total Accuracy: 95.10%

## Repository Contents
- `TwoStageOpenMax.ipynb`: Jupyter notebook containing the implementation code
- `osr_model.pth`: Trained model weights
- `README.md`: This file

## Requirements
The code requires the following libraries:
- PyTorch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/TwoStageOpenMax.git
cd TwoStageOpenMax
```

2. Run the Jupyter notebook to see the implementation:
```bash
jupyter notebook TwoStageOpenMax.ipynb
```

3. To use the pre-trained model, load the weights:
```python
model = TwoStageOpenMax(embedding_dim=64).to(device)
model = retrieve_model(model, "osr_model.pth")
```

## License
This project is available for academic and educational purposes.

## Citation
If you find this work useful for your research, please consider citing:
```
@article{twostageopenmax,
  title={TwoStageOpenMax: Open Set Recognition with Class-Mean Based OOD Detection},
  author={Brami, Nissim and Abaev, Amnon},
  year={2025}
}
```
