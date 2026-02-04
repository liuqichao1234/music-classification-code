# CT-GateNet: Music Genre Classification with CNN-Transformer Hybrid Model

[![Paper](https://img.shields.io/badge/Paper-PLOS_One-blue)](https://journals.plos.org/plosone/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of **CT-GateNet** (Convolutional Neural Network-Transformer Gating Network), as described in the paper:
> **"Adaptive Feature Fusion Gate and Gated Channel-Spatial Attention in CNN-Transformer Models for Music Genre Classification"**

## 🌟 Project Overview

CT-GateNet is an innovative hybrid architecture designed for high-performance music genre classification. It addresses the limitations of unimodal architectures by synergistically combining:
1.  **CNN Branch**: Captures fine-grained local spectral patterns.
2.  **Transformer Branch**: Models long-range global temporal dependencies.
3.  **Gated Channel-Spatial Attention (GCSA)**: Enhances feature discriminability.
4.  **Adaptive Feature Fusion Gate (AFFG)**: Dynamically balances contributions from both branches.

### Key Results
| Dataset | Type | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GTZAN** | Balanced | **97.63%** | 97.64% | 97.63% | 97.63% |
| **FMA-Small** | Balanced | **90.38%** | 59.68% | 60.79% | 60.23% |
| **FMA-Medium** | Unbalanced | **69.07%** | - | - | - |

---
<img width="1218" height="545" alt="fmaresult" src="https://github.com/user-attachments/assets/a834d412-8b96-45b1-b734-ea57df9e73a1" />

## 📂 Repository Structure

```text
.
├── attention/              # Gated Channel-Spatial Attention (GCSA) implementation
│   └── attention.py
├── dataload/               # Data loading & preprocessing scripts
│   ├── dataload.py         # Main data loading logic
│   ├── data_process（mel）.py # Mel spectrogram generation
│   └── data_process（stft）.py # STFT feature extraction
├── generative_model/       # DDIM-based data augmentation for minority classes
│   └── generating_music.py
├── model/                  # Model architecture definitions
│   ├── model1.py           # CT-GateNet (Proposed Model)
│   ├── model2.py           # Variant architecture
│   └── baseline.py         # Baseline CNN/Transformer models
├── data/                   # Dataset storage (GTZAN/FMA-Small/FMA-Medium)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/liuqichao1234/music-classification.git
cd music-classification
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download and place datasets in the `data/` directory:
- **GTZAN**: [Kaggle Link](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
- **FMA**: [GitHub Link](https://github.com/mdeff/fma) (Includes Small and Medium versions)

### 3. Data Preprocessing
Generate Mel spectrograms:
```bash
python dataload/data_process（mel）.py
```

---

## 📊 Reproducing Results

### Step 1: Data Augmentation (DDIM)
To alleviate data scarcity or imbalance, use the Denoising Diffusion Probabilistic Model (DDIM):
```bash
python generative_model/generating_music.py
```

### Step 2: Training
Train the CT-GateNet model on your chosen dataset:
```bash
# Default training script for CT-GateNet
python model/model1.py
```

### Step 3: Evaluation & Inference
The training script automatically performs 10-fold cross-validation. To run inference on a single audio file:
```python
# Example inference snippet
from model.model1 import MyCNNTransformerModel
model = MyCNNTransformerModel(input_shape=(128, 128, 1), num_classes=10)
model.load_weights('path_to_weights.h5')
# ... load and preprocess audio ...
prediction = model.predict(processed_audio)
```

---

## 💾 Model Weights
Model weights and checkpoints can be downloaded from the following links:
- [Google Drive Placeholder]
- [HuggingFace Placeholder]

---

## 📝 Citation
```bibtex
@article{ma2026adaptive,
  title={Adaptive Feature Fusion Gate and Gated Channel-Spatial Attention in CNN-Transformer Models for Music Genre Classification},
  author={Ma, Yunyan and Ding, Zhenwu and Wan, Shuang and Li, Hui and Xu, Yuan},
  journal={PLOS One},
  year={2026}
}
```
