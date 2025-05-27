# Model Implementations for Kinetics Estimation from Wearable and Multimodal Data

This repository contains PyTorch implementations of multiple state-of-the-art (SOTA) models used in our study for human kinetics estimation using wearable sensors, EMG signals, and video data.

Each file corresponds to one or more related models described in the manuscript, implemented from scratch or adapted from previous work.

## 📁 Model Files

### `ffn_hf_ffn_bilstm_convnet.py`
Implements the following baseline deep learning models:

- **FFN (HF)**  
  A feedforward neural network built on 15 handcrafted features (e.g., mean, RMS, standard deviation, etc.) extracted per signal channel over a 0.5s window. Batch normalization, dropout, and ReLU activations are used across fully connected layers.

- **FFN**  
  A raw-data-based feedforward network with three fully connected layers using ReLU and dropout, processing batch-normalized IMU/EMG/video signals.

- **Bi-LSTM**  
  A recurrent model with two Bi-LSTM layers, each followed by dropout. Outputs are flattened and fed into a dense layer for kinetics prediction.

- **2D Conv. Network**  
  A CNN model consisting of four blocks of Conv2D → BatchNorm → MaxPooling, followed by two ReLU-activated fully connected layers with dropout, and a final regression layer.

---

### `lmfn_tfn_fusion_models.py`
Implements two multimodal fusion architectures:

- **LMFN (Low-Rank Multimodal Fusion Network)**  
  Each modality is encoded via Bi-LSTM layers. Outputs are concatenated and passed through a low-rank fusion module followed by dense layers with ReLU activations.

- **TFN (Tensor Fusion Network)**  
  Same architecture as the original TFN, adapted to our setting. Bi-LSTM encoders produce modality-specific embeddings, which are fused via tensor products. Outputs are downsampled to reduce GPU memory usage before final prediction layers.

---

### `kinetics_fm_models.py`
Implements two recent models based on fusion of kinematic features and temporal reasoning:

- **Kinetics-FM-DLR-Net**  
  Reimplemented in PyTorch from Keras for consistency. Combines kinetics features with deep temporal modeling and dynamic label refinement.

- **DL-Kinetics-FM-Net**  
  Also reimplemented in PyTorch. An enhanced variant incorporating domain-level knowledge alongside standard kinetics feature fusion.

---

## 📚 Dataset Information
These models are evaluated on:
- **Dataset A**: 100-sample windows (0.5s)  
- **Dataset B**: 50-sample windows (0.5s)  
Input modalities include IMU, EMG, and optionally video-based features.

---

## 🚀 Requirements
- numpy==1.20.3
- pandas==1.3.4
- matplotlib==3.4.3
- scikit-learn==0.24.2
- scipy==1.7.1
- seaborn==0.11.2
- torch==1.13.1+cu117
- torchvision==0.14.1+cu117
- torchaudio==0.13.1+cu117
- h5py==3.3.0
- tqdm==4.62.

## 🧪 Running the Models

You can run each script as a standalone training/evaluation module:
```bash
python ffn_hf_ffn_bilstm_convnet.py
python lmfn_tfn_fusion_models.py
python kinetics_fm_models.py

