# Model Implementations for Kinetics Estimation from Wearable and Multimodal Data

This repository contains PyTorch implementations of multiple state-of-the-art (SOTA) models used in our study for human kinetics estimation using wearable sensors, EMG signals, and video data.

Each file corresponds to one or more related models described in the manuscript, implemented from scratch or adapted from previous work.

---

## 📁 Model Files

### `Dataset_A_ffn_hf_ffn_bilstm_convnet.py` and `Dataset_B_ffn_hf_ffn_bilstm_convnet.py`
Implement the following baseline deep learning models for Dataset A and Dataset B respectively:

- **FFN (HF)**  
- **FFN**  

- **Bi-LSTM**  

- **2D Conv. Network**  

---

### `Dataset_A_lmfn_tfn_fusion_models.py` and `Dataset_B_lmfn_tfn_fusion_models.py`
Multimodal fusion models for Dataset A and Dataset B:

- **LMFN (Low-Rank Multimodal Fusion Network)**  

- **TFN (Tensor Fusion Network)**  


---

### `Dataset_A_kinetics_fm_models.py` and `Dataset_B_kinetics_fm_models.py`

- **Kinetics-FM-DLR-Net**  

- **DL-Kinetics-FM-Net**  

---

### `Dataset_A_Sensor_Distillation.py` and `Dataset_B_sensor_distillation.py`
Implements **Sensor Distillation** models:

- Train a teacher model on full sensor modalities.
- Distill knowledge to a student model with fewer sensors (e.g., Sparse IMUs).

---

### `Dataset_A_model_ablation.py` and `Dataset_B_model_ablation.py`
Implements **model ablation studies**:

- Ablates key model components (e.g., fusion layers, temporal modules).
- Helps analyze the performance and contribution of each architectural block.
- Results provide justification for full model complexity.

---

## 📚 Dataset Information

These models are evaluated on:

- **Dataset A**: 100-sample windows (0.5s sampling @ 200Hz)  
- **Dataset B**: 50-sample windows (0.5s sampling @ 100Hz)  
- Input modalities include IMU, EMG, and optionally video-based features.

---

## 🚀 Requirements

```bash
numpy==1.20.3
pandas==1.3.4
matplotlib==3.4.3
scikit-learn==0.24.2
scipy==1.7.1
seaborn==0.11.2
torch==1.13.1+cu117
torchvision==0.14.1+cu117
torchaudio==0.13.1+cu117
h5py==3.3.0
tqdm==4.62.3
