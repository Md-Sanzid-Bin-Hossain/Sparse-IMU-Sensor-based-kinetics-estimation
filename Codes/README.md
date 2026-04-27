# Model Implementations for Kinetics Estimation from Wearable and Multimodal Data

This repository contains PyTorch implementations of multiple state-of-the-art (SOTA) models used in our study for human kinetics estimation using wearable sensors, EMG signals, and video data.

Each file corresponds to one or more related models described in the manuscript, implemented from scratch or adapted from previous work.

---

## 📁 Model Files

### `Dataset_A_ffn_hf_ffn_bilstm_convnet.py` and `Dataset_B_ffn_hf_ffn_bilstm_convnet.py`
Implement the following baseline deep learning models for Dataset A and Dataset B respectively:
- **FFN (HF)** — Feed-Forward Network with hand-crafted features
- **FFN** — Feed-Forward Network
- **Bi-LSTM** — Bidirectional Long Short-Term Memory
- **2D Conv. Network** — 2D Convolutional Network

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
Implements our proposed **Kinetics-MFFM-Net with Knowledge + Sensor Distillation** framework. See the [detailed explanation below](#-kinetics-mffm-net--knowledge--sensor-distillation-detailed).

---

### `Dataset_A_model_ablation.py` and `Dataset_B_model_ablation.py`
Implements **model ablation studies**:
- Ablates key model components (e.g., fusion layers, temporal modules).
- Helps analyze the performance and contribution of each architectural block.
- Results provide justification for full model complexity.

---

## 🧠 Kinetics-MFFM-Net + Knowledge & Sensor Distillation (Detailed)

This is the **core contribution** of our work. The framework predicts joint kinetics (joint moments and ground reaction forces — 7 targets total) from a reduced set of IMU sensors, by transferring knowledge from a richer multimodal teacher.

### 🎯 The Problem We Solve

A high-performing teacher uses **all four IMUs (acc + gyr) plus EMG signals**. But in deployment (e.g., wearables, prosthetics), we usually only have a few IMUs and no EMG. A naïvely trained small model on sparse IMUs underperforms. Our solution: train the teacher first, then distill both its **output predictions** and its **intermediate feature representations** into a series of student models that progressively use fewer sensors.

### 🏗️ Architecture Overview

Both teacher and students share the same backbone, called the **Multi-Modal Fusion Module (MFFM)**:

```
Per-modality input → BatchNorm → Encoder_1 (BiLSTM) ─┐
                                                     ├→ GatingModule → modality features
                              → Encoder_2 (BiGRU) ───┘

Concatenated modalities → [Self-Attention | Channel-Gating | Feature-Weighting] → Final Gate → FC → 7 outputs
```

### 🔧 Building Blocks

**1. Dual-Encoder per Modality (`Encoder_1` + `Encoder_2`)**
Each modality (acc, gyr, emg) is processed by two parallel temporal encoders:
- `Encoder_1`: 2-layer Bidirectional **LSTM** (128 → 64 hidden, dropout between layers)
- `Encoder_2`: 2-layer Bidirectional **GRU** (128 → 64 hidden, dropout between layers)

Both output a 128-dim feature sequence (T, 128). Using LSTM and GRU together captures complementary temporal dynamics.

**2. GatingModule (Encoder Fusion)**
Combines the LSTM and GRU outputs adaptively per timestep:
```
gate = sigmoid(Linear([lstm_out, gru_out]))
fused = gate * lstm_out + (1 - gate) * gru_out
```
This lets the network learn — at each timestep — whether LSTM or GRU features are more useful, instead of just concatenating them.

**3. Three-Branch Fusion Head (the "MFFM" core)**
After all modalities are encoded, their features are concatenated along the channel axis: `x ∈ (B, T, M·128)` where M = 3 for the teacher (acc/gyr/emg) and M = 2 for students (acc/gyr). Three parallel branches process this:

| Branch | Mechanism | Purpose |
|---|---|---|
| **1. Self-Attention** | `MultiheadAttention(x, x, x)` with 4 heads | Captures long-range temporal dependencies |
| **2. Channel Gating** | `sigmoid(Linear(x)) * x` | Learns which feature channels matter |
| **3. Modality Weighting** | Per-modality scalar weights via `weighted_feat`, then weighted sum across modalities | Learns relative importance of acc vs. gyr vs. emg |

The three branch outputs are concatenated, passed through a final gate, and projected to the 7 kinetics targets via a FC layer.

**4. Distillation Bottleneck (`fc_kd`)**
A linear projection that maps the student's pre-fusion features to the **same 128-dim space** as the teacher's features. This is what enables feature-level (sensor) distillation — the L2 distance between teacher and student embeddings can only be computed because both live in the same space.

### 👨‍🏫 Teacher Model

- **Inputs:** 4 IMUs × (acc + gyr) = 12 + 12 channels, plus 11 EMG channels
- **Encoders:** 3 modalities × 2 encoders = 6 dual-encoders
- **Trained alone** with RMSE loss against the 7 kinetics targets
- Saved as `_teacher.pth` and frozen for all student training

### 🎓 Student Models (4 Sensor Configurations)

Students are trained at increasing sensor counts to study the trade-off between hardware complexity and accuracy:

| Student | Sensors Used | acc/gyr channels |
|---|---|---|
| Student 1 | Foot only | 3 + 3 |
| Student 2 | Foot + Pelvis | 6 + 6 |
| Student 3 | Foot + Pelvis + Shank | 9 + 9 |
| Student 4 | Foot + Pelvis + Shank + Thigh | 12 + 12 |

Each student has the **same MFFM backbone** but with only 2 modalities (acc, gyr — no EMG).

### 📉 The Distillation Loss

For each student, training combines three loss terms:

```
L_total = L_task  +  α · L_KD  +  β · L_SD

where:
  L_task = RMSE(student_output, ground_truth)        # supervised target loss
  L_KD   = RMSE(student_output, teacher_output)      # knowledge distillation (output-level)
  L_SD   = RMSE(student_features, teacher_features)  # sensor distillation (feature-level)
```

In our setup: **α = 0.50, β = 1.00**.

- `L_KD` softens the targets using the teacher's predictions — a standard knowledge distillation idea applied to regression.
- `L_SD` is our **sensor distillation** term: it forces the student's 128-dim bottleneck (`x_KD`) to match the teacher's 128-dim bottleneck. This is where the teacher's knowledge of "what EMG and the missing IMUs were telling us" gets transferred into the student's representation, even though the student never sees those inputs.

### 🔁 Training Loop (per student)

```python
for batch in train_loader:
    student_out, student_feat = student(student_inputs)
    with torch.no_grad():
        teacher_out, teacher_feat = teacher(full_inputs)   # teacher frozen

    loss = RMSE(student_out, target) \
         + α * RMSE(student_out,  teacher_out) \
         + β * RMSE(student_feat, teacher_feat)

    loss.backward(); optimizer.step()
```

- Optimizer: Adam, lr = 1e-3
- Early stopping: patience = 10 epochs on validation loss
- Max epochs: 40
- Window size: 100 samples; batch size: 64

### 📊 Evaluation

For each LOSO test subject, we compute per-target metrics over all 7 outputs (4 joint moments + 3 GRF components):
- **NRMSE (%)** — RMSE normalized by the target's range
- **PCC** — Pearson correlation coefficient between prediction and ground truth

Results from each ablation (teacher, baseline students, KD+SD students) are stacked into a single CSV per subject.

### 💡 Why This Works

1. **Teacher sees everything** — it builds a rich 128-dim representation that encodes information from EMG and all IMUs.
2. **Sensor distillation** forces the student's bottleneck to mimic that rich representation, so the student "hallucinates" the missing modalities from the IMUs it does have.
3. **Knowledge distillation** further regularizes the student's outputs toward the teacher's predictions, smoothing the supervision signal.
4. **MFFM backbone** ensures the student has enough representational capacity (dual encoders, 3-branch fusion) to actually absorb the distilled knowledge.

The result: a student with as few as **one foot IMU** can approach teacher-level performance on kinetics estimation — making the system practical for real wearable deployments.

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
```
