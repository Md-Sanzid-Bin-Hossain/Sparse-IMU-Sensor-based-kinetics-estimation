# Lightweight and Efficient Kinetics Estimation with Sparse IMUs via Multi-Modal Sensor Distillation

> **Paper:** *Hossain, Hadley, Guo, Choi — IEEE Sensors Journal, 2024.*
> **Code:** [github.com/Md-Sanzid-Bin-Hossain/Sparse-IMU-Sensor-based-kinetics-estimation](https://github.com/Md-Sanzid-Bin-Hossain/Sparse-IMU-Sensor-based-kinetics-estimation)

This repository contains the PyTorch implementation of **Kinetics-MFFM-Net** and a **Sensor Distillation (SD)** framework for estimating joint moments and ground reaction forces (GRFs) from wearable sensors. The method achieves up to ~30% lower NRMSE than state-of-the-art models while using up to ~130× fewer parameters.

---

## 📑 Table of Contents

1. [TL;DR](#-tldr)
2. [Why This Work Matters](#-why-this-work-matters)
3. [Datasets](#-datasets)
4. [Method Overview](#-method-overview)
5. [Architecture Deep-Dive](#-architecture-deep-dive)
6. [Sensor Distillation (SD)](#-sensor-distillation-sd)
7. [Repository Structure](#-repository-structure)
8. [Reproduction Guide](#-reproduction-guide)
9. [Results Summary](#-results-summary)
10. [Limitations & Future Work](#-limitations--future-work)
11. [Citation](#-citation)

---

## ⚡ TL;DR

- **Goal:** Estimate **joint moments** and **3D ground reaction forces** from a **sparse set of IMUs** (as few as 1–2), avoiding optical motion capture and force plates.
- **Key Idea:** A *teacher* model is trained with a rich sensor set (full IMUs + EMG or smartphone video). A *student* model with sparse IMUs is trained to mimic both the teacher's **outputs** and its **internal feature representations**.
- **Architecture:** **Kinetics-MFFM-Net** — per-modality Gated Dual-Encoder Networks (Bi-LSTM + Bi-GRU) → Multi-Modal Feature Fusion Module (MFFM) → FC head.
- **Result:** With just **1 foot IMU** (Dataset A) or **2 foot IMUs + pelvis** (Dataset B), the student matches or beats much larger SOTA models — at **~130× fewer parameters** and **~7–10× lower inference latency**.

---

## 🌍 Why This Work Matters

Joint moments and GRFs are essential biomechanical quantities for:

- **Clinical assessment** of gait disorders — Parkinson's, multiple sclerosis, knee osteoarthritis (~14M cases in the U.S. alone).
- **Prosthetics & exoskeleton control** — these devices need fast, reliable kinetic feedback to assist or augment movement.
- **Rehabilitation & remote monitoring** — out-of-lab biomechanical assessment is currently very difficult.

**The conventional pipeline is impractical for real-world use:**

| Component | Cost | Limitation |
|---|---|---|
| Optical motion capture cameras | ~$70K | Lab-bound |
| Force plates | ~$60K | Floor-embedded |
| Reflective markers + analysis | ~$1K + days of expert labor | Slow, location-restricted |

**Wearable alternatives have their own problems:** musculoskeletal models need **15–17 IMUs** worn over the whole body; EMG suffers from skin impedance and motion artifacts; wearable force plates are heavy and disrupt natural gait.

**Sparse IMUs are the practical answer**, but they lose information critical for accurate kinetics estimation. Existing deep-learning solutions either underperform (generic architectures) or are massive (~250M parameters, unsuitable for embedded deployment).

This work closes the gap: **a lightweight architecture + cross-sensor knowledge transfer**, validated on diverse locomotion modes (treadmill, level-ground, ramp, stair).

---

## 📊 Datasets

Two publicly available gait datasets, with deliberately different sensor modalities to test generalization.

### Dataset A — Camargo et al. [45]
- **Participants:** 20 (12 M, 8 F) — original 22, two excluded for data quality.
- **Sensors:** 4 IMUs (thigh, shank, foot, torso) + 11 EMGs on major lower-limb muscles.
- **Locomotion modes:** treadmill (variable speeds), level-ground (5 CW + 5 CCW loops × 3 speeds), ramps (6 inclinations), stairs (4 heights).
- **Ground truth:** force plates (Bertec) + optical motion capture.
- **Targets (7 outputs):** hip flexion moment, hip abduction moment, knee flexion moment, ankle moment, 3D GRFs (mediolateral, anteroposterior, vertical).
- **Window:** ΔT = 100 samples (0.5 s @ 200 Hz).

### Dataset B — Tan et al. [35]
- **Participants:** 17 (all male).
- **Sensors:** 8 IMUs (trunk, pelvis, both thighs, both shanks, both feet) + 2 smartphone cameras → 2D joint centers via OpenPose (shoulder, hip, knee, ankle).
- **Locomotion modes:** instrumented treadmill walking under varied conditions — 3 speeds × 3 foot progression angles × 3 step widths × 3 trunk sway angles.
- **Targets (5 outputs):** KFM, KAM, 3D GRFs.
- **Window:** ΔT = 50 samples (0.5 s @ 100 Hz).

### Pre-processing pipeline

| Step | Dataset A | Dataset B |
|---|---|---|
| IMU sampling | 200 Hz | 100 Hz |
| GRF resampling | 1000 Hz → 200 Hz | 1000 Hz → 100 Hz |
| EMG processing | band-pass 20–460 Hz → rectify → low-pass 6 Hz → resample to 200 Hz | N/A |
| Video → joint centers | N/A | OpenPose (Body 25), resampled to match IMU |
| Segmentation | per gait cycle (force-plate contacts) | continuous (treadmill); zero-padding removed |
| Normalization | EMG / max activation; GRF / weight; moments / (height × weight) | same as A |

---

## 🧠 Method Overview

The framework operates in **three phases**:

```
Phase 1: Train Teacher                Phase 2: Build Student            Phase 3: Sensor Distillation
─────────────────────                  ──────────────────                ──────────────────────────
 [All IMU acc + gyr]                    [Sparse IMU acc + gyr]            Teacher (frozen)
        +                                       │                                │
 [EMG] or [Joint centers]                       ▼                                ▼  features + outputs
        │                                Kinetics-MFFM-Net(S)           ┌─────────────────┐
        ▼                                       │                       │  L_student      │
 Kinetics-MFFM-Net(T)                           ▼                       │  + α · L_SD1    │ ← outputs
        │                                  [7 or 5 targets]             │  + β · L_SD2    │ ← features
        ▼                                                               └─────────────────┘
 [7 or 5 targets]                                                                │
                                                                                 ▼
                                                                       Student matches teacher
                                                                       even with sparse IMUs
```

Both teacher and student share the **same backbone** (GDEN + MFFM + FCM); they differ only in the number of input modalities. The teacher is trained alone, frozen, then used to supervise the student.

---

## 🏗️ Architecture Deep-Dive

### High-level pipeline

```
Per-modality input
   │
   ▼
[BatchNorm] ──→ [GDEN: Bi-LSTM + Bi-GRU + Gating] ──→ modality features  (one per modality)
                                                            │
                                                            ▼ concat across modalities
                                                       [MFFM]
                                                            │
                                                            ▼
                                                       [FCM: Linear] ──→ kinetics targets
```

### 1️⃣ Gated Dual-Encoder Network (GDEN)

Each modality (acc, gyr, emg/video) has its **own** GDEN. Within a GDEN:

```
                    ┌──────────────► Bi-LSTM(2-layer) ──► X_lstm ─┐
input (B, T, D)     │                                              ├─► gate ─► X_gated
                    └──────────────► Bi-GRU (2-layer) ──► X_gru  ─┘
```

**Why two encoders + a gate?** Bi-LSTM and Bi-GRU capture slightly different temporal dynamics. Rather than committing to one (which is dataset-dependent), the gate learns per-feature, per-timestep how to mix them:

$$
W_{\text{gate}} = \sigma\!\big(\text{FC}([X_{\text{lstm}}, X_{\text{gru}}])\big)
$$

$$
X_{\text{gated}} = W_{\text{gate}} \odot X_{\text{lstm}} + (1 - W_{\text{gate}}) \odot X_{\text{gru}}
$$

The output of GDEN for each modality is a sequence of fused features, e.g. $X_{\text{acc, gated}}$, $X_{\text{gyr, gated}}$, $X_{\text{emg/vid, gated}}$.

### 2️⃣ Multi-Modal Feature Fusion Module (MFFM)

After per-modality encoding, features are concatenated:

$$
X_{\text{concat}} = [X_{\text{acc, gated}};\ X_{\text{gyr, gated}};\ X_{\text{emg/vid, gated}}]
$$

The MFFM then routes $X_{\text{concat}}$ through **three parallel fusion branches**, then fuses those branches:

```
                      ┌──► MWFS  (modality weighting & sum)  ─┐
                      │                                       │
X_concat ─────────────┼──► MWFF  (channel weighting)         ─┼──► concat ──► MMF gate ──► X_mmf
                      │                                       │
                      └──► MAF   (multi-head self-attention) ─┘
```

**(i) Multi-modal Weighted Feature Summation (MWFS).** Each modality gets its own scalar weight, then weighted sum across modalities. This *reduces* dimensionality and asks the model "how important is each modality at each timestep?"

$$
W_m = \sigma(\text{FC}(X_{m,\text{gated}})), \quad m \in \{\text{acc, gyr, emg/vid}\}
$$

$$
X_{\text{mwfs}} = \sum_{m} W_m \odot X_{m,\text{gated}}
$$

**(ii) Multi-modal Weighted Feature Fusion (MWFF).** Channel-wise gating on the concatenated features — emphasizes important channels:

$$
X_{\text{mwff}} = \sigma(\text{FC}(X_{\text{concat}})) \odot X_{\text{concat}}
$$

**(iii) Multi-modal Attention Fusion (MAF).** Standard multi-head self-attention captures temporal and cross-modal long-range dependencies:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

with $Q = X_{\text{concat}} W_Q$, $K = X_{\text{concat}} W_K$, $V = X_{\text{concat}} W_V$, multi-head with $h$ heads.

**(iv) Multi-modal Module Fusion (MMF).** Concatenate the three branches, then apply a final gate:

$$
X_{\text{concat,mmf}} = [X_{\text{mwfs}};\ X_{\text{mwff}};\ X_{\text{maf}}]
$$

$$
X_{\text{mmf}} = \sigma(\text{FC}(X_{\text{concat,mmf}})) \odot X_{\text{concat,mmf}}
$$

### 3️⃣ Fully Connected Module (FCM)

A simple linear projection to the target space:

$$
\hat{K} = \text{FC}(X_{\text{mmf}}) \in \mathbb{R}^{B \times \Delta T \times D_k}
$$

where $D_k = 7$ for Dataset A and $D_k = 5$ for Dataset B.

### Why this design?

| Component | What it solves |
|---|---|
| **Per-modality GDEN** | Different sensor modalities have different statistics — encoding them separately preserves modality-specific information before fusion. |
| **Bi-LSTM + Bi-GRU + gate** | Robustness to dataset/modality variation — neither RNN type dominates everywhere. |
| **MWFS** | Forces the model to reason about **modality importance** at each timestep. |
| **MWFF** | Reweights individual feature channels — fine-grained importance. |
| **MAF** | Captures long-range **temporal and cross-modal** dependencies. |
| **MMF** | Lets the network choose which fusion strategy to rely on per timestep. |

---

## 🎓 Sensor Distillation (SD)

### The intuition

The teacher sees more than the student does: full IMUs + EMG (Dataset A) or + joint centers (Dataset B). When the teacher predicts kinetics, its **internal 128-dim feature representation** carries information from those extra modalities. If we force the student to produce a similar representation — even though the student sees only sparse IMUs — the student is implicitly learning to "hallucinate" what EMG or video would have told it.

### The loss

The student is trained with **three loss terms**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{student}} + \alpha \cdot \mathcal{L}_{\text{SD1}} + \beta \cdot \mathcal{L}_{\text{SD2}}
$$

| Term | Compares | Meaning |
|---|---|---|
| $\mathcal{L}_{\text{student}}$ | student output vs. ground truth | standard task RMSE |
| $\mathcal{L}_{\text{SD1}}$ | student output vs. teacher output | **output-level distillation** (KD) |
| $\mathcal{L}_{\text{SD2}}$ | student linearly-projected features vs. teacher's | **feature-level distillation** (sensor distillation) |

All three are RMSE losses. In our experiments: **α = 0.50, β = 1.00**.

Formally, for a batch of size $B$ and window $\Delta T$:

$$
\mathcal{L}_{\text{student}} = \sqrt{\frac{1}{B \Delta T} \sum_{b,k} \big(K_{\text{gt}}^{b,k} - K_{\text{est}}^{\text{student},\,b,k}\big)^2}
$$

$$
\mathcal{L}_{\text{SD1}} = \sqrt{\frac{1}{B \Delta T} \sum_{b,k} \big(K_{\text{est}}^{\text{student},\,b,k} - K_{\text{est}}^{\text{teacher},\,b,k}\big)^2}
$$

$$
\mathcal{L}_{\text{SD2}} = \sqrt{\frac{1}{B \Delta T} \sum_{b,k} \big(X_{\text{concat,lin}}^{\text{student},\,b,k} - X_{\text{concat,lin}}^{\text{teacher},\,b,k}\big)^2}
$$

A linear projection (`fc_kd` in the code) maps the student's pre-fusion features to the **same 128-dim space** as the teacher's, so $\mathcal{L}_{\text{SD2}}$ can be computed.

### Training loop

```python
teacher.eval()  # frozen after Phase 1

for epoch in range(num_epochs):
    for batch in train_loader:
        # Student forward (sparse IMUs)
        K_student, X_student_feat = student(acc_sparse, gyr_sparse)

        # Teacher forward (full sensor set), no grad
        with torch.no_grad():
            K_teacher, X_teacher_feat = teacher(acc_full, gyr_full, emg_or_video)

        loss = RMSE(K_student, target) \
             + alpha * RMSE(K_student, K_teacher) \
             + beta  * RMSE(X_student_feat, X_teacher_feat)

        loss.backward(); optimizer.step()
```

### Why this beats naive student training

In the paper's ablations, swapping a 3-IMU naive student for a 2-IMU SD student gives **the same accuracy with one less sensor** — the distilled feature space compensates for the missing input. This is the headline result.

---

## 📁 Repository Structure

```
Sparse-IMU-Sensor-based-kinetics-estimation/
│
├── Dataset_A_ffn_hf_ffn_bilstm_convnet.py       # External baselines: FFN(HF), FFN, Bi-LSTM, 2D Conv
├── Dataset_B_ffn_hf_ffn_bilstm_convnet.py
│
├── Dataset_A_lmfn_tfn_fusion_models.py          # External baselines: LMFN, TFN
├── Dataset_B_lmfn_tfn_fusion_models.py
│
├── Dataset_A_kinetics_fm_models.py              # External baselines: Kinetics-FM-DLR-Net, DL-Kinetics-FM-Net
├── Dataset_B_kinetics_fm_models.py
│
├── Dataset_A_Sensor_Distillation.py             # ⭐ MAIN: Teacher + Student + SD (this paper)
├── Dataset_B_sensor_distillation.py
│
├── Dataset_A_model_ablation.py                  # Internal baselines: Early Fusion, Feature Concat, MFFM-only
└── Dataset_B_model_ablation.py
```

### What each file does

| File group | Models implemented | Purpose |
|---|---|---|
| `*_ffn_hf_ffn_bilstm_convnet.py` | FFN (HF), FFN, Bi-LSTM, 2D Conv. Network | Generic deep-learning baselines from prior work [31, 34, 36, 38, 39]. |
| `*_lmfn_tfn_fusion_models.py` | LMFN, TFN | Multimodal fusion baselines [35]. |
| `*_kinetics_fm_models.py` | Kinetics-FM-DLR-Net, DL-Kinetics-FM-Net | Heavy IMU-specific baselines [40, 41]. |
| `*_Sensor_Distillation.py` | **Kinetics-MFFM-Net (T)**, **Kinetics-MFFM-Net (S)**, **+ SD** | Main contribution: trains teacher, then 4 students at increasing sensor counts, with and without distillation. |
| `*_model_ablation.py` | Early Fusion (Bi-LSTM/GRU), Feature Concat (Bi-LSTM/GRU), MFFM (Bi-LSTM/GRU) | Internal ablations to justify each architectural component. |

### Inside `Dataset_A_Sensor_Distillation.py`

The script trains and evaluates the full pipeline for **one LOSO test subject**. For each student configuration, it produces 9 ablation rows: the teacher, four naive students (1, 2, 3, 4 IMUs), four SD students (1, 2, 3, 4 IMUs).

Key classes (all defined inline — Colab-style single-cell code):

| Class | Role |
|---|---|
| `Encoder_1` | 2-layer Bi-LSTM with dropout |
| `Encoder_2` | 2-layer Bi-GRU with dropout |
| `GatingModule` | Sigmoid-gated fusion of two encoder outputs |
| `teacher` | Full Kinetics-MFFM-Net(T) with 3 modalities (acc, gyr, EMG) |
| `student_1`…`student_4` | Naive sparse-IMU students (no SD), 1 → 4 IMUs |
| `student_1_KD`…`student_4_KD` | SD-trained students (return both prediction and `x_KD` features) |

Sensor configurations for Dataset A students:

| Student | Channels | Sensors |
|---|---|---|
| 1 | acc[0:3], gyr[0:3] | Foot |
| 2 | acc[0:3]+[9:12], gyr[0:3]+[9:12] | Foot + Torso |
| 3 | acc[0:6]+[9:12], gyr[0:6]+[9:12] | Foot + Shank + Torso |
| 4 | acc[0:9]+[9:12], gyr[0:9]+[9:12] | Foot + Shank + Thigh + Torso |

Output: a CSV per LOSO subject with NRMSE and PCC for all 9 ablations × 7 targets.

---

## 🚀 Reproduction Guide

### Environment

```bash
# Tested with these exact versions
python==3.8+
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

Hardware used in the paper:
- **Training:** NVIDIA TITAN XP
- **Inference benchmarking:** NVIDIA T4

### Data preparation

1. Download Dataset A from [Camargo et al. 2021](https://doi.org/10.1016/j.jbiomech.2021.110320) and Dataset B from [Tan et al. 2022](https://doi.org/10.1109/TII.2022.3225507).
2. Pre-process (band-pass + rectify + low-pass for EMG; resample GRFs; segment per gait cycle).
3. Store as a single HDF5 with one group per subject — the loader in the script expects:
   ```
   /All_subjects/Subject_<id>/Treadmill, /Levelground, /Ramp, /Stair
   ```
   where each leaf is an `(N_samples, 129)` float array (column layout below).

**Column layout (Dataset A, 129 cols):**

| Index range | Content |
|---|---|
| `0:24` | 4 IMUs × (acc xyz + gyr xyz) |
| `24:47` | Joint kinematics (23 channels) |
| `47:70` | Joint kinetics (full) |
| `70:79` | 9-channel GRF block (we use first 3) |
| `79:84` | Goniometer (5 channels) |
| `84:106` | EMG (22 channels — we use 11) |
| `106:129` | Joint power (23 channels) |

The script extracts: 24 IMU + 12 kinematics + 11 EMG + 7 JP + 5 goniometer + 4 kinetics + 3 GRF = **66 input cols**, with cols 59–66 reserved for the **7 prediction targets**.

### Training pipeline (one LOSO subject, e.g. Subject_30)

```bash
# Phase 1: Train teacher
python Dataset_A_Sensor_Distillation.py
# This single script:
#   - loads all 20 subjects, holds out Subject_30 as test
#   - trains the teacher (full IMU + EMG)
#   - trains 4 naive students (1, 2, 3, 4 IMUs) — no SD
#   - trains 4 SD students (1, 2, 3, 4 IMUs) using frozen teacher
#   - evaluates each on the LOSO subject
#   - saves: <subject>_<encoder>_KD_SD_results.csv
```

### Hyperparameters (held constant for fair comparison)

| Hyperparameter | Value |
|---|---|
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | RMSE |
| Epochs (max) | 40 |
| Early stopping patience | 10 |
| Validation split | 20% |
| Random seed | 42 |
| Window size | 100 (Dataset A), 50 (Dataset B) |
| Distillation weights | α = 0.50, β = 1.00 |

### Evaluation metrics

- **NRMSE (%)** = RMSE normalized by the range of the ground truth, expressed as a percentage. Per-target.
- **PCC** = Pearson correlation coefficient between prediction and ground truth. Per-target.
- Both reported as **mean ± std across LOSO subjects**.
- Statistical tests: repeated-measures ANOVA + Bonferroni post-hoc (across models); paired *t*-test (Kinetics-MFFM-Net(S) vs. +SD), with significance at *p* < 0.05.

### Sanity-check before running

- HDF5 path in the script: `/home/sanzidpr/Dataset A-Kinetics/All_subjects_data.h5` — **edit this** to match your environment.
- Output path: `/home/sanzidpr/Journal_3/Dataset_A_model_results/Subject<N>/` — **edit this**.
- The script trains for one held-out subject at a time. To get full LOSO results, loop over all 20 subjects (e.g., wrap in a shell script that swaps the test subject ID).

---

## 📈 Results Summary

### Internal ablations (Table I of the paper)

Progressive component additions, both datasets, all modalities:

| Configuration | Dataset A NRMSE (%) | Dataset B NRMSE (%) |
|---|---|---|
| Early Fusion (Bi-LSTM) | 4.63 ± 0.75 | 4.12 ± 0.47 |
| Early Fusion (Bi-GRU) | 4.65 ± 0.83 | 4.07 ± 0.45 |
| Feature Concat (Bi-LSTM) | 4.80 ± 0.82 | 3.95 ± 0.42 |
| Feature Concat (Bi-GRU) | 4.72 ± 0.70 | 4.00 ± 0.45 |
| MFFM (Bi-LSTM) | 4.25 ± 0.71 | 3.67 ± 0.41 |
| MFFM (Bi-GRU) | 4.19 ± 0.76 | 3.68 ± 0.48 |
| **Kinetics-MFFM-Net (T)** | **4.16 ± 0.77** | **3.61 ± 0.47** |

**Takeaway:** MFFM gives the biggest jump; GDEN adds small but consistent gains. All variants stay under 3.5M parameters and ~6 ms inference.

### SOTA comparison with all modalities (Table II)

| Model | Dataset A NRMSE | Dataset B NRMSE | Params (M) |
|---|---|---|---|
| FFN (HF) | 6.04 | 5.43 | 0.5–0.9 |
| FFN | 5.56 | 4.32 | 1.8–9.1 |
| Bi-LSTM | 4.93 | 4.21 | 5.2–19.8 |
| LMFN | 4.89 | 4.55 | 0.1–0.2 |
| TFN | 4.69 | 4.05 | 4.7–4.8 |
| **Kinetics-MFFM-Net (T)** | **4.16** | **3.61** | **3.3–3.5** |

**Takeaway:** Up to **31% lower NRMSE** than Bi-LSTM and FFN baselines; statistically significant (*p* < 0.05) against all but TFN, where the margin is still ≥10%.

### SOTA comparison with sparse IMUs (Tables III–IV)

| IMU config | Best baseline (NRMSE / Params) | Kinetics-MFFM-Net(S)+SD (NRMSE / Params) | Speedup |
|---|---|---|---|
| Dataset A — 2 IMUs (Foot+Torso) | DL-Kinetics-FM-Net: 5.06 / 254M | **4.91** / **1.86M** | ~9× faster |
| Dataset A — 4 IMUs | Kinetics-FM-DLR-Net: 4.91 / 254M | **4.75** / **1.90M** | ~9× faster |
| Dataset B — 3 IMUs (Feet+Pelvis) | DL-Kinetics-FM-Net: 4.40 / 77M | **4.38** / **1.90M** | ~8× faster |
| Dataset B — 8 IMUs | DL-Kinetics-FM-Net: 3.95 / 78M | **3.90** / **1.95M** | ~8× faster |

**The big punchlines:**
1. **Sensor compensation:** Naive 3-IMU student → NRMSE 4.98. SD 2-IMU student → NRMSE 4.91. **One fewer IMU, same accuracy.**
2. **Parameter efficiency:** Up to **130×** fewer params than DL-Kinetics-FM-Net for matching or better accuracy (Dataset A).
3. **Real-time ready:** Inference under **4.25 ms/sample** on T4 GPU, well below the 0.5 s window — leaves headroom for embedded deployment.

---

## ⚠️ Limitations & Future Work

**Acknowledged in the paper:**

1. **Activity scope** — datasets focus on walking; running, jumping, and transitions need testing.
2. **Demographics** — both datasets are young adults; clinical populations (elderly, post-stroke, amputees) need separate validation.
3. **Sensor diversity** — pressure insoles or other modalities could further improve GRF estimation.
4. **Real-time validation** — inference latency is benchmarked, but full embedded deployment with wireless IMUs is future work.
5. **Privacy** — federated learning could enable cross-clinic adaptation without sharing raw data.

**Implicit but worth noting:**

- The teacher's 128-dim bottleneck size is fixed; ablating bottleneck dimensionality could further compress.
- Distillation weights (α=0.5, β=1.0) were chosen empirically; learnable or scheduled weights could help.
- The current student receives the same window length as the teacher — for prosthetic control, *forecasting* (predicting 100–200 ms ahead) is the natural extension.

---

## 📎 Citation

```bibtex
@article{hossain2024kineticsmffm,
  title   = {Lightweight and Efficient Kinetics Estimation with Sparse IMUs via Multi-Modal Sensor Distillation},
  author  = {Hossain, Md Sanzid Bin and Hadley, Dexter and Guo, Zhishan and Choi, Hwan},
  journal = {IEEE Sensors Journal},
  year    = {2024}
}
```

**Funded by:** NSF Grants FRR-2246671 and FRR-2246672.

---

## 🙋 Questions

For questions about the code or extensions, contact the corresponding author or open a GitHub issue. The most likely things to need adapting for a new dataset are:

1. **Channel layout** (the `0:24`, `24:47`, etc. column slices in the data loader).
2. **Sensor masks** for sparse student configurations (the `data_acc[:,:,0:3]` etc. slicing in each `train_mm_student_*` function).
3. **Number of targets** — change `output_dim`, the final FC out-features (`nn.Linear(..., 7)`), and `m1`/`m2` in the data slicing.
