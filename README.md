# Lightweight and Efficient Kinetics Estimation with Sparse IMUs via Multi-Modal Sensor Distillation

> **Paper:** Hossain et al., *IEEE Sensors Journal*, 2026.

PyTorch implementation of **Kinetics-MFFM-Net** and a **Sensor Distillation (SD)** framework that estimates joint moments and ground reaction forces (GRFs) from a small number of IMUs — without optical motion capture or force plates.

---

## 📑 Contents

1. [Quick Overview](#-quick-overview)
2. [Why This Work Matters](#-why-this-work-matters)
3. [Datasets](#-datasets)
4. [Method](#-method)
5. [Architecture](#-architecture)
6. [Sensor Distillation](#-sensor-distillation)
7. [Repository Structure](#-repository-structure)
8. [How to Run](#-how-to-run)
9. [Adapting to Your Own Data](#-adapting-to-your-own-data)
10. [Limitations & Extensions](#-limitations--extensions)
11. [Citation](#-citation)

---

## ⚡ Quick Overview

- **Goal:** Estimate joint moments and 3D GRFs from a sparse set of IMUs (as few as 1–2).
- **Idea:** A *teacher* model trained with a rich sensor set (full IMUs + EMG or smartphone video) supervises a *student* model that only sees sparse IMUs. The student learns to mimic both the teacher's outputs **and** its internal feature representations.
- **Architecture:** **Kinetics-MFFM-Net** — per-modality Gated Dual-Encoder Networks (Bi-LSTM + Bi-GRU) → Multi-Modal Feature Fusion Module (MFFM) → FC head.
- **Outcome:** With as few as 1 foot IMU (Dataset A) or 2 feet + pelvis (Dataset B), the student matches much larger SOTA models — at a small fraction of their parameter count and inference latency.

---

## 🌍 Why This Work Matters

Joint moments and GRFs are essential for clinical gait assessment (Parkinson's, MS, knee osteoarthritis), prosthetic & exoskeleton control, and rehabilitation. The conventional pipeline — optical motion capture + force plates + musculoskeletal modeling — costs >$130K, is lab-bound, and takes days to process per session.

Wearable alternatives have their own problems:
- **Musculoskeletal IMU pipelines** need 15–17 sensors over the whole body.
- **EMG** suffers from skin impedance, motion artifacts, and inter-user variability.
- **Wearable force plates** are heavy and disrupt natural gait.

**Sparse IMUs are the practical answer**, but they lose information that's critical for accurate kinetics. Existing deep-learning solutions either underperform (generic architectures) or are massive (hundreds of millions of parameters, unsuitable for embedded deployment). This work closes the gap with a lightweight architecture and cross-sensor knowledge transfer, validated across treadmill, level-ground, ramp, and stair walking.

---

## 📊 Datasets

Two publicly available gait datasets, deliberately different in their non-IMU modality, to test generalization.

### Dataset A — Camargo et al. (2021)
- **Source:** [*A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions...*, Journal of Biomechanics, 119:110320](https://doi.org/10.1016/j.jbiomech.2021.110320)
- 20 participants (12 M, 8 F).
- 4 IMUs (thigh, shank, foot, torso) + 11 EMGs + force plates + motion capture.
- Locomotion: treadmill, level-ground, ramps (6 inclinations), stairs (4 heights).
- **Targets (7):** hip flexion moment, hip abduction moment, knee flexion moment, ankle moment, 3D GRF.
- Window ΔT = 100 samples (0.5 s @ 200 Hz).

### Dataset B — Tan et al. (2022)
- **Source:** [*IMU and Smartphone Camera Fusion for Knee Adduction and Knee Flexion Moment Estimation During Walking*, IEEE Transactions on Industrial Informatics](https://doi.org/10.1109/TII.2022.3225507)
- 17 participants (all male).
- 8 IMUs (trunk, pelvis, both thighs/shanks/feet) + 2 smartphone cameras → 2D joint centers via OpenPose.
- Locomotion: instrumented treadmill walking under varied speeds, foot progression angles, step widths, and trunk sway angles.
- **Targets (5):** KFM, KAM, 3D GRF.
- Window ΔT = 50 samples (0.5 s @ 100 Hz).

### Pre-processing summary

| Step | Notes |
|---|---|
| GRF resampling | 1000 Hz → matches IMU rate |
| EMG (Dataset A) | band-pass 20–460 Hz → rectify → low-pass 6 Hz → resample |
| Video (Dataset B) | OpenPose (Body 25) → 2D joint centers → resample |
| Segmentation | per gait cycle on overground/ramp/stair; continuous on treadmill |
| Normalization | EMG / max activation; GRF / weight; moments / (height × weight) |

---

## 🧠 Method

The framework runs in **three phases**:

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

Both teacher and student share the **same backbone** (GDEN + MFFM + FCM); they differ only in how many input modalities they receive. The teacher is trained alone, frozen, then used to supervise the student.

---

## 🏗️ Architecture

### Pipeline at a glance

```
Per-modality input  →  [BatchNorm]  →  [GDEN]  →  modality features
                                                      │
                                                      ▼ concat across modalities
                                                  [MFFM]  →  [FC]  →  kinetics targets
```

### 1. Gated Dual-Encoder Network (GDEN)

Each modality (acc, gyr, emg/video) gets its own GDEN:

```
                     ┌── Bi-LSTM(2-layer) ── X_lstm ──┐
input (B, T, D)  ────┤                                ├── gate ── X_gated
                     └── Bi-GRU (2-layer) ── X_gru  ──┘
```

**Why two encoders + a gate?** Bi-LSTM and Bi-GRU capture slightly different temporal dynamics. Rather than committing to one (which is dataset-dependent), the gate learns per-feature, per-timestep how to mix them:

$$
W_{\text{gate}} = \sigma\!\big(\text{FC}([X_{\text{lstm}}, X_{\text{gru}}])\big)
$$

$$
X_{\text{gated}} = W_{\text{gate}} \odot X_{\text{lstm}} + (1 - W_{\text{gate}}) \odot X_{\text{gru}}
$$

### 2. Multi-Modal Feature Fusion Module (MFFM)

After GDEN, modality features are concatenated and routed through **three parallel branches**, then fused:

```
                  ┌──► MWFS  (modality weighting & sum)  ─┐
                  │                                       │
X_concat  ────────┼──► MWFF  (channel weighting)         ─┼──► concat ──► MMF gate ──► X_mmf
                  │                                       │
                  └──► MAF   (multi-head self-attention) ─┘
```

| Branch | What it does | Why it helps |
|---|---|---|
| **MWFS** | Single scalar weight per modality; weighted sum | Asks *"how important is each modality at each timestep?"* and reduces feature complexity |
| **MWFF** | Channel-wise sigmoid gating on concatenated features | Reweights individual feature channels — fine-grained importance |
| **MAF** | Multi-head self-attention on the concatenated sequence | Captures long-range temporal and cross-modal dependencies |
| **MMF** | Concatenates the three branches, then a final sigmoid gate | Lets the network choose which fusion strategy to rely on per timestep |

### 3. Fully Connected Module (FCM)

A linear projection from $X_{\text{mmf}}$ to the kinetics targets ($D_k = 7$ for Dataset A, $D_k = 5$ for Dataset B).

### Why this design?

| Component | What it solves |
|---|---|
| Per-modality GDEN | Different sensors have different statistics — encoding them separately preserves modality-specific information before fusion |
| Bi-LSTM + Bi-GRU + gate | Robustness to dataset/modality variation — neither RNN type dominates everywhere |
| MFFM (three branches) | Captures complementary aspects of fusion — modality importance, channel importance, and temporal/cross-modal context |

---

## 🎓 Sensor Distillation

### The intuition

The teacher sees more than the student does — full IMUs + EMG (Dataset A) or + joint centers (Dataset B). When the teacher predicts kinetics, its internal feature representation carries information from those extra modalities. If we force the student to produce a similar representation — even though the student sees only sparse IMUs — the student is implicitly learning to **hallucinate what EMG or video would have told it**.

### The loss

The student is trained with three terms:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{student}} + \alpha \cdot \mathcal{L}_{\text{SD1}} + \beta \cdot \mathcal{L}_{\text{SD2}}
$$

| Term | Compares | Meaning |
|---|---|---|
| $\mathcal{L}_{\text{student}}$ | student output vs. ground truth | standard task RMSE |
| $\mathcal{L}_{\text{SD1}}$ | student output vs. teacher output | output-level distillation |
| $\mathcal{L}_{\text{SD2}}$ | student features vs. teacher features (via linear projection) | **feature-level (sensor) distillation** |

All three are RMSE losses. The paper uses **α = 0.50, β = 1.00**.

A small linear projection (`fc_kd` in the code) maps the student's pre-fusion features into the **same 128-dim space** as the teacher's, so $\mathcal{L}_{\text{SD2}}$ can be computed.

### Training loop (sketch)

```python
teacher.eval()  # frozen after Phase 1

for batch in train_loader:
    K_student, X_student_feat = student(acc_sparse, gyr_sparse)
    with torch.no_grad():
        K_teacher, X_teacher_feat = teacher(acc_full, gyr_full, emg_or_video)

    loss = RMSE(K_student, target) \
         + alpha * RMSE(K_student, K_teacher) \
         + beta  * RMSE(X_student_feat, X_teacher_feat)

    loss.backward(); optimizer.step()
```

### Why this beats naive student training

In the paper's results, an SD-trained student with **one fewer IMU** matches the accuracy of a naive student. The distilled feature space compensates for the missing input — that's the headline finding.

---

## 📁 Repository Structure

```
Sparse-IMU-Sensor-based-kinetics-estimation/
│
└── Codes/
    ├── Dataset_A_ffn_hf_ffn_bilstm_convnet.py       # Baselines: FFN(HF), FFN, Bi-LSTM, 2D Conv
    ├── Dataset_B_ffn_hf_ffn_bilstm_convnet.py
    │
    ├── Dataset_A_lmfn_tfn_fusion_models.py          # Multimodal fusion baselines: LMFN, TFN
    ├── Dataset_B_lmfn_tfn_fusion_models.py
    │
    ├── Dataset_A_kinetics_fm_models.py              # Heavy IMU baselines: Kinetics-FM-DLR-Net, DL-Kinetics-FM-Net
    ├── Dataset_B_kinetics_fm_models.py
    │
    ├── Dataset_A_Sensor_Distillation.py             # ⭐ MAIN: Teacher + Student + SD (this paper)
    ├── Dataset_B_sensor_distillation.py
    │
    ├── Dataset_A_model_ablation.py                  # Internal ablations: Early Fusion, Feature Concat, MFFM-only
    └── Dataset_B_model_ablation.py
```

| File group | Purpose |
|---|---|
| `*_ffn_hf_ffn_bilstm_convnet.py` | Generic deep-learning baselines |
| `*_lmfn_tfn_fusion_models.py` | Multimodal fusion baselines |
| `*_kinetics_fm_models.py` | Prior IMU-specific baselines (large) |
| `*_Sensor_Distillation.py` | **Main contribution** — trains teacher, then 4 students at increasing sensor counts, with and without SD |
| `*_model_ablation.py` | Internal ablations to justify each architectural component |

### Inside `Dataset_A_Sensor_Distillation.py`

The script trains and evaluates the full pipeline for **one LOSO test subject**. Key classes (defined inline):

| Class | Role |
|---|---|
| `Encoder_1` | 2-layer Bi-LSTM with dropout |
| `Encoder_2` | 2-layer Bi-GRU with dropout |
| `GatingModule` | Sigmoid-gated fusion of two encoder outputs |
| `teacher` | Full Kinetics-MFFM-Net(T) with 3 modalities (acc, gyr, EMG) |
| `student_1` … `student_4` | Naive sparse-IMU students (no SD), 1 → 4 IMUs |
| `student_1_KD` … `student_4_KD` | SD-trained students (return both prediction and `x_KD` features) |

Dataset A student sensor configurations:

| Student | Channels | Sensors |
|---|---|---|
| 1 | acc[0:3], gyr[0:3] | Foot |
| 2 | acc[0:3]+[9:12], gyr[0:3]+[9:12] | Foot + Torso |
| 3 | acc[0:6]+[9:12], gyr[0:6]+[9:12] | Foot + Shank + Torso |
| 4 | acc[0:9]+[9:12], gyr[0:9]+[9:12] | Foot + Shank + Thigh + Torso |

---

## 🚀 How to Run

### Environment

```bash
python>=3.8
torch==1.13.1+cu117
numpy==1.20.3
pandas==1.3.4
scikit-learn==0.24.2
scipy==1.7.1
h5py==3.3.0
```

Hardware used in the paper: training on NVIDIA TITAN XP, inference benchmarked on T4.

### Training one LOSO subject (Dataset A example)

```bash
# Edit the HDF5 path and output directory at the top of the script first:
#   '/home/sanzidpr/Dataset A-Kinetics/All_subjects_data.h5'
#   '/home/sanzidpr/Journal_3/Dataset_A_model_results/Subject<N>/'

python Codes/Dataset_A_Sensor_Distillation.py
```

The script:
1. Loads all 20 subjects, holds out one as test (Subject_30 in the current script).
2. Trains the teacher (full IMU + EMG).
3. Trains 4 naive students (1, 2, 3, 4 IMUs).
4. Trains 4 SD students using the frozen teacher.
5. Evaluates each on the held-out subject.
6. Saves a CSV with NRMSE and PCC for all 9 ablations × 7 targets.

For full LOSO results, loop the script over all 20 subjects (e.g., wrap in a shell loop that swaps the test subject ID).

### Hyperparameters (held constant for fair comparison)

| Hyperparameter | Value |
|---|---|
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | RMSE |
| Max epochs | 40 |
| Early stopping patience | 10 |
| Validation split | 20% |
| Random seed | 42 |
| Window size | 100 (Dataset A), 50 (Dataset B) |
| Distillation weights | α = 0.50, β = 1.00 |

### Evaluation

- **NRMSE (%)** — RMSE normalized by ground-truth range, per target.
- **PCC** — Pearson correlation, per target.
- Reported as mean ± std across LOSO subjects.
- Statistical tests in the paper: repeated-measures ANOVA + Bonferroni post-hoc; paired *t*-test for SD vs. no-SD comparison.

---

## 🔧 Adapting to Your Own Data

The most likely things that need editing for a new dataset:

1. **HDF5 layout & paths** — the loader expects `/All_subjects/Subject_<id>/{Treadmill, Levelground, Ramp, Stair}` arrays. Update the paths and group names at the top of the script.
2. **Channel layout** — the data slicing (`x_train[:,0:24]` for IMUs, `[24:47]` for kinematics, etc.) is hardcoded. Adjust to your column order.
3. **Sensor masks for students** — slicing like `data_acc[:,:,0:3]` defines which IMU channels each student sees. Match these to your sensor placement order.
4. **Number of targets** — change `output_dim`, the final FC out-features (`nn.Linear(..., 7)`), and the `m1`/`m2` slice indices for the labels.
5. **Window size** — `w = 100` is hardcoded; change to match your sampling rate × desired window duration.

---

## ⚠️ Limitations & Extensions

**Acknowledged in the paper:**
- Datasets are walking-only; running, jumping, transitions need testing.
- Both datasets are young adults; clinical populations (elderly, post-stroke, amputees) need separate validation.
- Inference latency is benchmarked, but full embedded deployment with wireless IMUs is future work.

**Worth thinking about:**
- The teacher's 128-dim bottleneck size is fixed; ablating bottleneck dimensionality could compress further.
- Distillation weights (α, β) are empirically chosen — learnable or scheduled weights may help.
- The student receives the same window length as the teacher; for prosthetic control, **forecasting** (predicting 100–200 ms ahead) is the natural extension.

---

## 📎 Citation

```bibtex
@article{hossain2026kineticsmffm,
  title   = {Lightweight and Efficient Kinetics Estimation with Sparse IMUs via Multi-Modal Sensor Distillation},
  author  = {Hossain, Md Sanzid Bin and Hadley, Dexter and Guo, Zhishan and Choi, Hwan},
  journal = {IEEE Sensors Journal},
  year    = {2026}
}
```

Funded by NSF Grants FRR-2246671 and FRR-2246672.
