# Lightweight and Efficient Kinetics Estimation with Sparse IMUs via Multi-Modal Sensor Distillation

## Overview
This paper introduces **Kinetics-MFFM-Net**, a lightweight model designed to estimate gait kinetic parameters (such as joint moments and ground reaction forces) using sparse inertial measurement unit (IMU) sensors. The model aims to overcome the limitations of traditional optical motion capture systems and full-body IMU sensor setups, which can be expensive, cumbersome, and impractical for daily use.

## Key Contributions
- **Sparse IMU Setup**: Instead of using a large number of IMU sensors, this study explores the use of a reduced number of IMUs, allowing for more practical and flexible sensor placement.
- **Kinetics-MFFM-Net**: A lightweight deep learning model that employs a multi-modal fusion technique with two encoders and a gating mechanism to estimate gait kinetics.
- **Sensor Distillation**: A novel approach based on knowledge distillation. A **teacher model** (Kinetics-MFFM-Net(T)) is trained using a full set of IMU sensors and other sensor types, while a **student model** (Kinetics-MFFM-Net(S)) is trained using sparse IMUs, significantly improving the performance of sparse sensor setups.
  
## Results
- **Improved Performance**: The sensor distillation technique enhances estimation performance on two datasets with different sparse sensor combinations.
- **Efficiency**: Both the teacher and student models demonstrate computational efficiency and achieve statistically significant improvements compared to state-of-the-art models.
- **Real-Time Deployment**: The models are lightweight and suitable for real-time deployment, making them ideal for practical applications.
- 
## Conclusion
The proposed **Kinetics-MFFM-Net** and its sensor distillation approach provide a practical and efficient solution for gait kinetics estimation using sparse IMU sensors, outperforming current methods in both accuracy and computational efficiency.

