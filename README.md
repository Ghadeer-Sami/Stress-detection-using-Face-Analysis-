# Stress Detection Using Facial Expression Analysis

This project focuses on building a machine learning module that detects **stress vs. no-stress** from facial expressions.  
The system was developed using both **video-based** and **image-based** approaches, with the final model relying on the **FER-2013** dataset and a **Convolutional Neural Network (CNN)**.

---

## Project Overview

Psychologists often struggle to detect subtle signs of stress and anxiety, as these emotions are harder to observe compared to basic emotions like anger or happiness.  
This project aims to support mental health assessment by automatically identifying stress-related facial cues.

Two approaches were explored:

1. **Video-based model** using RAVDESS dataset (MobileNetV2)  
2. **Image-based model** using FER-2013 dataset (CNN)

The image-based approach produced more stable and accurate results.

---

## Dataset Summary

### RAVDESS (Video-Based)
- High-quality emotional speech videos
- 8 emotion classes
- Used initially to capture temporal stress cues
- Results were limited due to:
  - Noisy labels  
  - Frame inconsistencies  
  - High computational cost  

### FER-2013 (Image-Based)
- 35,887 grayscale images (48×48)
- 7 emotion classes
- Collected “in the wild”
- Challenges:
  - Noisy labels  
  - Class imbalance  
  - Variations in lighting, pose, occlusion  

### Stress Label Mapping
To convert FER-2013 into a binary stress dataset:

| Stress Class | No-Stress Class |
|--------------|-----------------|
| Angry        | Happy           |
| Disgust      | Neutral         |
| Fear         |                 |

Sad and Surprise were excluded due to ambiguity.

---

## Model Architecture

A custom **CNN** was trained on the reorganized FER-2013 dataset.

### Key Features
- Grayscale input (48×48)
- Data augmentation:
  - Rotation (15°)
  - Width/height shift (0.1)
  - Horizontal flip
- Normalization (pixel scaling)
- Train/validation split using `ImageDataGenerator`

---

## Results

### CNN Performance (Image-Based)

| Metric      | Score |
|-------------|--------|
| Accuracy    | ~79%   |
| F1-Score    | ~0.73  |
| AUC         | ~0.88  |

The model demonstrated strong potential for detecting stress from facial expressions.

---

## Video-Based Model (MobileNetV2)

The MobileNetV2 model trained on RAVDESS produced lower performance due to:
- Noisy frame-level labels  
- Temporal inconsistency  
- High training cost  

This led to shifting the project toward the image-based CNN approach.

---

## Literature Review Summary

Key findings from reviewed studies:
- CNNs (VGG, ResNet, MobileNet) dominate emotion recognition tasks.
- Stress detection often relies on proxy emotions (anger, fear, disgust).
- Challenges include:
  - Cultural bias  
  - Noisy datasets  
  - Subtle micro-expressions  
  - Environmental variations  
- Hybrid and transfer learning models show strong performance.

A full comparison table is included in the report.

---

## Gap Analysis

- Most datasets focus on basic emotions, not clinical stress.
- Many images are acted, not real-world stress expressions.
- Static images miss temporal cues important for anxiety detection.
- Lack of diverse participants reduces generalization.
- Real clinical stress requires multimodal signals (audio, physiology).

---

## Future Work

- Collect real-world stress datasets with clinical labeling  
- Use multimodal inputs (audio, heart rate, thermal imaging)  
- Explore attention-based and transformer models  
- Improve robustness to lighting, pose, and occlusion  

