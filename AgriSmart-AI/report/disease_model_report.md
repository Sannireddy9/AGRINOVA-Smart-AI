# 🍃 Disease Detection Model Report

> **AgriSmart AI — Core Computer Vision Module**
> *Smart India Hackathon (SIH) 2026*

---

## 1. Task Definition
Foliar crop disease classification from standard single-leaf RGB images. The system ingests a field leaf photograph, performs image preprocessing and normalization, and predicts the pathological condition (or healthy state) with an associated softmax prediction confidence and top-$k$ candidate distribution.

---

## 2. Dataset & Splitting Methodology

### Development & Pre-Hackathon Pipeline
- **Dataset**: Public PlantVillage Benchmark (Hughes & Salathé, 2015; 54,303 images).
- **Split Ratio**: Stratified 80% Train / 20% Validation maintaining class balance across all discovered classes.
- **Preprocessing**: Image validation filtering corrupted/truncated files, RGB channel verification, deterministic inference transforms (`Resize(256)`, `CenterCrop(224)`, ImageNet normalization $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- **Training Augmentations**: `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=15)`, and `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`.

### Official SIH Held-Out Field Test Set Separation
- **Strict Separation Policy**: The hackathon problem statement stipulates that the official test set comprises unseen, field-condition leaf photographs provided by SIH organizers during evaluation.
- **Ethical Boundary**: Under zero circumstances is the held-out test set merged with or used during model training.

---

## 3. Model Architecture & Hyperparameters

| Component / Parameter | Specification | Source / Implementation |
|---|---|---|
| **Base Backbone** | ResNet-50 (`torchvision.models.resnet50`) | ImageNet-1K pre-trained weights (`DEFAULT`) |
| **Classification Head** | Linear head (`in_features=2048`, `out_features=num_classes`) | Replaced dynamically based on discovered dataset classes |
| **Input Dimensions** | $3 \times 224 \times 224$ (RGB) | `model.config.IMAGE_SIZE` |
| **Loss Function** | Categorical Cross-Entropy Loss (`nn.CrossEntropyLoss`) | Standard multi-class classification |
| **Optimizer** | AdamW (`lr=1e-4`, `weight_decay=1e-4`) | `torch.optim.AdamW` |
| **Learning Rate Schedule** | Cosine Annealing (`T_max=NUM_EPOCHS`, `eta_min=1e-6`) | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| **Regularization** | Dropout ($p=0.30$) before final linear projection | `model.model.build_model` |
| **Batch Size** | 32 samples per mini-batch | `model.config.BATCH_SIZE` |
| **Early Stopping** | Patience: 5 validation epochs monitoring validation loss | `model.train.EarlyStopping` |
| **Inference Interface** | Python `predict(image_path)` & CLI `python model/predict.py` | `model/predict.py` |

---

## 4. Evaluation Metrics & Results

### A. Internal Development Evaluation
- The internal training, validation, and evaluation pipeline is fully implemented in [`model/evaluate.py`](../model/evaluate.py).
- Code automatically calculates Macro-Averaged F1, Top-1 Accuracy, multi-class confusion matrix, and per-class Precision/Recall/F1 tables upon checkpoint generation.

### B. Official SIH Held-Out Field Evaluation
- **Evaluation Status**: **`[PENDING]`**
- **Official Result Statement**:
  > **"Final held-out SIH field-test Macro-F1 is pending organizer-provided evaluation data. No fabricated score is reported."**
- **Non-Fabrication Statement**:
  - No synthetic baseline comparison, fake confusion matrix, or manufactured accuracy score has been generated.
  - The application web interface operates in an explicitly badged **Development Mock Mode** whenever an official trained checkpoint is absent, displaying an amber disclosure banner to judges and farmers.

---

## 5. Known Operational Limitations & Field Boundaries

1. **Lab-to-Field Domain Shift**: Models pre-trained on uniform backgrounds (e.g. laboratory leaves) can suffer performance degradation under complex field foliage backgrounds, variable sunlight, or shadowing.
2. **Multi-Pathology Co-occurrence**: Leaves displaying simultaneous bacterial lesions and nutrient chlorosis are classified to a single top-1 class rather than multi-label pathology.
3. **Non-Foliar Diseases**: The computer vision model analyzes leaf surface symptoms; vascular root rots, stem borers, and soil nematodes cannot be detected from foliar imagery alone.
4. **Hardware-Free Decision Support**: Predictions provide agronomic precautionary guidance, not commercial chemical prescriptions.
