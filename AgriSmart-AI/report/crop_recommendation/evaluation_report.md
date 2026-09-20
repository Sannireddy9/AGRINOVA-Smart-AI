# 🌾 AgriSmart AI — Crop Recommendation Evaluation Report

> [!IMPORTANT]
> **Dataset & Evaluation Separation Notice**:
> This evaluation applies exclusively to the **Crop Recommendation Tabular Module** on its dedicated public benchmark test split.
> It is completely independent of the SIH Disease Detection model and the SIH held-out foliar disease test set.
> These metrics must not be conflated with disease-detection classification metrics.

## 1. Executive Summary

| Metric | Calculated Test Result |
|---|---|
| **Test Accuracy** | **99.39%** |
| **Macro-F1 Score** | **0.9939** |
| **Weighted-F1 Score** | **0.9939** |
| **Held-out Test Samples** | 330 |
| **Evaluated Crop Classes** | 22 |

## 2. Dataset Information

- **Dataset**: Crop Recommendation Dataset
- **Source Citation**: Atharva Inamdar / gabbygab1233 (Public Domain / Open Data for precision agriculture)
- **Public URL**: [https://raw.githubusercontent.com/gabbygab1233/Crop-Recommender/main/Crop_recommendation.csv](https://raw.githubusercontent.com/gabbygab1233/Crop-Recommender/main/Crop_recommendation.csv)
- **License**: Open Data / CC0 Public Domain
- **Total Dataset Rows**: 2200 rows (22 crops, 100 samples per crop, balanced)
- **Features Modeled (7)**: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`
- **Train / Val / Test Split**: 70% Train (1,540 rows) / 15% Validation (330 rows) / 15% Test (330 rows)
- **Random State**: 42 (stratified sampling, zero leakage)

## 3. Model Configuration

- **Model Family**: `sklearn.ensemble.RandomForestClassifier`
- **Number of Trees**: 100 estimators
- **Inference Output**: `predict_proba()` multi-class probability distribution across all 22 crops

## 4. Per-Class Performance Breakdown

| Crop Class | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| Apple | 1.000 | 1.000 | 1.000 | 15 |
| Banana | 1.000 | 1.000 | 1.000 | 15 |
| Blackgram | 1.000 | 0.933 | 0.966 | 15 |
| Chickpea | 1.000 | 1.000 | 1.000 | 15 |
| Coconut | 1.000 | 1.000 | 1.000 | 15 |
| Coffee | 1.000 | 1.000 | 1.000 | 15 |
| Cotton | 1.000 | 1.000 | 1.000 | 15 |
| Grapes | 1.000 | 1.000 | 1.000 | 15 |
| Jute | 0.938 | 1.000 | 0.968 | 15 |
| Kidneybeans | 1.000 | 1.000 | 1.000 | 15 |
| Lentil | 1.000 | 1.000 | 1.000 | 15 |
| Maize | 0.938 | 1.000 | 0.968 | 15 |
| Mango | 1.000 | 1.000 | 1.000 | 15 |
| Mothbeans | 1.000 | 1.000 | 1.000 | 15 |
| Mungbean | 1.000 | 1.000 | 1.000 | 15 |
| Muskmelon | 1.000 | 1.000 | 1.000 | 15 |
| Orange | 1.000 | 1.000 | 1.000 | 15 |
| Papaya | 1.000 | 1.000 | 1.000 | 15 |
| Pigeonpeas | 1.000 | 1.000 | 1.000 | 15 |
| Pomegranate | 1.000 | 1.000 | 1.000 | 15 |
| Rice | 1.000 | 0.933 | 0.966 | 15 |
| Watermelon | 1.000 | 1.000 | 1.000 | 15 |

## 5. Confusion Matrix Summary

A total of 330 predictions were evaluated across 22 classes on the test set.
- Correctly classified instances (diagonal sum): **328 / 330**
- Full 22x22 matrix is serialized to [`report/crop_recommendation/confusion_matrix.json`](report/crop_recommendation/confusion_matrix.json).

## 6. Honest Limitations & Operational Scope

1. **Contextual Variables Not in Training Set**:
   The SIH problem statement requests consideration of Soil Type, Water Availability, Season, Location, and Previous Crop.
   These fields are collected by the AgriSmart AI application interface for advisory context, but are **NOT** present in the benchmark tabular dataset.
   In accordance with competition integrity, no synthetic or fake values were created to force these fields into ML training.
2. **Suitability Score Interpretation**:
   Output percentages represent model class membership probabilities based on historical agro-climatic clusters. They indicate agro-climatic alignment, **NOT guaranteed agricultural yield**.
3. **Microclimate Variability**:
   Local weather anomalies, pest pressure, and irrigation variations can affect actual crop performance beyond the historical tabular envelope.
