# Project1-XAI-Ensemble-Alzheimers-Diagnosis
Clinical data Integration-Stacking based ensembles-Grad CAM visualizations-SHAP features

# XAI-Ensemble-Alzheimers-Diagnosis

## Overview

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder and one of the leading causes of dementia worldwide. Early diagnosis and accurate prediction of disease progression are critical for timely intervention and effective patient management. This repository presents an explainable deep learning and ensemble-based framework for Alzheimer’s disease diagnosis and progression prediction using neuroimaging and clinical data.

The proposed system is designed as a two-stage hybrid framework that combines classical machine learning models and deep learning techniques with explainable artificial intelligence (XAI) to achieve both high predictive performance and clinical interpretability.

---

## Key Contributions

* Hybrid ensemble framework combining Logistic Regression XGBoost and Artificial Neural Networks
* Multiclass classification of Cognitive Normal (CN) Mild Cognitive Impairment (MCI) and Alzheimer’s Disease (AD)
* Longitudinal prediction of MCI to AD progression using a dedicated ANN model
* Integration of SHAP explainability for transparent and clinically interpretable predictions
* Evaluation on large scale public datasets ADNI and ADNIMERGE
* Computationally efficient and reproducible pipeline

---

## Framework Description

### Stage 1: Diagnostic Classification

A stacked ensemble model is used to classify subjects into CN MCI and AD categories. The ensemble consists of:

* Logistic Regression for interpretable linear relationships
* XGBoost for modeling complex nonlinear feature interactions
* Artificial Neural Network for high level feature abstraction

The predicted probabilities from these models are combined using a logistic regression meta learner to generate the final diagnosis.

### Stage 2: Disease Progression Prediction

A separate longitudinal ANN model is trained on baseline MCI subjects from the ADNIMERGE dataset to predict progression from MCI to AD. This model captures temporal and clinical patterns associated with disease conversion.

### Explainability Layer

SHAP is applied across both stages to quantify feature level contributions for individual and global predictions. This enables clinicians to understand how cognitive scores and neuroimaging biomarkers influence model decisions.

---

## Datasets

### ADNI Dataset

Used for CN MCI AD classification

* Samples: 59,447 training and 14,812 testing
* Features: 18
* Data types: Demographics cognitive scores MRI volumetric measures

### ADNIMERGE Dataset

Used for MCI progression prediction

* Subjects: 1,101 baseline MCI
* Classes: Stable and Converter
* Longitudinal clinical and imaging features

Note: Raw datasets are not included in this repository due to data usage restrictions.

---

## Features Used

* Demographic: Age Gender Education APOE4
* Cognitive Scores: MMSE ADAS13 CDRSB FAQ
* Neuroimaging: Hippocampus Entorhinal Ventricles WholeBrain ICV
* Normalized MRI ratios

---

## Preprocessing

* Label encoding of categorical variables
* Median imputation for missing values
* Z score normalization for numeric features
* Class balancing using adaptive class weights
* Longitudinal labeling for MCI converters

---

## Models and Configuration

### Logistic Regression

* Multinomial classification
* SAGA solver
* L2 regularization

### XGBoost

* 200 estimators
* Max depth 5
* Learning rate 0.05
* Objective multi softprob

### ANN (Classification)

* Two hidden layers 128 and 64 units
* ReLU activation
* Batch normalization and dropout

### ANN (Progression)

* Two hidden layers 64 and 32 units
* Sigmoid output
* Binary cross entropy loss

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 score
* ROC AUC
* Confusion matrices

---

## Results Summary

* Logistic Regression Accuracy: 87.3%
* XGBoost Accuracy: 86.9%
* Stacked Ensemble Accuracy: 86.0%
* ANN Progression Accuracy: 72.2%
* ANN Progression AUC: 0.825

SHAP analysis identified hippocampal volume ADAS13 CDRSB and MMSE as the most influential predictors aligning with established clinical knowledge.

---

## Repository Structure

```
XAI-Ensemble-Alzheimers-Diagnosis/
├── README.md
├── requirements.txt
├── data/
├── preprocessing/
├── models/
├── ensemble/
├── explainability/
├── experiments/
├── results/
├── notebooks/
├── docs/
├── scripts/
└── LICENSE
```

---

## Computational Environment

* Python 3.10
* TensorFlow 2.15
* XGBoost 2.0.3
* Scikit learn 1.4
* SHAP 0.45.1
* Kaggle NVIDIA T4 GPU

---

## How to Run

1. Install dependencies using requirements.txt
2. Prepare ADNI and ADNIMERGE datasets
3. Run preprocessing scripts
4. Train base models and ensemble
5. Train progression ANN
6. Generate SHAP explanations

---

## Applications

* Early Alzheimer’s disease diagnosis
* MCI progression risk assessment
* Clinical decision support systems
* Explainable medical AI research

---

## Future Work

* Integration of PET fMRI and genetic biomarkers
* Advanced temporal models such as RNNs or Transformers
* Survival analysis for time to conversion prediction
* External cohort validation
* Federated learning for privacy preservation
* Interactive clinical explainability dashboards

---

## License

This project is released under an open source license for academic and research use.

---

## Acknowledgements

Data used in this work were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI).

---

## Citation

If you use this work in your research please cite the associated paper.
