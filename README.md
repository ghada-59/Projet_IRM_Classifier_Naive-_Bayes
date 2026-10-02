# 🧠 Brain MRI Classification with Classical Machine Learning

Academic/practice project for classifying brain MRI images into four categories present in the dataset: **glioma, meningioma, no tumor and pituitary**.

## 🎯 Objective

The project explores a classical machine-learning workflow for image classification using image-derived features rather than an end-to-end deep-learning model.

## 🔬 Feature Engineering

Each grayscale MRI is resized to **64 × 64** pixels. Two types of information are combined:

- flattened pixel intensities: 4,096 features
- HOG (Histogram of Oriented Gradients): 1,764 features

This produces **5,860 features per image** with the current configuration.

## 🤖 Models

The repository contains two classifier options:

- Gaussian Naive Bayes
- Support Vector Machine (SVM)

For SVM, `GridSearchCV` explores a small RBF-kernel hyperparameter grid using stratified 3-fold cross-validation.

Feature standardization is part of the scikit-learn `Pipeline`, so the scaler is fitted separately within each cross-validation fold.

## 📊 Evaluation

The final classifier is evaluated on the held-out **Testing** directory using:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

The default configuration uses at most **300 images per class from both Training and Testing**. When a split contains more than 300 images in a class, the subset is selected randomly with a fixed seed (`42`) for reproducibility.

This cap is an experimental convenience to keep the classical feature-based workflow manageable; it is **not a replacement for evaluation on the complete dataset**.

## 📂 Dataset Organization

```text
brain_mri_dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

Class names are read from the training directory and reused for the testing split.

## 🛠️ Technologies

Python · NumPy · scikit-image · scikit-learn · HOG · SVM · Naive Bayes · Matplotlib

## 🚀 Run

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset in `brain_mri_dataset/` using the structure above, then run:

```bash
python main.py
```

The confusion matrix is saved under `results/confusion_matrix.png`.

## ⚠️ Limitations

- This is an **academic/practice machine-learning project**, not a medical diagnostic system.
- The dataset is a specific public dataset; results should not be generalized to clinical populations without independent validation.
- The current workflow uses grayscale 64 × 64 images and HOG features plus raw pixels, which may discard clinically relevant image information.
- The default 300-images-per-class cap means the reported test metrics depend on the selected subset.
- The SVM hyperparameters are selected using accuracy as the cross-validation criterion; this does not establish clinical usefulness.
- Dataset-level performance does not establish patient-level clinical performance.

## 📁 Project Structure

```text
Projet_IRM_Classifier_Naive-_Bayes/
├── brain_mri_dataset/
├── data_loader.py
├── feature_extractor.py
├── models.py
├── evaluator.py
├── main.py
├── results/
├── requirements.txt
└── README.md
```

## 📝 Project Type

**Academic / practice project.** The project demonstrates image preprocessing, feature engineering, classical classification, cross-validation and evaluation on a public dataset.