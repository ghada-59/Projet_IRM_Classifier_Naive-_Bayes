# 🧠 Brain MRI Classification with Classical Machine Learning

An academic machine-learning project for classifying brain MRI images into four dataset categories: **glioma, meningioma, no tumor and pituitary**.

## 🎯 Project objective

The project compares classical machine-learning approaches using image-derived features rather than an end-to-end deep-learning model.

## 🔬 Feature engineering

Two types of image information are combined:

- **Raw pixel intensities**
- **HOG (Histogram of Oriented Gradients)** descriptors

The resulting feature vector contains **5,860 features per image** in the current implementation.

## 🤖 Models

The project evaluates:

- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- GridSearchCV for SVM hyperparameter exploration

Standardization is applied to the feature data as part of the classification workflow.

## 📊 Evaluation

The project reports accuracy, precision, recall, F1-score and a confusion matrix on the held-out test set.

The current reported test accuracy is **80.08% on 1,200 test images**. This result is specific to the dataset and experimental setup used in the repository and should not be interpreted as clinical performance.

## 📂 Structure

\`\`\`text
Projet_IRM_Classifier_Naive-_Bayes/
├── data_loader.py
├── feature_extractor.py
├── models.py
├── evaluator.py
├── main.py
├── results/
├── requirements.txt
└── README.md
\`\`\`

## 🛠️ Technologies

Python · NumPy · OpenCV/scikit-image · scikit-learn · HOG · SVM · Naive Bayes

## ⚠️ Limitations

This is an academic machine-learning project using a specific public dataset. The reported results should not be generalized to clinical diagnosis or real-world patient populations without independent validation.

## 🚀 Run

\`\`\`bash
pip install -r requirements.txt
python main.py
\`\`\`
