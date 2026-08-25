# 🧠 Automated Brain Tumor Detection and Classification System using MRI

This project is developed as part of an application of **Machine Learning in medical imaging**. The main objective is to design an intelligent, automated pipeline capable of classifying brain MRI scans into four distinct clinical categories, serving as a computer-aided decision support system for medical diagnosis.

---

## 🎯 Project Goal

The early and accurate detection of brain tumors is a critical challenge in neuro-oncology. This project aims to:
1. **Automate medical image analysis** to reduce interpretation workload and human error.
2. **Extract relevant visual biomarkers** (shapes, textures, edges) from raw MRI scans by combining gradient descriptors (**HOG** - Histogram of Oriented Gradients) and raw pixel intensities.
3. **Compare and optimize statistical classifiers** (Naive Bayes and Support Vector Machine - SVM) to achieve the most reliable diagnosis possible.

---

## 🏗️ Architecture and Modular Pipeline

The project is structured following clean and modular software design principles, separating each stage of the data processing workflow:

* **`data_loader.py`**: Rigorously loads training (`Training`) and testing (`Testing`) image datasets, ensuring a perfectly balanced distribution (300 images per class, totaling 1,200 test images).
* **`feature_extractor.py`**: Extracts features by fusing raw pixel intensities and the HOG descriptor, generating a high-dimensional vector space of **5,860 features per MRI scan**.
* **`models.py`**: Defines classification algorithms and integrates **GridSearchCV** for automated hyperparameter tuning.
* **`evaluator.py`**: Computes performance metrics (Precision, Recall, F1-score, Global Accuracy) and **automatically saves** the confusion matrix as a high-resolution image (`results/confusion_matrix.png`).
* **`main.py`**: The global orchestration script that executes the entire pipeline from end to end.


### 📂 Project Directory Structure


```text
Projet_IRM_Bayes/
│
├── brain_mri_dataset/              # Local dataset (ignored in Git via .gitignore)
│   ├── Training/                   # 4 classes: glioma, meningioma, notumor, pituitary
│   └── Testing/                    # 4 classes: glioma, meningioma, notumor, pituitary
│
├── results/                        # Generated automatically during execution
│   └── confusion_matrix.png        # Exported high-resolution confusion matrix
│
├── data_loader.py                  # Dataset loading and splitting module
├── feature_extractor.py            # HOG + Pixel feature extraction module
├── models.py                       # Classifier definition & GridSearchCV optimization
├── evaluator.py                    # Performance metrics & plotting module
├── main.py                         # Global orchestration script
├── requirements.txt                # Python package dependencies
└── README.md                       # Project documentation

```

### 🔄 Pipeline Data Flow Diagram


```text
 [ main.py ] (Orchestrator)
      │
      ├── 1. data_loader.py
      │     └─► Loads raw images & creates labels (Training / Testing splits)
      │
      ├── 2. feature_extractor.py
      │     └─► Extracts 5,860 features per image (Raw Pixels + HOG Descriptor)
      │
      ├── 3. StandardScaler (scikit-learn)
      │     └─► Normalizes data (fitted on Train, transformed on Test to prevent Data Leakage)
      │
      ├── 4. models.py
      │     └─► Trains model (GaussianNB or GridSearchCV-optimized SVM with RBF kernel)
      │
      └── 5. evaluator.py
            ├─► Computes Accuracy, Precision, Recall, F1-Score
            └─► Saves confusion matrix plot to ──► [ results/confusion_matrix.png ]

```

---

## 🔄 Technical Iterations and Model Improvements

To maximize model performance against the high complexity of brain tissue, several key improvements were implemented:

1. **Data Standardization (`StandardScaler`):**
* *Initial Issue:* Raw pixel values (ranging from 0 to 255) mathematically overpowered the smaller normalized values of the HOG descriptor (ranging from 0 to 1) during the SVM's geometric distance calculations.
* *Solution:* Applied center-scaling to the feature matrix. To strictly prevent **Data Leakage**, the scaler is fitted (`fit_transform`) exclusively on the training set and applied (`transform`) independently to the test set.


2. **Hyperparameter Optimization (`GridSearchCV`):**
* *Change:* Instead of relying on default parameters, we implemented a grid search with cross-validation (`cv=3`) to systematically test regularization strengths (`C`) and kernel shape parameters (`gamma`).
* *Optimal Configuration Found:* `{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}`. This configuration forces the model to construct a robust non-linear decision boundary tailored to the dataset.



---

## 📊 Results and Clinical Interpretations

Following the execution of the complete pipeline with the optimized SVM, the system achieves a **Global Accuracy of 80.08%** across 1,200 independent test patients.

### Performance Breakdown by Class:

* **Pituitary Tumor:**
* Precision: *88.04%* | Recall: *95.67%* | F1-Score: *91.69%*
* *Interpretation:* This is the easiest class for the model to identify, as its morphological features and location present distinct characteristics on MRI scans.


* **No Tumor (Healthy):**
* Outstanding Recall: *98.33%*
* *Interpretation:* The model yields very few false negatives, which is crucial in a clinical setting to avoid misdiagnosing a healthy patient as sick.


* **Glioma & Meningioma:**
* *Interpretation:* A minor confusion persists between these two tumor types. This is because their textures, irregular borders, and internal patterns share high visual similarity on standard scans—a challenge that human radiologists also face during initial screening phases.

---

---

## 🚀 Installation and Usage

### Prerequisites

```bash
pip install -r requirements.txt

```

### Running the Project

Place your MRI dataset inside the `brain_mri_dataset/` directory (containing `Training/` and `Testing/` subdirectories), then execute the main script:

```bash
python main.py

```