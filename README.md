# Brain MRI Image Classification with Classical Machine Learning

This repository contains two classical machine-learning experiments for image-level classification of brain MRI scans. They are intended for learning and experimentation, not for medical diagnosis.

## 1. Pixel and HOG baseline

The main workflow resizes each grayscale image to 64 × 64 pixels and combines:

- 4,096 flattened pixel-intensity values
- 1,764 Histogram of Oriented Gradients (HOG) descriptors

The resulting feature vector contains 5,860 values. The available classifiers are Gaussian Naive Bayes and an RBF-kernel Support Vector Machine (SVM). For SVM, hyperparameter selection uses stratified three-fold cross-validation, with standardization inside the scikit-learn pipeline.

The default run samples at most 300 images per class from each of the `Training` and `Testing` folders to keep computation manageable. This subset is reproducible with random seed 42, but it is not a substitute for evaluating the complete dataset.

## 2. Handcrafted descriptors and Random Forest

The separate workflow in [`experiments/structural_morphometrics/`](experiments/structural_morphometrics/README.md) extracts 37 intensity, gradient, texture and 2D shape-proxy descriptors and trains a Random Forest classifier.

Its Otsu-thresholded region is a simple image-processing proxy, not an annotated tumor mask. The derived shape values should not be interpreted as verified tumor measurements or clinical biomarkers.

## Results and dataset provenance

### Reference values reported in the supplied BRISC2025 source

The supplied source notebook/PDF reports 87.80% accuracy and 87.70% macro F1-score on a 1,000-image validation split. It also reports 88.18% macro precision and 87.57% macro recall. These are reference values transcribed from that source; they were not produced by the current scripts in this repository. The accompanying reference charts are labelled accordingly.

### Separate evaluation using Training and Testing folders

A separate run of the handcrafted-feature pipeline was evaluated on 5,600 training images and 1,600 test images, with 37 features and a Random Forest classifier. It achieved 86.25% accuracy and 85.74% macro F1-score. The recall for the `glioma` class was 63.50%.

This evaluation used local `brain_mri_dataset/Training` and `brain_mri_dataset/Testing` folders with class names `glioma`, `meningioma`, `notumor` and `pituitary`. The source of this dataset is not currently documented in the repository. These results should therefore not be attributed to BRISC2025 or compared directly with the source-reported BRISC2025 values. Full metrics and figures are available in the [structural morphometry report](experiments/structural_morphometrics/README.md#independent-test-evaluation).

## Dataset layout for the baseline workflow

Place the dataset under `brain_mri_dataset/`:

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

The dataset itself is not included in this repository. Class names are read from the training directory and reused for the testing split.

## Setup and run

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the dataset in the layout shown above, then run:

```bash
python main.py
```

The baseline confusion matrix is saved to `results/confusion_matrix.png`.

## Technologies

Python, NumPy, scikit-image, scikit-learn, HOG, Gaussian Naive Bayes, SVM, Random Forest, OpenCV and Matplotlib.

## Limitations

- Both workflows classify images; neither performs validated tumor detection or segmentation.
- The default 300-images-per-class cap limits the baseline evaluation to a subset.
- The reported results are dataset-level image classification results, not patient-level clinical performance.
- The separate Training/Testing evaluation has no documented dataset citation yet.
- Results from different datasets and configurations are not a controlled model comparison.
- External validation would be required before drawing conclusions about clinical use.

## Project structure

```text
.
├── data_loader.py
├── feature_extractor.py
├── models.py
├── evaluator.py
├── main.py
├── requirements.txt
├── results/
└── experiments/
    └── structural_morphometrics/
        ├── morphometric_pipeline.py
        ├── evaluate_independent_test.py
        ├── tests/
        └── reports_independent_test/
```
