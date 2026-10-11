# Brain MRI Classification with Handcrafted Features

This experiment uses 37 image-derived intensity, gradient, texture and 2D shape-proxy descriptors with a Random Forest classifier. It is separate from the repository's pixel/HOG baseline.

The workflow is intended for educational image classification. It does not perform validated tumor detection or segmentation.

## Method

1. Read images from class-specific directories.
2. Resize images to 224 × 224 pixels, normalize intensity values and apply Gaussian smoothing.
3. Calculate an Otsu-thresholded region and extract 37 handcrafted descriptors.
4. Fit a `StandardScaler` and a `RandomForestClassifier` with 400 trees and balanced class weights.
5. Save evaluation metrics, per-class scores, feature importances and diagnostic figures.

In the validation workflow, the images are split at image level using a stratified 80/20 split and random seed 42. The separate evaluation script instead fits the model on the supplied `Training` directory and evaluates it on `Testing`.

The Otsu-thresholded region is a bright-region proxy, not a radiologist-annotated tumor mask. Values such as area, thickness, circularity and complexity are therefore image-region descriptors, not verified tumor measurements or clinical biomarkers.

## Reference results reported in the supplied source

The supplied source notebook/PDF reports the following values for a stratified 80/20 split of 5,000 images, including 1,000 validation images.

| Metric | Source-reported value |
|---|---:|
| Accuracy | 87.80% |
| Macro precision | 88.18% |
| Macro recall | 87.57% |
| Macro F1-score | 87.70% |

The source reports per-class F1-scores of 81.60% for glioma, 82.96% for meningioma, 94.64% for no-tumor and 91.60% for pituitary.

These values are transcribed from the supplied source; they are not outputs from the scripts in this repository. The following charts reproduce the reference values and should be read as source summaries, not as newly generated evaluation results.

![Reference validation metrics](assets/source_validation_metrics.svg)

![Reference class distribution](assets/source_class_distribution.svg)

![Reference per-class metrics](assets/source_per_class_metrics.svg)

## Independent test evaluation

The separate script `evaluate_independent_test.py` fits the pipeline using the `Training` directory and evaluates it on the separate `Testing` directory without fitting on the test data.

**Dataset used for this run**
- Training images: 5,600 (1,400 per class)
- Test images: 1,600 (400 per class)
- Classes: `glioma`, `meningioma`, `notumor` and `pituitary`
- Features: 37
- Classifier: `StandardScaler` + `RandomForestClassifier` (400 trees)

These counts and labels differ from the BRISC2025 distribution shown in the source reference above. The local `Training` and `Testing` dataset used for this run has not yet been identified by a dataset citation in this repository; the results below should not be attributed to BRISC2025.

### Overall results

| Metric | Test result |
|---|---:|
| Accuracy | 86.25% |
| Macro precision | 86.67% |
| Macro recall | 86.25% |
| Macro F1-score | 85.74% |

### Per-class results

| Class | Precision | Recall | F1-score | Test images |
|---|---:|---:|---:|---:|
| glioma | 89.12% | 63.50% | 74.16% | 400 |
| meningioma | 78.02% | 88.75% | 83.04% | 400 |
| notumor | 87.91% | 100.00% | 93.57% | 400 |
| pituitary | 91.60% | 92.75% | 92.17% | 400 |

The model correctly classified 254 of 400 glioma images, giving a glioma recall of 63.5%. The overall accuracy should be interpreted alongside this class-specific result.

![Independent test confusion matrix](reports_independent_test/figures/independent_test_confusion_matrix.png)

![Top 15 feature importances](reports_independent_test/figures/feature_importance_top15.png)

Feature importance indicates how the fitted Random Forest uses the descriptors for prediction. It does not establish biological causality or clinical significance.

### Reproduce the independent evaluation

Run this command from the repository root. The dataset must be available in the indicated directories.

```powershell
python experiments/structural_morphometrics/evaluate_independent_test.py --train-dir "brain_mri_dataset/Training" --test-dir "brain_mri_dataset/Testing" --output-dir "experiments/structural_morphometrics/reports_independent_test" --random-state 42
```

The script writes a JSON metrics summary, a CSV classification report, feature importances, a confusion matrix and a feature-importance figure. Model bundles and per-image prediction files are generated locally but are excluded from version control.

## Run the validation workflow

The dataset directory should directly contain one folder per class. For BRISC2025, the expected folder for this script is `classification_task/train/`, not the dataset root or the `classification_task` parent folder.

Install the dependencies and run the script from this directory (`experiments/structural_morphometrics`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python morphometric_pipeline.py --data-dir "D:\datasets\brisc2025\brisc2025\classification_task\train" --output-dir reports
```

Replace the example dataset path with the path on your computer. To check the pipeline quickly before a full run, limit the number of images per class:

```powershell
python morphometric_pipeline.py --data-dir "D:\datasets\brisc2025\brisc2025\classification_task\train" --output-dir reports-smoke --max-images-per-class 10
```

Metrics from this small run are intended only as a software check and should not be presented as full-dataset performance. For the full run, omit `--max-images-per-class`.

## Output files

The validation workflow writes the following files under its output directory:

```text
reports/
├── classification_report.csv
├── feature_importance.csv
├── feature_table.csv
├── metrics_summary.json
├── model_bundle.joblib
├── validation_predictions.csv
└── figures/
    ├── class_distribution.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── per_class_metrics.png
    ├── sample_images.png
    └── validation_examples.png
```

The validation-example figure overlays the Otsu region and gradient map for visual inspection; it is not a tumor segmentation result.

## Tests

Run the unit tests from this directory:

```powershell
python -m pytest -q
```

## Limitations

- The source reference uses a validation split from the training data; it is not an independent test-set result from the supplied scripts.
- The separate `Training`/`Testing` run uses a dataset whose citation has not yet been recorded in this repository.
- Image-level splitting does not establish patient-level generalization.
- The Otsu region can select non-tumor structures.
- Model scores are not calibrated clinical probabilities.
- Performance on a public dataset does not establish clinical validity or utility.

## Reference

Fateh, A. et al. (2025). *BRISC: Annotated dataset for brain tumor segmentation and classification with Swin-HAFNet*. [arXiv:2506.14318](https://arxiv.org/abs/2506.14318).
