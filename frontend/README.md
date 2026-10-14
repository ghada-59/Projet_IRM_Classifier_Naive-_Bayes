# NeuroMetric — Classical MRI ML Workbench

Optional React/Vite and FastAPI interface for running the repository's existing pixel + HOG feature pipeline with Gaussian Naive Bayes or RBF SVM.

## Dataset ZIP format

Create a ZIP archive containing both Training/ and Testing/ class folders, for example:

~~~text
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
~~~

Class names can differ, but the Training and Testing class mappings must match. The API limits uploads to 250 MB and samples up to 300 images per class per split.

## Start the local API

From the repository root:

~~~bash
python -m pip install -r requirements-react-api.txt
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
~~~

In another terminal:

~~~bash
cd frontend
npm install
npm run dev
~~~

Open the local URL printed by Vite, normally http://127.0.0.1:5173/.

## Evaluation outputs

The UI displays test accuracy, macro F1, train/test sample counts, confusion matrix and per-class precision/recall/F1/support from the current run. The SVM path uses the existing stratified 3-fold grid search; Gaussian Naive Bayes uses the existing scaled pipeline.

## Limitations and privacy

This project classifies individual images using 4,096 flattened pixel values and 1,764 HOG descriptors (5,860 features total). It does not perform tumor detection or segmentation. Metrics are dependent on the dataset and selected subset; image-level evaluation is not patient-level validation. This is an educational research project, not a diagnostic tool. The API binds to loopback; do not expose it publicly. Uploaded archives are extracted into a temporary directory and deleted when the request finishes.
