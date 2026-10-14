"""Local API for evaluating the repository's classical MRI classifiers on an uploaded dataset ZIP."""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from data_loader import load_mri_split
from feature_extractor import extract_features
from models import train_classifier

app = FastAPI(title="MRI Classical ML Workbench", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "MRI Classical ML Workbench"}


@app.post("/api/evaluate")
async def evaluate(
    dataset_zip: UploadFile = File(...),
    classifier_type: str = Form("bayes"),
    max_images_per_class: int = Form(100),
):
    if classifier_type not in {"bayes", "svm"}:
        raise HTTPException(400, "Choose 'bayes' or 'svm'.")
    if not 10 <= max_images_per_class <= 300:
        raise HTTPException(400, "max_images_per_class must be between 10 and 300.")
    if not dataset_zip.filename or not dataset_zip.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Upload a ZIP archive containing Training/ and Testing/ folders.")

    with tempfile.TemporaryDirectory(prefix="mri_ml_") as temp:
        root = Path(temp)
        archive = root / "dataset.zip"
        total = 0
        with archive.open("wb") as handle:
            while chunk := await dataset_zip.read(1024 * 1024):
                total += len(chunk)
                if total > 250 * 1024 * 1024:
                    raise HTTPException(413, "Dataset ZIP exceeds the 250 MB upload limit.")
                handle.write(chunk)

        try:
            with zipfile.ZipFile(archive) as zf:
                entries = zf.infolist()
                if len(entries) > 20000:
                    raise HTTPException(413, "Archive contains too many files.")
                for entry in entries:
                    target = (root / "extracted" / entry.filename).resolve()
                    base = (root / "extracted").resolve()
                    if target != base and base not in target.parents:
                        raise HTTPException(400, "Unsafe path detected in ZIP archive.")
                    if entry.file_size > 100 * 1024 * 1024:
                        raise HTTPException(413, "An individual archive member exceeds 100 MB.")
                zf.extractall(root / "extracted")
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "The uploaded file is not a valid ZIP archive.") from exc

        base = root / "extracted"
        train_candidates = list(base.rglob("Training"))
        train_candidates = [p for p in train_candidates if p.is_dir() and (p.parent / "Testing").is_dir()]
        if not train_candidates:
            raise HTTPException(
                400,
                "Could not find matching Training/ and Testing/ directories. ZIP the dataset root, not just the image files.",
            )
        train_path = train_candidates[0]
        test_path = train_path.parent / "Testing"
        try:
            X_train_raw, y_train, class_names = load_mri_split(
                train_path, max_images_per_class=max_images_per_class
            )
            X_test_raw, y_test, test_class_names = load_mri_split(
                test_path, class_names=class_names, max_images_per_class=max_images_per_class
            )
            if len(X_train_raw) == 0 or len(X_test_raw) == 0:
                raise ValueError("Both Training and Testing must contain supported images.")
            if test_class_names != class_names:
                raise ValueError("Training and Testing class mappings do not match.")
            if len(np.unique(y_train)) < 2:
                raise ValueError("At least two classes with images are required for training.")
            X_train = extract_features(X_train_raw)
            X_test = extract_features(X_test_raw)
            model = train_classifier(X_train, y_train, classifier_type)
            predictions = model.predict(X_test)
            matrix = confusion_matrix(y_test, predictions, labels=range(len(class_names)))
            report = classification_report(
                y_test, predictions, labels=range(len(class_names)),
                target_names=class_names, output_dict=True, zero_division=0
            )
            return {
                "classifier": classifier_type,
                "class_names": class_names,
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "feature_count": int(X_train.shape[1]),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                "confusion_matrix": matrix.tolist(),
                "report": {
                    name: {
                        "precision": float(values["precision"]),
                        "recall": float(values["recall"]),
                        "f1_score": float(values["f1-score"]),
                        "support": int(values["support"]),
                    }
                    for name, values in report.items() if isinstance(values, dict)
                },
                "note": "Evaluation uses the supplied dataset and a capped random subset from each split. Results are educational image-level metrics, not patient-level validation or a diagnostic performance claim.",
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, "Dataset evaluation failed: " + str(exc)) from exc
