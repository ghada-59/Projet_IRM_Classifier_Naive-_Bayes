"""Independent test evaluation for the structural MRI classifier.
Educational use only; not for medical diagnosis.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import experiments.structural_morphometrics.morphometric_pipeline as mp


def load_features(data_dir, description):
    manifest = mp.discover_images(data_dir)
    rows, vectors = [], []

    for row in tqdm(manifest.itertuples(index=False),
                    total=len(manifest), desc=description):
        image = cv2.imread(row.image_path, cv2.IMREAD_UNCHANGED)
        try:
            vector = mp.extract_features(image)
        except (ValueError, cv2.error) as exc:
            warnings.warn(f"Skipping unreadable image {row.image_path}: {exc}")
            continue

        rows.append({
            "image_path": row.image_path,
            "relative_path": row.relative_path,
            "label": row.label,
        })
        vectors.append(vector)

    if not rows:
        raise ValueError(f"No readable images found in {data_dir}")

    table = pd.DataFrame(rows)
    if table["label"].nunique() < 2:
        raise ValueError(f"At least two readable classes are required in {data_dir}")

    return table, np.vstack(vectors), table["label"].to_numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Train on Training and evaluate once on independent Testing."
    )
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_table, X_train, y_train = load_features(
        args.train_dir, "Extracting training features"
    )
    test_table, X_test, y_test = load_features(
        args.test_dir, "Extracting independent test features"
    )

    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    if train_classes != test_classes:
        raise ValueError(
            f"Training/testing classes differ: {train_classes} vs {test_classes}"
        )

    labels = sorted(train_classes)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )),
    ])

    print("Training the model using Training only...")
    model.fit(X_train, y_train)

    print("Evaluating on Testing without retraining...")
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "n_training_images": int(len(y_train)),
        "n_independent_test_images": int(len(y_test)),
        "training_class_counts": {
            str(k): int(v) for k, v in train_table["label"].value_counts().items()
        },
        "test_class_counts": {
            str(k): int(v) for k, v in test_table["label"].value_counts().items()
        },
        "feature_count": len(mp.FEATURE_NAMES),
        "classes": labels,
        "evaluation": "Model fitted on Training; evaluated on separate Testing folder.",
        "random_state": args.random_state,
        "clinical_use": "Educational only; not for diagnosis. Otsu mask is not tumor segmentation.",
    }

    (output / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv")

    predictions = test_table.copy()
    predictions["predicted_label"] = y_pred
    predictions["max_model_score"] = probabilities.max(axis=1)
    predictions.to_csv(output / "test_predictions.csv", index=False)

    classifier = model.named_steps["classifier"]
    importance = pd.DataFrame({
        "feature": mp.FEATURE_NAMES,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)

    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=labels, display_labels=labels,
        xticks_rotation=25, values_format="d", colorbar=False, ax=ax
    )
    ax.set_title("Independent test confusion matrix")
    fig.tight_layout()
    fig.savefig(figures / "independent_test_confusion_matrix.png", dpi=160)
    plt.close(fig)

    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Top 15 Random Forest feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(figures / "feature_importance_top15.png", dpi=160)
    plt.close(fig)

    joblib.dump({
        "model": model,
        "feature_names": mp.FEATURE_NAMES,
        "image_size": mp.IMAGE_SIZE,
        "gaussian_sigma": mp.GAUSSIAN_SIGMA,
        "metrics": metrics,
    }, output / "model_bundle.joblib")

    print(json.dumps(metrics, indent=2))
    print(f"Reports and figures saved to: {output.resolve()}")


if __name__ == "__main__":
    main()