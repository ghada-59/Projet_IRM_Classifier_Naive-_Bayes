"""Classical brain-MRI classification using handcrafted image-region descriptors.

Educational project only. The Otsu region below is not a tumor segmentation mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

IMAGE_SIZE = 224
GAUSSIAN_SIGMA = 2.0
RANDOM_STATE = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

FEATURE_NAMES = [
    "threshold_mask_area_ratio", "mean_intensity", "std_intensity",
    "median_intensity", "intensity_skewness", "intensity_kurtosis",
    "gradient_magnitude_mean", "gradient_magnitude_std", "edge_density_proxy",
    "structure_anisotropy_mean", "structure_anisotropy_std",
    "thickness_proxy_mean", "thickness_proxy_max", "circularity_proxy",
    "shape_complexity_proxy", "laplacian_variance", "local_texture_variance_mean",
    "local_texture_variance_std", "intensity_entropy", "gradient_orientation_mean",
    "gradient_orientation_std", "intensity_percentile_10", "intensity_percentile_25",
    "intensity_percentile_75", "intensity_percentile_90",
]
FEATURE_NAMES += [
    f"region{i}_{stat}" for i in range(1, 5)
    for stat in ("mean_intensity", "std_intensity", "mask_fraction")
]


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _prepare_gray(image: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image is empty or could not be decoded.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _mask_and_maps(gray: np.ndarray, sigma: float = GAUSSIAN_SIGMA):
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    threshold = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    binary = (threshold > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == largest
    else:
        mask = binary.astype(bool)
    if not mask.any():
        mask = np.ones_like(gray, dtype=bool)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    edge = magnitude / (float(magnitude.max()) + 1e-8)
    return smooth, mask, gx, gy, magnitude, edge


def extract_features(
    image: np.ndarray,
    image_size: int = IMAGE_SIZE,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Return 37 finite descriptors from a grayscale, BGR, or BGRA image."""
    gray = _prepare_gray(image, image_size)
    smooth, mask, gx, gy, magnitude, _ = _mask_and_maps(gray, sigma)
    pixels = smooth[mask].astype(np.float64)
    if pixels.size == 0:
        pixels = smooth.ravel().astype(np.float64)

    angle = np.arctan2(gy, gx)
    grad = magnitude[mask] if mask.any() else magnitude.ravel()
    edge_density = float(np.mean(grad > np.percentile(grad, 75))) if grad.size else 0.0

    tensor_xx = ndimage.gaussian_filter(gx * gx, sigma=2)
    tensor_yy = ndimage.gaussian_filter(gy * gy, sigma=2)
    tensor_xy = ndimage.gaussian_filter(gx * gy, sigma=2)
    trace = tensor_xx + tensor_yy
    determinant = tensor_xx * tensor_yy - tensor_xy * tensor_xy
    discriminant = np.maximum(trace * trace - 4.0 * determinant, 0.0)
    eig1 = (trace + np.sqrt(discriminant)) / 2.0
    eig2 = (trace - np.sqrt(discriminant)) / 2.0
    anisotropy = (eig1 - eig2) / (eig1 + eig2 + 1e-8)

    distance = ndimage.distance_transform_edt(mask)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if contours:
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        area = float(cv2.contourArea(contour))
        circularity = 4.0 * np.pi * area / (perimeter**2 + 1e-8)
        complexity = perimeter**2 / (4.0 * np.pi * area + 1e-8)
    else:
        circularity, complexity = 0.0, 0.0

    laplacian = cv2.Laplacian(smooth, cv2.CV_32F)
    local_mean = cv2.GaussianBlur(smooth.astype(np.float32), (0, 0), 5)
    local_var = cv2.GaussianBlur(
        (smooth.astype(np.float32) - local_mean) ** 2, (0, 0), 5
    )
    hist, _ = np.histogram(pixels, bins=16, range=(0, 256))
    probabilities = hist / (hist.sum() + 1e-12)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))

    features = [
        float(mask.mean()), float(np.mean(pixels) / 255.0),
        float(np.std(pixels) / 255.0), float(np.median(pixels) / 255.0),
        _finite(skew(pixels, bias=True)), _finite(kurtosis(pixels, fisher=True, bias=True)),
        float(np.mean(grad) / 255.0), float(np.std(grad) / 255.0), edge_density,
        float(np.mean(anisotropy[mask])), float(np.std(anisotropy[mask])),
        float(2.0 * np.mean(distance[mask]) / image_size),
        float(2.0 * np.max(distance) / image_size), circularity, complexity,
        float(np.var(laplacian[mask]) / (255.0**2)),
        float(np.mean(local_var[mask]) / (255.0**2)),
        float(np.std(local_var[mask]) / (255.0**2)), entropy,
        float(np.mean(angle[mask])), float(np.std(angle[mask])),
        float(np.percentile(pixels, 10) / 255.0),
        float(np.percentile(pixels, 25) / 255.0),
        float(np.percentile(pixels, 75) / 255.0),
        float(np.percentile(pixels, 90) / 255.0),
    ]
    height, width = smooth.shape
    regions = [(0, .5, 0, .5), (0, .5, .5, 1), (.5, 1, 0, .5), (.5, 1, .5, 1)]
    for y0, y1, x0, x1 in regions:
        ys, ye, xs, xe = int(height*y0), int(height*y1), int(width*x0), int(width*x1)
        region_mask = mask[ys:ye, xs:xe]
        region_img = smooth[ys:ye, xs:xe]
        values = region_img[region_mask]
        if values.size == 0:
            values = region_img.ravel()
        features.extend([
            float(np.mean(values) / 255.0),
            float(np.std(values) / 255.0),
            float(region_mask.mean()),
        ])

    vector = np.nan_to_num(
        np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if vector.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(f"Expected {len(FEATURE_NAMES)} features, got {vector.shape}")
    return vector


def discover_images(data_dir: str | Path) -> pd.DataFrame:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    class_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    folder_names = {p.name.lower() for p in class_dirs}
    if folder_names & {"classification_task", "segmentation_task"}:
        raise ValueError(
            "Point --data-dir to BRISC2025/classification_task/train, not the dataset root."
        )
    if {"train", "test"}.issubset(folder_names):
        raise ValueError("Point --data-dir directly to the train folder containing class folders.")
    records = []
    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "image_path": str(path),
                    "relative_path": path.relative_to(root).as_posix(),
                    "label": class_dir.name,
                })
    if not records:
        raise ValueError(f"No supported images found under {root}.")
    manifest = pd.DataFrame(records).sort_values(
        ["label", "relative_path"], kind="stable"
    ).reset_index(drop=True)
    counts = manifest["label"].value_counts()
    if manifest["label"].nunique() < 2 or (counts < 2).any():
        raise ValueError("At least two classes with at least two images per class are required.")
    return manifest


def _save_figures(manifest, y_true, y_pred, probabilities, labels, model, output_dir):
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    counts = manifest["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values)
    ax.bar_label(bars)
    ax.set(title="Validation split class distribution", xlabel="Class", ylabel="Images")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figures / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, display_labels=labels,
        values_format="d", xticks_rotation=25, ax=ax, colorbar=False,
    )
    ax.set_title("Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    importances = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.named_steps["classifier"].feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(output_dir / "feature_importance.csv", index=False)
    top = importances.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set(title="Top 20 Random Forest feature importances",
           xlabel="Impurity-based importance", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(figures / "feature_importance.png", dpi=160)
    plt.close(fig)

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    class_metrics = pd.DataFrame({
        label: {metric: report[label][metric] for metric in ("precision", "recall", "f1-score")}
        for label in labels
    }).T
    ax = class_metrics.plot(kind="bar", figsize=(9, 5), ylim=(0, 1.12), rot=15)
    ax.set_title("Per-class validation metrics")
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(figures / "per_class_metrics.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(len(labels), 5, figsize=(11, 2.4 * len(labels)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for row, label in enumerate(labels):
        paths = manifest.loc[manifest.label == label, "image_path"].head(5)
        for col, path in enumerate(paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                axes[row, col].imshow(img, cmap="gray")
                axes[row, col].set_title(label if col == 0 else f"Example {col + 1}", fontsize=8)
    fig.suptitle("Example images by validation label")
    fig.tight_layout()
    fig.savefig(figures / "sample_images.png", dpi=160)
    plt.close(fig)

    chosen = np.linspace(0, len(y_true) - 1, min(8, len(y_true)), dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), chosen):
        path = manifest.iloc[int(idx)].image_path
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gray = _prepare_gray(img)
        _, mask, _, _, _, edge = _mask_and_maps(gray)
        ax.imshow(gray, cmap="gray")
        ax.imshow(edge, cmap="magma", alpha=.25, vmin=0, vmax=1)
        ax.contour(mask.astype(float), levels=[.5], colors=["cyan"], linewidths=.8)
        ax.set_title(
            f"True: {y_true[int(idx)]}\nPred: {y_pred[int(idx)]} "
            f"({np.max(probabilities[int(idx)]):.2f})",
            fontsize=8,
        )
    fig.suptitle("Validation examples: Otsu region + edge map (not tumor segmentation)")
    fig.tight_layout()
    fig.savefig(figures / "validation_examples.png", dpi=160)
    plt.close(fig)


def run_experiment(
    data_dir: str | Path,
    output_dir: str | Path = "reports",
    test_size: float = .2,
    random_state: int = RANDOM_STATE,
    max_images_per_class: int | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = discover_images(data_dir)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if max_images_per_class is not None:
        if max_images_per_class < 5:
            raise ValueError("Use at least 5 images per class for a smoke test.")
        manifest = pd.concat([
            group.sample(n=min(len(group), max_images_per_class), random_state=random_state)
            for _, group in manifest.groupby("label", sort=True)
        ]).sort_values(["label", "relative_path"], kind="stable").reset_index(drop=True)

    valid_rows, feature_rows = [], []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Extracting features"):
        image = cv2.imread(row.image_path, cv2.IMREAD_UNCHANGED)
        try:
            vector = extract_features(image)
        except (ValueError, cv2.error) as exc:
            warnings.warn(f"Skipping unreadable image {row.image_path}: {exc}")
            continue
        feature_rows.append(vector)
        valid_rows.append({
            "image_path": row.image_path,
            "relative_path": row.relative_path,
            "label": row.label,
        })

    manifest = pd.DataFrame(valid_rows)
    if manifest.empty or (manifest["label"].value_counts() < 2).any():
        raise ValueError("After skipping unreadable images, each class must retain at least two images.")
    X = np.vstack(feature_rows)
    y = manifest["label"].to_numpy()
    all_indices = np.arange(len(y))
    X_train, X_val, y_train, y_val, _, val_idx = train_test_split(
        X, y, all_indices, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=random_state, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    labels = model.named_steps["classifier"].classes_.tolist()
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "n_images_total": int(len(manifest)),
        "n_images_train": int(len(y_train)),
        "n_images_validation": int(len(y_val)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "feature_count": len(FEATURE_NAMES),
        "classes": labels,
        "evaluation": "stratified image-level validation split from the supplied folder",
        "clinical_use": "Not for diagnosis; the Otsu mask is not tumor segmentation.",
    }
    (output / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report = classification_report(y_val, y_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
    feature_table = pd.DataFrame(X, columns=FEATURE_NAMES)
    feature_table["label"] = y
    feature_table["relative_path"] = manifest["relative_path"]
    feature_table.to_csv(output / "feature_table.csv", index=False)
    pd.DataFrame({
        "image_path": manifest.iloc[val_idx]["image_path"].to_numpy(),
        "true_label": y_val,
        "predicted_label": y_pred,
        "max_model_score": np.max(probabilities, axis=1),
    }).to_csv(output / "validation_predictions.csv", index=False)
    joblib.dump({
        "model": model, "feature_names": FEATURE_NAMES,
        "image_size": IMAGE_SIZE, "gaussian_sigma": GAUSSIAN_SIGMA, "metrics": metrics,
    }, output / "model_bundle.joblib")
    _save_figures(
        manifest.iloc[val_idx].reset_index(drop=True), y_val, y_pred,
        probabilities, labels, model, output,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Reports and figures saved to: {output.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder directly containing class-name subfolders.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--max-images-per-class", type=int, default=None,
        help="Optional cap for smoke testing only; do not report capped-subset scores as final results.",
    )
    args = parser.parse_args()
    run_experiment(
        args.data_dir, args.output_dir, args.test_size,
        args.random_state, args.max_images_per_class,
    )


if __name__ == "__main__":
    main()
