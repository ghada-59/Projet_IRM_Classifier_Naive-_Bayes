from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(classifier, X_test, y_test, class_names, output_dir=None):
    """Evaluate a fitted classifier and save its confusion matrix."""
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Global Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print(classification_report(
        y_test, y_pred, target_names=class_names, digits=4, zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(cm, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title(f"Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")
    axis.set_xticks(range(len(class_names)), class_names, rotation=20)
    axis.set_yticks(range(len(class_names)), class_names)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontweight="bold",
            )

    figure.tight_layout()
    destination = Path(output_dir) if output_dir else Path.cwd() / "results"
    destination.mkdir(parents=True, exist_ok=True)
    save_path = destination / "confusion_matrix.png"
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[+] Confusion matrix saved to: {save_path}")

    return y_pred, accuracy, cm


def analyze_class_performance(cm, class_names):
    """Print one-vs-rest precision, recall, and F1 for each class."""
    if cm.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix shape does not match class_names.")

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        print(
            f"Class [{class_name.upper()}]: "
            f"Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}"
        )
