from pathlib import Path

from data_loader import load_mri_split
from feature_extractor import extract_features
from models import train_classifier
from evaluator import evaluate_model, analyze_class_performance


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_PATH = PROJECT_ROOT / "brain_mri_dataset"
CLASSIFIER_TYPE = "svm"
MAX_PER_CLASS = 300


def main():
    train_path = BASE_PATH / "Training"
    test_path = BASE_PATH / "Testing"

    print("=" * 60)
    print(f"PROJECT: BRAIN MRI IMAGE CLASSIFICATION ({CLASSIFIER_TYPE.upper()})")
    print("=" * 60)

    X_train_raw, y_train, class_names = load_mri_split(
        train_path,
        max_images_per_class=MAX_PER_CLASS,
    )
    if len(X_train_raw) == 0:
        raise FileNotFoundError(
            f"Training data not found or empty: {train_path}"
        )

    X_test_raw, y_test, test_class_names = load_mri_split(
        test_path,
        class_names=class_names,
        max_images_per_class=MAX_PER_CLASS,
    )
    if len(X_test_raw) == 0:
        raise FileNotFoundError(
            f"Testing data not found or empty: {test_path}"
        )
    if test_class_names != class_names:
        raise ValueError("Training and testing class mappings are inconsistent.")

    X_train = extract_features(X_train_raw)
    X_test = extract_features(X_test_raw)

    print(f"\nDimensions: {X_train.shape[1]} features extracted per MRI")
    print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    classifier = train_classifier(X_train, y_train, CLASSIFIER_TYPE)

    y_pred, accuracy, cm = evaluate_model(
        classifier,
        X_test,
        y_test,
        class_names,
        output_dir=PROJECT_ROOT / "results",
    )
    analyze_class_performance(cm, class_names)


if __name__ == "__main__":
    main()
