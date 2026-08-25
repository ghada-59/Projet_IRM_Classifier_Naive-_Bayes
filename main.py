import os
import warnings
from data_loader import load_mri_split
from feature_extractor import extract_features
from models import train_classifier
from evaluator import evaluate_model, analyze_class_performance
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def main():
    BASE_PATH = "brain_mri_dataset"
    CLASSIFIER_TYPE = "svm"  # Default GaussianNB is "bayes"
    MAX_PER_CLASS = 300      # Number of images per class (set to None to load all)
    
    TRAIN_PATH = os.path.join(BASE_PATH, "Training")
    TEST_PATH = os.path.join(BASE_PATH, "Testing")
    
    print("="*60)
    print(f"PROJECT: AUTOMATED MRI TUMOR DETECTION ({CLASSIFIER_TYPE.upper()})")
    print("="*60)
    
    # 1. Loading MRI images (Training & Testing)
    X_train_raw, y_train, class_names = load_mri_split(TRAIN_PATH, max_images_per_class=MAX_PER_CLASS)
    if len(X_train_raw) == 0:
        print("[!] Error: Make sure the 'brain_mri_dataset/Training' folder exists.")
        return
        
    X_test_raw, y_test, _ = load_mri_split(TEST_PATH, class_names=class_names, max_images_per_class=MAX_PER_CLASS)
    
    # 2. Extracting HOG features + Intensities
    X_train = extract_features(X_train_raw)
    X_test = extract_features(X_test_raw)
    
    print(f"\nDimensions: {X_train.shape[1]} features extracted per MRI")
    print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    
    # 3. Training the classifier
    classifier = train_classifier(X_train, y_train, CLASSIFIER_TYPE)
    
    # 4. Evaluation on test data
    y_pred, accuracy, cm = evaluate_model(classifier, X_test, y_test, class_names)
    
    # 5. Detailed performance analysis
    analyze_class_performance(cm, class_names)
    
    # Displaying data dimensions for verification (Sanity Check)
    print(f"Dimensions: {X_train.shape[1]} features extracted per MRI")
    print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # --- DATA SCALING (Standardization) ---
    print("\nStandardizing features (Scaling)...")
    scaler = StandardScaler()

    # .fit_transform() computes the mean and standard deviation on the training set, then applies them
    X_train = scaler.fit_transform(X_train)

    # .transform() normalizes the test set using ONLY the training parameters 
    # (This is essential to avoid Data Leakage)
    X_test = scaler.transform(X_test) 
    # -----------------------------------------------------

    # Training the classifier (Bayes or SVM) on normalized data
    classifier = train_classifier(X_train, y_train, classifier_type=CLASSIFIER_TYPE)

if __name__ == "__main__":
    main()