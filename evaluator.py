import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(classifier, X_test, y_test, class_names):
    """
    Evaluates the model on the test set, displays the classification report,
    plots the confusion matrix, and saves it in a 'results/' directory.
    """
    print("\nPredictions on the test set...")
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Global Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    print("DETAILED CLASSIFICATION REPORT:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 2. Figure structure and visual markers
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    
    plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.2f}%)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    # Configure axis ticks before writing text
    plt.xticks(range(len(class_names)), class_names, rotation=20)
    plt.yticks(range(len(class_names)), class_names)
    
    # 3. Numeric annotations in each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight='bold'
            )
            
    plt.tight_layout()
    
    # --- NEW: AUTOMATED IMAGE SAVING ---
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Confusion matrix successfully saved to: '{save_path}'")
    # -----------------------------------
    
    plt.show()
    
    return y_pred, accuracy, cm


def analyze_class_performance(cm, class_names):
    """
    Calculates and displays Precision, Recall (Sensitivity), and F1-score 
    for each tumor class.
    """
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE BY CLASS:")
    print("="*60)
    
    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Tumor [{class_name.upper()}] :")
        print(f"  Precision: {precision:.4f} | Recall (Sensitivity): {recall:.4f} | F1-score: {f1:.4f}")
        print(f"  Correct diagnoses: {TP} | Errors: {FP + FN}\n")