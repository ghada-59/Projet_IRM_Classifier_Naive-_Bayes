from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_classifier(X_train, y_train, classifier_type='svm'):
    """
    Trains a classifier (Naive Bayes or SVM optimized via GridSearchCV).
    """
    if classifier_type == 'bayes':
        print("\nTraining Naive Bayes Classifier (GaussianNB)...")
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
    elif classifier_type == 'svm':
        print("\nSearching for the best hyperparameters for SVM (GridSearchCV)...")
        # Define a parameter grid to test to find the optimal combination
        param_grid = {
            'C': [1, 10, 50, 100],
            'gamma': ['scale', 'auto', 0.01, 0.001],
            'kernel': ['rbf']
        }
        
        # GridSearchCV tests all combinations using cross-validation (cv=3)
        base_svc = SVC(probability=True, random_state=42)
        classifier = GridSearchCV(base_svc, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        
        print("Launching optimized training...")
        classifier.fit(X_train, y_train)
        
        # Display the best parameters found
        print(f"[+] Best SVM parameters found: {classifier.best_params_}")
        
    else:
        raise ValueError("Invalid classifier type (choose 'bayes' or 'svm').")
        
    return classifier