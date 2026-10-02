from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_classifier(X_train, y_train, classifier_type="svm"):
    """Train a classifier with scaling performed inside the CV workflow."""
    if classifier_type == "bayes":
        print("\nTraining Gaussian Naive Bayes...")
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ])
        classifier.fit(X_train, y_train)
        return classifier

    if classifier_type == "svm":
        print("\nSearching for SVM hyperparameters with 3-fold cross-validation...")
        param_grid = {
            "classifier__C": [1, 10, 50, 100],
            "classifier__gamma": ["scale", "auto", 0.01, 0.001],
            "classifier__kernel": ["rbf"],
        }
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=42)),
        ])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        classifier = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1
        )
        classifier.fit(X_train, y_train)
        print(f"[+] Best SVM parameters found: {classifier.best_params_}")
        return classifier

    raise ValueError("Invalid classifier type (choose 'bayes' or 'svm').")
