```python
import numpy as np
from puf_model_pipeline import my_map

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


# Load train/test datasets
Z_trn = np.loadtxt("secret_trn.txt")
Z_tst = np.loadtxt("secret_tst.txt")

X_train = Z_trn[:, :-1]
y_train = Z_trn[:, -1]

X_test = Z_tst[:, :-1]
y_test = Z_tst[:, -1]

# Feature engineering using existing project pipeline
X_train_feat = my_map(X_train)
X_test_feat = my_map(X_test)


def evaluate_model(model, name):
    model.fit(X_train_feat, y_train)
    pred = model.predict(X_test_feat)
    acc = accuracy_score(y_test, pred)
    print(f"{name} Accuracy: {acc:.4f}")


print("========== Logistic Regression Experiments ==========")

# L1 Regularization
logreg_l1 = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=1.0,
    max_iter=1000
)

# L2 Regularization
logreg_l2 = LogisticRegression(
    penalty='l2',
    solver='liblinear',
    C=1.0,
    max_iter=1000
)

evaluate_model(logreg_l1, "Logistic Regression (L1)")
evaluate_model(logreg_l2, "Logistic Regression (L2)")


print("\n========== Hyperparameter Tuning (C values) ==========")

for c in [0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(
        penalty='l2',
        solver='liblinear',
        C=c,
        max_iter=1000
    )

    model.fit(X_train_feat, y_train)
    pred = model.predict(X_test_feat)
    acc = accuracy_score(y_test, pred)

    print(f"C={c}  Accuracy={acc:.4f}")


print("\n========== Linear SVM ==========")

linear_svm = LinearSVC(
    C=1.0,
    max_iter=5000
)

evaluate_model(linear_svm, "Linear SVM")


print("\n========== Polynomial Kernel SVM ==========")

poly_svm = SVC(
    kernel='poly',
    degree=8,
    C=1.0
)

evaluate_model(poly_svm, "Polynomial Kernel SVM (Degree 8)")


print("\n========== Experiment Summary ==========")
print("Completed experiments:")
print("1. Logistic Regression with L1 regularization")
print("2. Logistic Regression with L2 regularization")
print("3. Hyperparameter tuning using multiple C values")
print("4. Linear SVM comparison")
print("5. Polynomial Kernel SVM comparison")
```
