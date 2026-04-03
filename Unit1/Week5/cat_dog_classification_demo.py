
"""
Simple end-to-end example:
- Make a tiny synthetic dataset of cats/dogs with made-up numeric features
- Split into train/test
- Train two models: Naive Bayes + Logistic Regression
- Predict labels (cat/dog) on the test set
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def make_synthetic_cat_dog_data(seed: int = 42):
    """
    Create a simple dataset with numeric features:
      1) weight_kg
      2) height_cm
      3) ear_length_cm
      4) bark_volume_db
      5) purr_freq_hz
    Label: 'cat' or 'dog'
    """
    rng = np.random.default_rng(seed)

    n_cats = 60
    n_dogs = 60

    cats = np.column_stack([
        rng.normal(4.5, 1.0, n_cats),
        rng.normal(25.0, 3.5, n_cats),
        rng.normal(6.0, 1.2, n_cats),
        rng.normal(5.0, 3.0, n_cats),
        rng.normal(85.0, 10.0, n_cats),
    ])

    dogs = np.column_stack([
        rng.normal(18.0, 6.0, n_dogs),
        rng.normal(45.0, 8.0, n_dogs),
        rng.normal(10.0, 3.0, n_dogs),
        rng.normal(70.0, 12.0, n_dogs),
        rng.normal(5.0, 4.0, n_dogs),
    ])

    X = np.vstack([cats, dogs])
    y = np.array(["cat"] * n_cats + ["dog"] * n_dogs)

    feature_names = [
        "weight_kg",
        "height_cm",
        "ear_length_cm",
        "bark_volume_db",
        "purr_freq_hz",
    ]

    return X, y, feature_names


def main():
    X, y, feature_names = make_synthetic_cat_dog_data(seed=7)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    nb = GaussianNB()
    lr = LogisticRegression(max_iter=2000)

    nb.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    nb_pred = nb.predict(X_test)
    lr_pred = lr.predict(X_test)

    print("Features:", feature_names)
    print("\n--- Sample predictions (first 10 test rows) ---")

    for i in range(min(10, len(X_test))):
        row = ", ".join(
            f"{name}={X_test[i, j]:.2f}"
            for j, name in enumerate(feature_names)
        )
        print(
            f"#{i:02d} | {row} | true={y_test[i]} | "
            f"NB={nb_pred[i]} | LR={lr_pred[i]}"
        )

    print("\n=== Naive Bayes Results ===")
    print("Accuracy:", accuracy_score(y_test, nb_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, nb_pred))
    print(classification_report(y_test, nb_pred))

    print("\n=== Logistic Regression Results ===")
    print("Accuracy:", accuracy_score(y_test, lr_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, lr_pred))
    print(classification_report(y_test, lr_pred))


if __name__ == "__main__":
    main()
