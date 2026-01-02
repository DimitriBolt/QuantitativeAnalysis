import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def fit_predict_proba(X_train, y_train, X_test, model_type: str = "logreg") -> np.ndarray:
    if model_type != "logreg":
        raise ValueError(f"Unsupported model_type: {model_type}")

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]
    )

    clf.fit(X_train.values, y_train.values)
    p = clf.predict_proba(X_test.values)[:, 1]
    return p
