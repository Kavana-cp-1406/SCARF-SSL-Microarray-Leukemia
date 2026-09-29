"""
Classical baseline models (Section 2.5): Random Forest, SVM-RBF, XGBoost.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
    seed: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_svm_rbf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 10.0,
    gamma: str = "scale",
    seed: int = 42,
) -> SVC:
    model = SVC(kernel="rbf", C=C, gamma=gamma, decision_function_shape="ovo", random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
