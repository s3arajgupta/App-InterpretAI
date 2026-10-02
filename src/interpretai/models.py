"""Machine Learning Models for Interpretability Benchmarking.

Provides Logistic Regression (Linear/Intrinsic), Random Forest (Tree Ensemble/MDI),
and Multi-Layer Perceptron (Neural Network/Black-Box) with consistent interfaces.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

class LogisticRegressionModel:
    """Logistic Regression Classifier optimized with L2 regularization."""
    def __init__(self, c: float = 0.205, max_iter: int = 300, lr: float = 0.05, random_state: int = 42):
        self.c = c
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.coef_ = rng.normal(0, 0.01, size=n_features).astype(np.float32)
        self.intercept_ = 0.0

        l2_reg = 1.0 / max(1e-4, self.c)

        for _ in range(self.max_iter):
            logits = np.dot(X, self.coef_) + self.intercept_
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
            err = probs - y

            grad_w = (np.dot(X.T, err) + l2_reg * self.coef_) / n_samples
            grad_b = float(np.mean(err))

            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.coef_) + self.intercept_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(np.int32)


class SimpleDecisionStump:
    """Fast axis-aligned binary decision stump for Random Forest ensemble."""
    def __init__(self):
        self.feature_idx: int = 0
        self.threshold: float = 0.0
        self.pred_left: float = 0.0
        self.pred_right: float = 1.0
        self.impurity_gain: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, candidate_features: np.ndarray):
        best_gain = -1.0
        n = len(y)
        p_root = np.mean(y)
        gini_root = 1.0 - (p_root ** 2 + (1.0 - p_root) ** 2)

        for f_idx in candidate_features:
            col = X[:, f_idx]
            # Use percentiles as candidate split thresholds
            thresh_candidates = np.percentile(col, [25, 50, 75])
            for t in thresh_candidates:
                left_mask = col <= t
                right_mask = ~left_mask
                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                if n_l < 5 or n_r < 5:
                    continue

                p_l = np.mean(y[left_mask])
                p_r = np.mean(y[right_mask])
                gini_l = 1.0 - (p_l ** 2 + (1.0 - p_l) ** 2)
                gini_r = 1.0 - (p_r ** 2 + (1.0 - p_r) ** 2)
                gain = gini_root - ((n_l / n) * gini_l + (n_r / n) * gini_r)

                if gain > best_gain:
                    best_gain = gain
                    self.feature_idx = f_idx
                    self.threshold = t
                    self.pred_left = p_l
                    self.pred_right = p_r
                    self.impurity_gain = max(0.0, gain)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        left_mask = X[:, self.feature_idx] <= self.threshold
        preds = np.where(left_mask, self.pred_left, self.pred_right)
        return np.column_stack([1.0 - preds, preds])


class RandomForestModel:
    """Random Forest Ensemble with Gini Impurity feature importance tracking."""
    def __init__(self, n_estimators: int = 50, max_features: Optional[int] = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees: list = []
        self.feature_importances_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        n_samples, n_features = X.shape
        m_feat = self.max_features or int(np.sqrt(n_features))
        rng = np.random.RandomState(self.random_state)

        self.trees = []
        importances = np.zeros(n_features, dtype=np.float32)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = rng.randint(0, n_samples, size=n_samples)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            # Feature subspace
            cand_features = rng.choice(n_features, size=m_feat, replace=False)
            stump = SimpleDecisionStump()
            stump.fit(X_boot, y_boot, cand_features)
            self.trees.append(stump)
            importances[stump.feature_idx] += stump.impurity_gain

        # Normalize feature importances
        total_imp = np.sum(importances)
        if total_imp > 0:
            self.feature_importances_ = importances / total_imp
        else:
            self.feature_importances_ = np.ones(n_features) / n_features

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_probs = np.zeros((len(X), 2), dtype=np.float32)
        for t in self.trees:
            all_probs += t.predict_proba(X)
        return all_probs / len(self.trees)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)


class NeuralNetworkModel:
    """Multi-Layer Perceptron (MLP) Classifier with ReLU hidden layer."""
    def __init__(self, hidden_dim: int = 32, alpha: float = 0.01, max_iter: int = 100, lr: float = 0.05, random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state
        self.W1: np.ndarray = np.array([])
        self.b1: np.ndarray = np.array([])
        self.W2: np.ndarray = np.array([])
        self.b2: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkModel":
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # He initialization
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_features), size=(n_features, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden_dim), size=(self.hidden_dim, 1)).astype(np.float32)
        self.b2 = 0.0

        for _ in range(self.max_iter):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = np.maximum(0, z1)  # ReLU
            z2 = np.dot(a1, self.W2) + self.b2
            probs = 1.0 / (1.0 + np.exp(-np.clip(z2.squeeze(), -25.0, 25.0)))

            # Backward pass
            dz2 = (probs - y)[:, np.newaxis]
            dW2 = (np.dot(a1.T, dz2) + self.alpha * self.W2) / n_samples
            db2 = float(np.mean(dz2))

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (z1 > 0)
            dW1 = (np.dot(X.T, dz1) + self.alpha * self.W1) / n_samples
            db1 = np.mean(dz1, axis=0)

            # Gradient step
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z2.squeeze(), -25.0, 25.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)
