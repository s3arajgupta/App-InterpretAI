"""Unit tests for machine learning models."""

import numpy as np
from src.interpretai.models import LogisticRegressionModel, RandomForestModel, NeuralNetworkModel

def test_logistic_regression():
    X = np.random.randn(100, 10).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)

    model = LogisticRegressionModel(max_iter=50)
    model.fit(X, y)

    assert model.coef_.shape == (10,)
    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

def test_random_forest():
    X = np.random.randn(100, 10).astype(np.float32)
    y = (X[:, 1] > 0).astype(np.int32)

    model = RandomForestModel(n_estimators=10)
    model.fit(X, y)

    assert model.feature_importances_.shape == (10,)
    assert np.allclose(np.sum(model.feature_importances_), 1.0)

    preds = model.predict(X)
    assert len(preds) == 100

def test_neural_network():
    X = np.random.randn(80, 8).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

    model = NeuralNetworkModel(hidden_dim=16, max_iter=30)
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (80, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
