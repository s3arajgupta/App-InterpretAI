"""Unit tests for classification evaluation metrics."""

import numpy as np
from src.interpretai.evaluator import compute_classification_metrics

def test_compute_metrics_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])

    m = compute_classification_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["roc_auc"] == 1.0

def test_compute_metrics_imperfect():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_prob = np.array([0.8, 0.6, 0.4, 0.2])

    m = compute_classification_metrics(y_true, y_pred, y_prob)
    assert 0.0 < m["accuracy"] < 1.0
    assert 0.0 < m["roc_auc"] <= 1.0
