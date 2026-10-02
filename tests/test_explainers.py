"""Unit tests for XAI explainer algorithms."""

import numpy as np
from src.interpretai.models import LogisticRegressionModel, RandomForestModel
from src.interpretai.explainers import (
    explain_logistic_regression,
    explain_random_forest,
    explain_permutation_importance,
    explain_shap_local
)

def test_explain_logistic_regression():
    model = LogisticRegressionModel()
    model.coef_ = np.array([0.5, -1.2, 0.0, 0.8])
    model.intercept_ = -0.3
    feature_names = ["age", "hours", "education", "capital_gain"]

    exp = explain_logistic_regression(model, feature_names, top_n=2)
    assert len(exp["top_features"]) == 2
    # Top absolute impact should be hours (-1.2)
    assert exp["top_features"][0]["feature"] == "hours"
    assert exp["top_features"][0]["direction"] == "Negative"

def test_explain_random_forest():
    model = RandomForestModel()
    model.feature_importances_ = np.array([0.1, 0.5, 0.3, 0.1])
    feature_names = ["f1", "f2", "f3", "f4"]

    exp = explain_random_forest(model, feature_names, top_n=2)
    assert len(exp) == 2
    assert exp[0]["feature"] == "f2"

def test_explain_shap_local_efficiency_axiom():
    # Test that sum of Shapley values equals f(x) - E[f(x)]
    X = np.random.randn(50, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    model = LogisticRegressionModel(max_iter=30).fit(X, y)

    inst = np.array([[1.5, 0.2, -0.5, 0.8]], dtype=np.float32)
    feature_names = ["A", "B", "C", "D"]

    shap_res = explain_shap_local(model, background_X=X, instance_vector=inst, feature_names=feature_names)
    assert "base_value" in shap_res
    assert "prediction_probability" in shap_res
    
    # Check additive efficiency
    shap_sum = sum(c["shap_value"] for c in shap_res["all_contributions"])
    expected_delta = shap_res["prediction_probability"] - shap_res["base_value"]
    assert abs(shap_sum - expected_delta) < 1e-3
