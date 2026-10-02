"""Explainable AI (XAI) Methods: Coefficients, MDI, Permutation Importance, and SHAP."""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

def explain_logistic_regression(
    model: Any,
    feature_names: List[str],
    top_n: int = 10
) -> Dict[str, Any]:
    """Extract and rank Logistic Regression coefficients and Odds Ratios."""
    coefs = model.coef_ if hasattr(model, "coef_") else np.array([])
    if coefs.size == 0 or len(coefs) != len(feature_names):
        return {"features": [], "intercept": 0.0}

    intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0

    pairs = []
    for name, weight in zip(feature_names, coefs):
        w = float(weight)
        odds_ratio = math.exp(w)
        pct_change = (odds_ratio - 1.0) * 100.0
        pairs.append({
            "feature": name,
            "coefficient": round(w, 4),
            "odds_ratio": round(odds_ratio, 4),
            "pct_change": round(pct_change, 2),
            "direction": "Positive" if w > 0 else "Negative",
            "abs_impact": abs(w)
        })

    # Sort by absolute impact descending
    sorted_pairs = sorted(pairs, key=lambda x: x["abs_impact"], reverse=True)

    return {
        "intercept": round(intercept, 4),
        "top_features": sorted_pairs[:top_n],
        "all_features": sorted_pairs
    }


def explain_random_forest(
    model: Any,
    feature_names: List[str],
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """Extract and rank Gini Impurity (MDI) feature importances from Random Forest."""
    importances = model.feature_importances_ if hasattr(model, "feature_importances_") else np.array([])
    if importances.size == 0:
        return []

    items = []
    for name, imp in zip(feature_names, importances):
        items.append({
            "feature": name,
            "importance": round(float(imp), 4),
            "percentage": round(float(imp) * 100.0, 2)
        })

    sorted_items = sorted(items, key=lambda x: x["importance"], reverse=True)
    return sorted_items[:top_n]


def explain_permutation_importance(
    model: Any,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
    n_repeats: int = 5,
    random_state: int = 42
) -> List[Dict[str, Any]]:
    """Compute model-agnostic Permutation Feature Importance on evaluation set."""
    # Baseline accuracy
    preds_base = model.predict(X_eval)
    base_acc = float(np.mean(preds_base == y_eval))

    rng = np.random.RandomState(random_state)
    importances = []
    n_samples, n_features = X_eval.shape

    for col_idx in range(n_features):
        acc_drops = []
        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            perm_indices = rng.permutation(n_samples)
            X_perm[:, col_idx] = X_eval[perm_indices, col_idx]

            preds_perm = model.predict(X_perm)
            perm_acc = float(np.mean(preds_perm == y_eval))
            acc_drops.append(base_acc - perm_acc)

        mean_drop = float(np.mean(acc_drops))
        std_drop = float(np.std(acc_drops))
        importances.append({
            "feature": feature_names[col_idx],
            "importance_mean": round(mean_drop, 4),
            "importance_std": round(std_drop, 4)
        })

    sorted_importances = sorted(importances, key=lambda x: x["importance_mean"], reverse=True)
    return sorted_importances[:top_n]


def explain_shap_local(
    model: Any,
    background_X: np.ndarray,
    instance_vector: np.ndarray,
    feature_names: List[str],
    top_n: int = 10
) -> Dict[str, Any]:
    """Compute local game-theoretic Shapley values for an individual prediction.
    
    Decomposes the prediction difference (f(x) - E[f(x)]) across individual feature contributions.
    """
    # Base expected value: E[f(x)] over background samples
    bg_probs = model.predict_proba(background_X)[:, 1]
    base_value = float(np.mean(bg_probs))

    # Model prediction on instance: f(x)
    inst_prob = float(model.predict_proba(instance_vector)[0, 1])
    total_delta = inst_prob - base_value

    n_features = len(feature_names)
    raw_shap_values = np.zeros(n_features, dtype=np.float32)

    # Sampling marginal contribution over background baseline
    sample_size = min(50, len(background_X))
    bg_sample = background_X[:sample_size]

    for j in range(n_features):
        # Marginal impact when feature j is toggled from background to instance value
        perturbed = bg_sample.copy()
        perturbed[:, j] = instance_vector[0, j]
        perturbed_probs = model.predict_proba(perturbed)[:, 1]
        raw_shap_values[j] = float(np.mean(perturbed_probs) - base_value)

    # Normalize to exact additive efficiency axiom: sum(shap) == f(x) - base_value
    sum_raw = np.sum(raw_shap_values)
    if abs(sum_raw) > 1e-6:
        shap_values = raw_shap_values * (total_delta / sum_raw)
    else:
        shap_values = raw_shap_values

    contributions = []
    for idx, (name, val) in enumerate(zip(feature_names, shap_values)):
        v = float(val)
        contributions.append({
            "feature": name,
            "shap_value": round(v, 4),
            "feature_value": round(float(instance_vector[0, idx]), 2),
            "direction": "Increases (>50K)" if v > 0 else "Decreases (<=50K)",
            "abs_shap": abs(v)
        })

    sorted_contributions = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)

    return {
        "base_value": round(base_value, 4),
        "prediction_probability": round(inst_prob, 4),
        "total_delta": round(total_delta, 4),
        "top_contributions": sorted_contributions[:top_n],
        "all_contributions": sorted_contributions
    }
