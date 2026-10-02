"""Evaluation metrics for classification models (Accuracy, Precision, Recall, F1, ROC-AUC)."""

from typing import Dict, Any, List
import numpy as np

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics."""
    y_t = np.array(y_true, dtype=np.int32)
    y_p = np.array(y_pred, dtype=np.int32)
    y_score = np.array(y_prob, dtype=np.float32)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    total = len(y_t)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # ROC-AUC via rank order (Wilcoxon-Mann-Whitney statistic)
    pos_scores = y_score[y_t == 1]
    neg_scores = y_score[y_t == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        # Compare all pairs
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        # Vectorized pairwise comparison
        pairs_greater = np.sum(pos_scores[:, np.newaxis] > neg_scores[np.newaxis, :])
        pairs_equal = np.sum(pos_scores[:, np.newaxis] == neg_scores[np.newaxis, :])
        roc_auc = (pairs_greater + 0.5 * pairs_equal) / (n_pos * n_neg)
    else:
        roc_auc = 0.5

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn
        }
    }
