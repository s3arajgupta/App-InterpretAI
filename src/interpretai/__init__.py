"""InterpretAI: Production Explainable AI & Model Interpretability Suite."""

from .config import InterpretAIConfig, get_default_config
from .data_pipeline import CensusDataPipeline
from .models import LogisticRegressionModel, RandomForestModel, NeuralNetworkModel
from .explainers import (
    explain_logistic_regression,
    explain_random_forest,
    explain_permutation_importance,
    explain_shap_local
)
from .evaluator import compute_classification_metrics
from .engine import InterpretAIEngine

__all__ = [
    "InterpretAIConfig",
    "get_default_config",
    "CensusDataPipeline",
    "LogisticRegressionModel",
    "RandomForestModel",
    "NeuralNetworkModel",
    "explain_logistic_regression",
    "explain_random_forest",
    "explain_permutation_importance",
    "explain_shap_local",
    "compute_classification_metrics",
    "InterpretAIEngine",
]
