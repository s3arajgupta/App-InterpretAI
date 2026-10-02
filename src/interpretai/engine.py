"""Unified Engine Facade for InterpretAI."""

from typing import Dict, Any, List, Optional
import numpy as np

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

class InterpretAIEngine:
    """Production Facade for Explainable AI model training, scoring, and explanations."""
    def __init__(self, config: Optional[InterpretAIConfig] = None):
        self.config = config or get_default_config()
        self.pipeline = CensusDataPipeline(self.config)
        self.pipeline.fit_transform(max_samples=5000)  # High performance training subset

        self.X_train, self.X_test, self.y_train, self.y_test = self.pipeline.get_train_test_split(
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        # Initialize models
        self.lr_model = LogisticRegressionModel(c=self.config.lr_c, max_iter=self.config.lr_max_iter)
        self.rf_model = RandomForestModel(n_estimators=self.config.rf_n_estimators, random_state=self.config.random_state)
        self.nn_model = NeuralNetworkModel(max_iter=self.config.mlp_max_iter, random_state=self.config.random_state)

        self.models_trained: bool = False
        self._train_models()

    def _train_models(self):
        """Train all three models on training split."""
        self.lr_model.fit(self.X_train, self.y_train)
        self.rf_model.fit(self.X_train, self.y_train)
        self.nn_model.fit(self.X_train, self.y_train)
        self.models_trained = True

    def get_model(self, model_name: str) -> Any:
        """Retrieve model instance by name."""
        name_lower = model_name.lower()
        if "logistic" in name_lower or "lr" in name_lower:
            return self.lr_model
        elif "random" in name_lower or "rf" in name_lower or "forest" in name_lower:
            return self.rf_model
        elif "neural" in name_lower or "mlp" in name_lower or "nn" in name_lower:
            return self.nn_model
        return self.rf_model

    def explain_individual(self, profile_dict: Dict[str, Any], model_name: str = "Random Forest", top_n: int = 10) -> Dict[str, Any]:
        """Generate prediction and local SHAP explanations for an individual applicant."""
        model = self.get_model(model_name)
        inst_vector = self.pipeline.transform_single(profile_dict)

        shap_results = explain_shap_local(
            model=model,
            background_X=self.X_train[:self.config.shap_sample_size],
            instance_vector=inst_vector,
            feature_names=self.pipeline.feature_names,
            top_n=top_n
        )
        return shap_results

    def get_global_explanations(self, model_name: str = "Random Forest", top_n: int = 10) -> Dict[str, Any]:
        """Retrieve global feature importances for the requested model."""
        name_lower = model_name.lower()
        if "logistic" in name_lower:
            return explain_logistic_regression(self.lr_model, self.pipeline.feature_names, top_n=top_n)
        elif "random" in name_lower or "forest" in name_lower:
            items = explain_random_forest(self.rf_model, self.pipeline.feature_names, top_n=top_n)
            return {"top_features": items}
        elif "neural" in name_lower or "mlp" in name_lower:
            items = explain_permutation_importance(
                self.nn_model, self.X_test[:200], self.y_test[:200],
                self.pipeline.feature_names, top_n=top_n
            )
            return {"top_features": items}
        return {"top_features": []}

    def evaluate_all_models(self) -> Dict[str, Any]:
        """Compute evaluation metrics across all 3 models on the test set."""
        evals = {}
        for name, m in [("Logistic Regression", self.lr_model), ("Random Forest", self.rf_model), ("Neural Network", self.nn_model)]:
            probs = m.predict_proba(self.X_test)[:, 1]
            preds = m.predict(self.X_test)
            metrics = compute_classification_metrics(self.y_test, preds, probs)
            evals[name] = metrics
        return evals
