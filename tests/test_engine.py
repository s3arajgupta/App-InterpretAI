"""Unit tests for InterpretAIEngine facade."""

from src.interpretai.engine import InterpretAIEngine

def test_engine_initialization_and_predictions():
    engine = InterpretAIEngine()
    assert engine.models_trained is True

    evals = engine.evaluate_all_models()
    assert "Logistic Regression" in evals
    assert "Random Forest" in evals
    assert "Neural Network" in evals
    assert evals["Random Forest"]["accuracy"] > 0.65

def test_engine_individual_explanation():
    engine = InterpretAIEngine()
    sample_profile = {
        "AGE": 45,
        "EDUCATION_NUM": 13,
        "HOURS_PER_WEEK": 40,
        "WORKCLASS": "Private",
        "MARITAL_STATUS": "Married-civ-spouse",
        "OCCUPATION": "Exec-managerial",
        "RELATIONSHIP": "Husband",
        "RACE": "White",
        "SEX": "Male",
        "NATIONALITY": "United-States",
        "EDUCATION": "Bachelors"
    }

    res = engine.explain_individual(sample_profile, model_name="Random Forest", top_n=5)
    assert "prediction_probability" in res
    assert "base_value" in res
    assert len(res["top_contributions"]) == 5

def test_engine_global_explanations():
    engine = InterpretAIEngine()
    lr_exp = engine.get_global_explanations(model_name="Logistic Regression", top_n=5)
    assert len(lr_exp["top_features"]) == 5

    rf_exp = engine.get_global_explanations(model_name="Random Forest", top_n=5)
    assert len(rf_exp["top_features"]) == 5
