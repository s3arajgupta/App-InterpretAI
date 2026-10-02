"""Configuration parameters and schemas for InterpretAI."""

from typing import List, Tuple
from pydantic import BaseModel, Field

class InterpretAIConfig(BaseModel):
    """Configuration hyperparameters and dataset schemas."""
    
    # Dataset Paths & Columns
    data_path: str = Field(default="Census_Income.csv", description="Path to Adult Census Income dataset")
    target_col: str = Field(default="INCOME", description="Target classification column")
    
    numeric_cols: List[str] = Field(
        default=["AGE", "EDUCATION_NUM", "HOURS_PER_WEEK"],
        description="Continuous numerical features requiring standardization"
    )
    
    categorical_cols: List[str] = Field(
        default=[
            "WORKCLASS", "EDUCATION", "MARITAL_STATUS", "OCCUPATION",
            "RELATIONSHIP", "RACE", "SEX", "NATIONALITY"
        ],
        description="Categorical features requiring one-hot encoding"
    )
    
    # Model Hyperparameters
    test_size: float = Field(default=0.2, description="Evaluation holdout fraction")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    
    # Logistic Regression
    lr_c: float = Field(default=0.205, description="Inverse regularization parameter C")
    lr_max_iter: int = Field(default=300, description="Max solver iterations")
    
    # Random Forest
    rf_n_estimators: int = Field(default=100, description="Number of decision trees")
    rf_max_depth: int = Field(default=20, description="Maximum tree depth")
    
    # Multi-Layer Perceptron (Neural Network)
    mlp_hidden_layers: Tuple[int, ...] = Field(default=(50, 50), description="Hidden layer architecture")
    mlp_alpha: float = Field(default=0.06, description="L2 penalty parameter")
    mlp_max_iter: int = Field(default=150, description="Maximum training iterations")
    
    # Explainability & Sampling
    shap_sample_size: int = Field(default=100, description="Background sample size for SHAP estimation")
    permutation_repeats: int = Field(default=5, description="Permutation importance evaluation repeats")

def get_default_config() -> InterpretAIConfig:
    """Return default InterpretAI configuration instance."""
    return InterpretAIConfig()
