"""Data pipeline for Adult Census Income: standardizing, one-hot encoding, and feature mapping."""

import csv
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from .config import InterpretAIConfig, get_default_config

class CensusDataPipeline:
    """Robust data processing pipeline with standardization and one-hot encoding."""
    def __init__(self, config: Optional[InterpretAIConfig] = None):
        self.config = config or get_default_config()
        self.header: List[str] = []
        self.raw_rows: List[List[str]] = []
        
        # Metadata and statistics
        self.feature_names: List[str] = []
        self.numeric_stats: Dict[str, Tuple[float, float]] = {}  # col -> (mean, std)
        self.category_levels: Dict[str, List[str]] = {}           # col -> [level1, level2...]
        
        # Processed arrays
        self.X: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def load_data(self, data_path: Optional[str] = None) -> List[List[str]]:
        """Load and clean Census_Income.csv."""
        path = Path(data_path or self.config.data_path)
        if not path.exists():
            # Check local file in cwd
            local_fallback = Path("Census_Income.csv")
            if local_fallback.exists():
                path = local_fallback

        if not path.exists():
            # Return synthetic sample for tests/demo
            return self._generate_synthetic_rows()

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            self.header = [h.strip() for h in next(reader, [])]
            self.raw_rows = [[val.strip() for val in r] for r in reader if len(r) == len(self.header)]

        return self.raw_rows

    def fit_transform(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fit scaler and encoders, and transform raw rows into feature matrix X and target y."""
        if not self.raw_rows:
            self.load_data()

        rows = self.raw_rows if max_samples is None else self.raw_rows[:max_samples]
        col_idx = {name: i for i, name in enumerate(self.header)}

        # 1. Compute numeric statistics (mean, std)
        for num_col in self.config.numeric_cols:
            idx = col_idx[num_col]
            vals = [float(r[idx]) for r in rows if r[idx].replace('.', '', 1).isdigit()]
            mean = float(np.mean(vals)) if vals else 0.0
            std = float(np.std(vals)) if vals else 1.0
            if std == 0:
                std = 1.0
            self.numeric_stats[num_col] = (mean, std)

        # 2. Extract distinct category levels
        for cat_col in self.config.categorical_cols:
            idx = col_idx[cat_col]
            distinct_levels = sorted(list(set(r[idx] for r in rows if r[idx] and r[idx] != '?')))
            self.category_levels[cat_col] = distinct_levels

        # 3. Build Feature Names Schema: [num_cols...] + [cat_col_level...]
        self.feature_names = []
        for num_col in self.config.numeric_cols:
            self.feature_names.append(num_col)

        for cat_col in self.config.categorical_cols:
            for lvl in self.category_levels[cat_col]:
                self.feature_names.append(f"{cat_col}_{lvl}")

        # 4. Transform Matrix X
        num_features = len(self.feature_names)
        num_rows = len(rows)
        X = np.zeros((num_rows, num_features), dtype=np.float32)
        y = np.zeros(num_rows, dtype=np.int32)

        target_idx = col_idx[self.config.target_col]

        for i, r in enumerate(rows):
            # Target: >50K is 1, <=50K is 0
            t_val = r[target_idx]
            y[i] = 1 if ">50K" in t_val else 0

            feat_cursor = 0
            # Numeric columns
            for num_col in self.config.numeric_cols:
                raw_v = float(r[col_idx[num_col]]) if r[col_idx[num_col]].replace('.', '', 1).isdigit() else self.numeric_stats[num_col][0]
                mean, std = self.numeric_stats[num_col]
                X[i, feat_cursor] = (raw_v - mean) / std
                feat_cursor += 1

            # One-hot categorical columns
            for cat_col in self.config.categorical_cols:
                val = r[col_idx[cat_col]]
                for lvl in self.category_levels[cat_col]:
                    if val == lvl:
                        X[i, feat_cursor] = 1.0
                    feat_cursor += 1

        self.X = X
        self.y = y
        self.is_fitted = True
        return self.X, self.y, self.feature_names

    def transform_single(self, input_dict: Dict[str, Any]) -> np.ndarray:
        """Transform a single input dictionary into an encoded 1 x D feature vector."""
        if not self.is_fitted:
            self.fit_transform()

        vector = np.zeros((1, len(self.feature_names)), dtype=np.float32)
        cursor = 0

        # Numeric columns
        for num_col in self.config.numeric_cols:
            raw_v = float(input_dict.get(num_col, self.numeric_stats[num_col][0]))
            mean, std = self.numeric_stats[num_col]
            vector[0, cursor] = (raw_v - mean) / std
            cursor += 1

        # Categorical columns
        for cat_col in self.config.categorical_cols:
            cur_val = str(input_dict.get(cat_col, "")).strip()
            for lvl in self.category_levels[cat_col]:
                if cur_val == lvl:
                    vector[0, cursor] = 1.0
                cursor += 1

        return vector

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split into train and test sets using deterministic permutation."""
        if not self.is_fitted:
            self.fit_transform()

        n = len(self.X)
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n)
        test_count = int(n * test_size)
        
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]

        return self.X[train_idx], self.X[test_idx], self.y[train_idx], self.y[test_idx]

    def _generate_synthetic_rows(self) -> List[List[str]]:
        """Generate synthetic dataset rows when file is not present."""
        self.header = [
            "AGE", "WORKCLASS", "EDUCATION", "EDUCATION_NUM", "MARITAL_STATUS",
            "OCCUPATION", "RELATIONSHIP", "RACE", "SEX", "HOURS_PER_WEEK",
            "NATIONALITY", "INCOME"
        ]
        sample = [
            ["39", "State-gov", "Bachelors", "13", "Never-married", "Adm-clerical", "Not-in-family", "White", "Male", "40", "United-States", "<=50K"],
            ["50", "Self-emp-not-inc", "Bachelors", "13", "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", "13", "United-States", "<=50K"],
            ["38", "Private", "HS-grad", "9", "Divorced", "Handlers-cleaners", "Not-in-family", "White", "Male", "40", "United-States", "<=50K"],
            ["53", "Private", "11th", "7", "Married-civ-spouse", "Handlers-cleaners", "Husband", "Black", "Male", "40", "United-States", "<=50K"],
            ["28", "Private", "Bachelors", "13", "Married-civ-spouse", "Prof-specialty", "Wife", "Black", "Female", "40", "Cuba", "<=50K"],
            ["52", "Self-emp-not-inc", "HS-grad", "9", "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", "45", "United-States", ">50K"],
            ["31", "Private", "Masters", "14", "Never-married", "Prof-specialty", "Not-in-family", "White", "Female", "50", "United-States", ">50K"],
            ["42", "Private", "Bachelors", "13", "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", "40", "United-States", ">50K"],
        ]
        self.raw_rows = sample * 10
        return self.raw_rows
