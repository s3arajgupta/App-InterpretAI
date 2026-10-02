"""Unit tests for Census Income data pipeline."""

import numpy as np
from src.interpretai.data_pipeline import CensusDataPipeline

def test_pipeline_fit_transform():
    pipeline = CensusDataPipeline()
    X, y, feature_names = pipeline.fit_transform(max_samples=200)

    assert len(X) == 200
    assert len(y) == 200
    assert len(feature_names) > 10
    assert X.shape[1] == len(feature_names)
    assert set(np.unique(y)).issubset({0, 1})

def test_transform_single():
    pipeline = CensusDataPipeline()
    pipeline.fit_transform(max_samples=200)

    sample = {
        "AGE": 40,
        "EDUCATION_NUM": 13,
        "HOURS_PER_WEEK": 45,
        "WORKCLASS": "Private",
        "MARITAL_STATUS": "Married-civ-spouse",
        "OCCUPATION": "Exec-managerial",
        "RELATIONSHIP": "Husband",
        "RACE": "White",
        "SEX": "Male",
        "NATIONALITY": "United-States",
        "EDUCATION": "Bachelors"
    }

    vec = pipeline.transform_single(sample)
    assert vec.shape == (1, len(pipeline.feature_names))
    assert np.count_nonzero(vec) > 0

def test_train_test_split():
    pipeline = CensusDataPipeline()
    pipeline.fit_transform(max_samples=200)
    X_train, X_test, y_train, y_test = pipeline.get_train_test_split(test_size=0.25)

    assert len(X_train) == 150
    assert len(X_test) == 50
    assert len(y_train) == 150
    assert len(y_test) == 50
