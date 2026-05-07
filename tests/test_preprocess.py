"""Tests for the preprocessing module."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import preprocess, load_params


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.rand(n),
        "category": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n),
    })


def test_preprocess_returns_tuple(sample_dataframe):
    """Preprocess should return (X, y) tuple."""
    X, y = preprocess(sample_dataframe)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_preprocess_removes_target(sample_dataframe):
    """Target column should not be in features."""
    X, y = preprocess(sample_dataframe)
    assert "target" not in X.columns


def test_preprocess_encodes_categoricals(sample_dataframe):
    """Categorical columns should be encoded to numeric."""
    X, y = preprocess(sample_dataframe)
    assert X["category"].dtype in [np.int32, np.int64, np.float64]


def test_preprocess_no_nulls(sample_dataframe):
    """Output should have no null values."""
    # Inject some NaN
    sample_dataframe.loc[0, "feature_1"] = np.nan
    sample_dataframe.loc[5, "feature_2"] = np.nan
    X, y = preprocess(sample_dataframe)
    assert X.isnull().sum().sum() == 0


def test_preprocess_shape(sample_dataframe):
    """Feature count should be n_columns - 1 (target removed)."""
    X, y = preprocess(sample_dataframe)
    assert X.shape[1] == sample_dataframe.shape[1] - 1
