import pytest
import pandas as pd
import numpy as np
from fashion_pipeline.pipeline import (
    num_clothingid_pipeline,
    num_age_pipeline,
    num_positive_feedback_count_pipeline,
    cat_pipeline,
    tfidf_pipeline
)

# ---- 1. NUMERIC PIPELINE TESTS ---- #

def test_num_clothingid_pipeline_output_shape():
    df = pd.DataFrame({'Clothing ID': [1001, 1002, 1001]})
    result = num_clothingid_pipeline.fit_transform(df)
    assert result.shape == (3, 1)
    assert isinstance(result, np.ndarray)

def test_num_age_pipeline_scaling():
    df = pd.DataFrame({'Age': [25, 35, np.nan]})
    result = num_age_pipeline.fit_transform(df)
    assert result.shape == (3, 1)
    assert not np.isnan(result).any()

def test_log_transform_pipeline():
    df = pd.DataFrame({'Positive Feedback Count': [0, 3, 7]})
    result = num_positive_feedback_count_pipeline.fit_transform(df)
    
    # Should be a column vector
    assert result.shape == (3, 1)

    # No NaNs or infs
    assert not np.isnan(result).any()
    assert np.isfinite(result).all()

    # Mean ≈ 0 and std ≈ 1 (since we scaled)
    np.testing.assert_almost_equal(result.mean(), 0.0, decimal=6)
    np.testing.assert_almost_equal(result.std(), 1.0, decimal=6)

# ---- 2. CATEGORICAL PIPELINE ---- #

def test_cat_pipeline_output_shape():
    df = pd.DataFrame({
        'Division Name': ['General', 'Petites', 'General'],
    })
    result = cat_pipeline.fit_transform(df)
    assert result.shape[0] == 3  # 3 samples
    assert isinstance(result, np.ndarray)

def test_cat_pipeline_handles_unknown():
    train = pd.DataFrame({'Division Name': ['General', 'Petites']})
    test = pd.DataFrame({'Division Name': ['Tall']})
    cat_pipeline.fit(train)
    result = cat_pipeline.transform(test)
    assert result.shape[0] == 1
    assert isinstance(result, np.ndarray)

# ---- 3. TEXT PIPELINE ---- #

def test_tfidf_pipeline_output_type():
    df = pd.Series([
        "This is a wonderful dress!",
        "I did not like it at all.",
        "Fantastic and elegant."
    ])
    result = tfidf_pipeline.fit_transform(df)
    assert hasattr(result, "shape") and result.shape[0] == 3
    assert result.shape[1] > 0
