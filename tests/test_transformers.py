import pytest
import pandas as pd
import numpy as np
from fashion_pipeline.transformers import (
    normalize_text_input,
    ReviewCountEncoder,
    CountCharacter,
    ContrastWordCounter,
    MeasureTextLength,
    CharacterCountSingleColumnTransformer,
    CharacterCountsMultiColumnTransformer,
    TfidfMultiColumnTransformer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


# ---------- normalize_text_input ----------

@pytest.mark.parametrize("input_data,expected_type", [
    ("hello", pd.Series),
    (["a", "b"], pd.Series),
    (np.array(["a", "b"]), pd.Series),
    (pd.Series(["a", "b"]), pd.Series),
])
def test_normalize_text_input_valid_types(input_data, expected_type):
    result = normalize_text_input(input_data)
    assert isinstance(result, expected_type)
    assert all(isinstance(x, str) for x in result)


# ---------- ReviewCountEncoder ----------

def test_review_count_encoder():
    df = pd.DataFrame({'Clothing ID': [101, 102, 101]})
    encoder = ReviewCountEncoder(column='Clothing ID')
    transformed = encoder.fit_transform(df)
    assert transformed.shape == (3, 1)
    assert transformed.iloc[0, 0] == 2
    assert transformed.iloc[1, 0] == 1


# ---------- CountCharacter ----------

def test_count_character():
    data = pd.Series(["aab", "abc", ""])
    counter = CountCharacter(character="a")
    result = counter.transform(data)
    expected = [[2/3], [1/3], [0]]
    np.testing.assert_almost_equal(result, expected)


# ---------- ContrastWordCounter ----------

def test_contrast_word_counter():
    data = pd.Series(["I liked it but not a lot", "However, I enjoyed it"])
    counter = ContrastWordCounter()
    result = counter.transform(data)
    np.testing.assert_array_equal(result, np.array([[3], [1]]))



# ---------- MeasureTextLength ----------

def test_measure_text_length():
    data = pd.Series(["test", "", "12345"])
    measurer = MeasureTextLength()
    result = measurer.transform(data)
    np.testing.assert_array_equal(result, np.array([[4], [0], [5]]))



# ---------- CharacterCountSingleColumnTransformer ----------

def test_character_count_single_column_transformer():
    base = CountCharacter(character="e")
    transformer = CharacterCountSingleColumnTransformer("text", base)
    df = pd.DataFrame({"text": ["hello", "excellent"]})
    transformer.fit(df)
    result = transformer.transform(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert "text_char_count" in result.columns[0]


# ---------- CharacterCountsMultiColumnTransformer ----------

def test_character_counts_multi_column_transformer():
    base = CountCharacter(character="o")
    transformer = CharacterCountsMultiColumnTransformer(["title", "review"], base)
    df = pd.DataFrame({
        "title": ["hello", "no one"],
        "review": ["soon", "zoo"]
    })
    transformer.fit(df)
    result = transformer.transform(df)
    assert result.shape == (2, 2)
    assert "title_char_count" in result.columns
    assert "review_char_count" in result.columns


# ---------- TfidfMultiColumnTransformer ----------

def test_tfidf_multi_column_transformer():
    tfidf_base = Pipeline([
        ('tfidf', TfidfVectorizer())
    ])
    transformer = TfidfMultiColumnTransformer(
        text_columns=["title", "review"],
        base_pipeline=tfidf_base
    )
    df = pd.DataFrame({
        "title": ["good", "bad"],
        "review": ["very good", "not great"]
    })
    transformer.fit(df)
    result = transformer.transform(df)
    assert result.shape[0] == 2  # samples
    assert result.shape[1] > 0   # some features
