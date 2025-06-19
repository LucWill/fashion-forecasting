import pandas as pd
import numpy as np
from fashion_pipeline.spacy_transformers import (
    SpacyLemmatizer,
    SpacyPosFeaturesSingleColumn,
    SpacyPosNerFeaturesMultiCharRelative,
    SpacyVectorTransformer,
    SpacySentimentTransformer
)


def test_spacy_lemmatizer_removes_stopwords_and_lemmatizes():
    lemmatizer = SpacyLemmatizer()
    input_data = pd.Series(["The cats are running fast"])
    output = lemmatizer.transform(input_data)
    assert isinstance(output, list)
    assert all(isinstance(x, str) for x in output)
    assert "cat" in output[0]  # lemmatized
    assert "running" not in output[0]  # removed if stopword


def test_spacy_pos_features_single_column_output_shape_and_keys():
    transformer = SpacyPosFeaturesSingleColumn(column="text")
    df = pd.DataFrame({"text": ["This is a beautiful dress."]})
    result = transformer.transform(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 2)
    assert "text_rel_adjectives_per_token" in result.columns
    assert "text_rel_adjectives_per_char" in result.columns


def test_spacy_pos_ner_features_multi_column_output_shape():
    transformer = SpacyPosNerFeaturesMultiCharRelative(["Title",
                                                        "Review Text"])
    df = pd.DataFrame({
        "Title": ["Lovely material"],
        "Review Text": ["This dress is elegant and comfortable."]
    })
    result = transformer.transform(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert "Title_rel_adjectives_per_token" in result.columns
    assert "Review Text_rel_adjectives_per_char" in result.columns


def test_spacy_vector_transformer_vector_shape():
    transformer = SpacyVectorTransformer("en_core_web_sm")
    input_data = pd.Series(["Nice fabric"])
    transformer.fit(input_data)
    result = transformer.transform(input_data)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1
    assert result.shape[1] > 0  # vector dimensionality


def test_spacy_sentiment_transformer_output_shape_and_range():
    transformer = SpacySentimentTransformer(columns=["Review Text"])
    df = pd.DataFrame({"Review Text": ["I love this item. It fits great!"]})
    result = transformer.transform(df)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)
    polarity, subjectivity = result[0]
    assert -1.0 <= polarity <= 1.0
    assert 0.0 <= subjectivity <= 1.0
