from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


from .transformers import ReviewCountEncoder
from .spacy_transformers import SpacyLemmatizer


num_clothingid_pipeline = Pipeline([
    (
        'review_count',
        ReviewCountEncoder(column='Clothing ID'),
    ),
    (
        'scaler',
        StandardScaler(),
    ),
])

num_age_pipeline = Pipeline([
    (
        'imputer',
        SimpleImputer(strategy='mean'),
    ),
    (
        'scaler',
        StandardScaler(),
    ),
])


def log_transform(x):
    return np.log(x + 1)


num_positive_feedback_count_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('transformer', FunctionTransformer(log_transform, validate=False)),
    ('scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    (
        'imputer',
        SimpleImputer(
            strategy='most_frequent',
        )
    ),
    (
        'cat_encoder',
        OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
        )
    ),
])


def squeeze_array(x):
    return np.array(x).squeeze()


initial_text_preprocess = Pipeline([
    (
        'dimension_reshaper',
        FunctionTransformer(
            squeeze_array, validate=False
        ),
    ),
])

tfidf_pipeline = Pipeline([
    (
        'dimension_reshaper',
        initial_text_preprocess,
    ),
    (
        'lemmatizer',
        SpacyLemmatizer(),
    ),
    (
        'tfidf_vectorizer',
        TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
        ),
    ),
])
