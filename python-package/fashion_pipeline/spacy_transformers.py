
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import spacy
import numpy as np
from spacytextblob.spacytextblob import SpacyTextBlob  # noqa: F401


class SpacyLemmatizer(BaseEstimator, TransformerMixin):
    """
    Transformer that lemmatizes input text using spaCy.

    Removes stopwords and replaces words with their lemmatized form.

    Parameters
    ----------
    model_name : str
        Name of the spaCy language model to use (default: "en_core_web_sm").
    """

    def __init__(self, model_name="en_core_web_sm"):
        self.model_name = model_name  # just store the name, not the object

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):
            texts = [X]
        elif isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].astype(str).tolist()
        elif isinstance(X, (pd.Series, list, np.ndarray)):
            texts = pd.Series(X).astype(str).tolist()
        else:
            raise TypeError(f"""Unsupported input type for
                            SpacyLemmatizer: {type(X)}""")

        nlp = spacy.load(self.model_name)
        lemmatized = [
            ' '.join(token.lemma_ for token in doc if not token.is_stop)
            for doc in nlp.pipe(texts)
        ]
        return lemmatized


class SpacyPosFeaturesSingleColumn(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts POS-based features from a single text column.

    Computes relative frequencies (per token and per character) for adjectives.

    Parameters
    ----------
    column : str
        Name of the text column to process.
    """

    def __init__(self, column):
        self.column = column
        self._nlp = spacy.load("en_core_web_sm")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, str):
            raise TypeError("Expected a DataFrame-like object, got a string")

        text_data = pd.Series(X[self.column]).astype(str)
        features = []

        for text in text_data:
            doc = self._nlp(text)
            num_tokens = max(len([t for t in doc
                                  if not t.is_punct and not t.is_space]), 1)
            char_len = max(len(text.strip()), 1)
            pos_counts = doc.count_by(spacy.attrs.POS)

            features.append({
                f"{self.column}_rel_adjectives_per_token":
                pos_counts.get(self._nlp.vocab.strings["ADJ"], 0) / num_tokens,
                f"{self.column}_rel_adjectives_per_char":
                pos_counts.get(self._nlp.vocab.strings["ADJ"], 0) / char_len,
            })

        return pd.DataFrame(features).reset_index(drop=True)


class SpacyPosNerFeaturesMultiCharRelative(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts POS and NER-based features from multiple text
    columns. Computes relative frequencies (per token and per character)
    for adjectives, nouns, and named entities, as well as binary indicators
    for PERSON and ORG entities.

    Parameters
    ----------
    text_columns : list of str
        Names of the text columns to process.
    """

    def __init__(self, text_columns):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = X.to_frame().T
        elif isinstance(X, str):
            raise TypeError("Expected a DataFrame-like object, got a string")

        all_features = []
        nlp = spacy.load("en_core_web_sm")

        for col in self.text_columns:
            col_data = X[col]
            col_data = pd.Series(col_data).astype(str)

            col_features = []
            for text in col_data:
                doc = nlp(text)
                num_tokens = max(len([t for t in doc
                                      if not t.is_punct
                                      and not t.is_space]), 1)
                char_len = max(len(text.strip()), 1)
                pos_counts = doc.count_by(spacy.attrs.POS)

                feature_row = {
                    f"{col}_rel_adjectives_per_token":
                    pos_counts.get(nlp.vocab.strings["ADJ"], 0) / num_tokens,
                    f"{col}_rel_adjectives_per_char":
                    pos_counts.get(nlp.vocab.strings["ADJ"], 0) / char_len,
                }
                col_features.append(feature_row)

            df_features = pd.DataFrame(col_features).reset_index(drop=True)
            assert df_features.shape[0] == X.shape[0], (
                f"[SpacyPosNerFeaturesMultiCharRelative] Expected "
                f"{X.shape[0]} rows, got {df_features.shape[0]}"
            )
            all_features.append(df_features)

        return pd.concat(all_features, axis=1)


class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts text into spaCy document vectors
    using a specified language model.

    Parameters
    ----------
    model_name : str
        The name of the spaCy model to use (e.g., 'en_core_web_md').
    """

    def __init__(self, model_name='en_core_web_md'):
        self.model_name = model_name
        self.nlp = None

    def fit(self, X, y=None):
        self.nlp = spacy.load(self.model_name)
        return self

    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        elif isinstance(X, dict):
            X = pd.DataFrame([X]).iloc[:, 0]
        elif isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            X = X[:, 0]
        X = pd.Series(X).astype(str)

        return np.array([self.nlp(text).vector
                         for text in X], dtype=np.float32)


class SpacySentimentTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes sentiment features (polarity and subjectivity)
    for text columns using spaCy and spaCyTextBlob.
    """

    def __init__(self, columns=['Review Text']):
        self.columns = columns
        self.nlp = spacy.load("en_core_web_sm")
        if 'spacytextblob' not in self.nlp.pipe_names:
            self.nlp.add_pipe('spacytextblob')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, str):
            X = pd.DataFrame([{self.columns[0]: X}])
        elif not isinstance(X, pd.DataFrame):
            raise TypeError(f"Unsupported input type: {type(X)}")

        features = []

        for col in self.columns:
            texts = X[col].fillna('').astype(str)
            polarities = []
            subjectivities = []

            for text in texts:
                doc = self.nlp(text)
                polarities.append(doc._.blob.polarity)
                subjectivities.append(doc._.blob.subjectivity)

            features.append(polarities)
            features.append(subjectivities)

        result = np.array(features).T
        assert result.shape[0] == X.shape[0], (
            f"[SpacySentimentTransformer] Expected {X.shape[0]}"
            f" rows, got {result.shape[0]}"
        )
        return result
