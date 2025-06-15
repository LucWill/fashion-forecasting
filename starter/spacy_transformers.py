
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
        nlp = spacy.load(self.model_name)
        texts = X.astype(str).copy().tolist()
        lemmatized = [
            ' '.join(
                token.lemma_ for token in doc
                if not token.is_stop
            )
            for doc in nlp.pipe(texts)
        ]
        return lemmatized


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
        all_features = []
        nlp = spacy.load("en_core_web_sm")
        for col in self.text_columns:
            col_features = []
            for text in X[col].astype(str):
                doc = nlp(text)
                num_tokens = len(
                    [t for t in doc if not t.is_punct and not t.is_space])
                num_tokens = max(num_tokens, 1)

                char_len = len(text.strip())
                char_len = max(char_len, 1)

                pos_counts = doc.count_by(spacy.attrs.POS)

                feature_row = {
                    f"{col}_rel_adjectives_per_token": pos_counts.get(
                        nlp.vocab.strings["ADJ"], 0) / num_tokens,

                    f"{col}_rel_adjectives_per_char": pos_counts.get(
                        nlp.vocab.strings["ADJ"], 0) / char_len,
                }
                col_features.append(feature_row)

            all_features.append(
                pd.DataFrame(col_features).reset_index(drop=True))

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
        # Handle both Series and 2D array from ColumnTransformer
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]  # select first column
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            X = X[:, 0]
        vectors = [self.nlp(str(text)).vector for text in X]
        return np.array(vectors, dtype=np.float32).copy()


class SpacySentimentTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes sentiment features (polarity and subjectivity)
    for text columns using spaCy and spaCyTextBlob.
    """

    def __init__(self, columns=['Review Text']):
        # Define which columns to process for sentiment
        self.columns = columns

        # Load the spaCy language model
        self.nlp = spacy.load("en_core_web_sm")

        # Add the spacytextblob pipeline component if not already present
        if 'spacytextblob' not in self.nlp.pipe_names:
            self.nlp.add_pipe('spacytextblob')

    def fit(self, X, y=None):
        # No fitting needed; return self
        return self

    def transform(self, X):
        # Ensure input is in DataFrame format
        X = pd.DataFrame(X)

        features = []

        # Process each specified text column
        for col in self.columns:
            # Clean and convert to string, fill missing values
            texts = X[col].fillna('').astype(str)

            # Prepare lists for polarity and subjectivity scores
            polarities = []
            subjectivities = []

            # Iterate over each text entry in the column
            for text in texts:
                doc = self.nlp(text)  # Run spaCy pipeline
                # Extract polarity and subjectivity from the TextBlob extension
                polarities.append(doc._.blob.polarity)
                subjectivities.append(doc._.blob.subjectivity)

            # Collect results per column
            features.append(polarities)       # One list per feature type
            features.append(subjectivities)

        # Transpose the final array so the shape is (n_samples, 2 * n_columns)
        return np.array(features).T
