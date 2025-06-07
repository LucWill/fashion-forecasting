
from sklearn.base import BaseEstimator, TransformerMixin

class ReviewCountEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that encodes each row with the count of how often a given value 
    (e.g., product ID) appears in the column. Useful for frequency-based encoding.

    Parameters
    ----------
    column : str
        Name of the column to count values for.
    fill_value : int or float, default=1
        Value to use if a value is not found in the training data.
    """

    def __init__(self, column, fill_value=1):
        self.column = column
        self.fill_value = fill_value
        self.review_counts_ = {}

    def fit(self, X, y=None):
        self.review_counts_ = X[self.column].value_counts().to_dict()
        return self

    def transform(self, X):
        encoded = X[self.column].map(self.review_counts_).fillna(self.fill_value)
        return encoded.to_frame(name=f'{self.column}_review_count')


class CountCharacter(BaseEstimator, TransformerMixin):
    """
    Transformer that counts the occurrences of a specified character in a string column.

    Parameters
    ----------
    character : str
        The character to count in each text entry.
    """

    def __init__(self, character: str):
        self.character = character

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[text.count(self.character)] for text in X]

class MeasureTextLength(BaseEstimator, TransformerMixin):
    """
    Transformer that measures the length of text entries.

    Returns a column with the number of characters for each entry.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[len(text)] for text in X]

import spacy
from sklearn.base import BaseEstimator, TransformerMixin

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

import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

class SpacyPosNerFeaturesMultiCharRelative(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts POS and NER-based features from multiple text columns.
    Computes relative frequencies (per token and per character) for adjectives, nouns,
    and named entities, as well as binary indicators for PERSON and ORG entities.

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
                num_tokens = len([t for t in doc if not t.is_punct and not t.is_space])
                num_tokens = max(num_tokens, 1)

                char_len = len(text.strip())
                char_len = max(char_len, 1)

                pos_counts = doc.count_by(spacy.attrs.POS)
                entity_types = [ent.label_ for ent in doc.ents]
                num_ents = len(entity_types)

                feature_row = {
                    f"{col}_rel_adjectives_per_token": pos_counts.get(nlp.vocab.strings["ADJ"], 0) / num_tokens,
                    f"{col}_rel_nouns_per_token": pos_counts.get(nlp.vocab.strings["NOUN"], 0) / num_tokens,
                    f"{col}_rel_entities_per_token": num_ents / num_tokens,

                    f"{col}_rel_adjectives_per_char": pos_counts.get(nlp.vocab.strings["ADJ"], 0) / char_len,
                    f"{col}_rel_nouns_per_char": pos_counts.get(nlp.vocab.strings["NOUN"], 0) / char_len,
                    f"{col}_rel_entities_per_char": num_ents / char_len,

                    f"{col}_has_person": int("PERSON" in entity_types),
                    f"{col}_has_org": int("ORG" in entity_types),
                }
                col_features.append(feature_row)

            all_features.append(pd.DataFrame(col_features).reset_index(drop=True))

        return pd.concat(all_features, axis=1)

import spacy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts text into spaCy document vectors using a specified language model.

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
        return np.array([self.nlp(str(text)).vector for text in X])

from scipy.sparse import hstack
from sklearn.pipeline import clone
from sklearn.base import BaseEstimator, TransformerMixin

class TfidfMultiColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a base text pipeline (e.g., TF-IDF) to multiple text columns,
    and horizontally stacks the resulting feature matrices.

    Parameters
    ----------
    text_columns : list of str
        The names of the columns to apply the base_pipeline to.
    base_pipeline : sklearn Pipeline
        A pipeline or transformer to clone and fit separately for each column.
    """

    def __init__(self, text_columns, base_pipeline):
        self.text_columns = text_columns
        self.base_pipeline = base_pipeline
        self.pipelines = {}

    def fit(self, X, y=None):
        for col in self.text_columns:
            pipe = clone(self.base_pipeline)
            pipe.fit(X[col], y)
            self.pipelines[col] = pipe
        return self

    def transform(self, X):
        transformed = [
            self.pipelines[col].transform(X[col]) for col in self.text_columns
        ]
        return hstack(transformed)

from sklearn.pipeline import clone
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CharacterCountsMultiColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a character-counting pipeline to multiple text columns
    and concatenates the resulting feature DataFrames side‑by‑side.

    Parameters
    ----------
    text_columns : list of str
        The names of the text columns to process.
    base_pipeline : sklearn Pipeline
        A pipeline/transformer that outputs a DataFrame with character‑based features.
    """

    def __init__(self, text_columns, base_pipeline):
        self.text_columns = text_columns
        self.base_pipeline = base_pipeline
        self.pipelines = {}

    def fit(self, X, y=None):
        for col in self.text_columns:
            pipe = clone(self.base_pipeline)
            pipe.fit(X[col], y)
            self.pipelines[col] = pipe
        return self

    def transform(self, X):
        all_features = []
        for col in self.text_columns:
            col_df = self.pipelines[col].transform(X[col])
            # Ensure DataFrame type and rename columns to reflect source column
            if isinstance(col_df, pd.DataFrame):
                col_df.columns = [f"{col}__{c}" for c in col_df.columns]
            else:
                col_df = pd.DataFrame(col_df, columns=[f"{col}_char_count"])
            all_features.append(col_df.reset_index(drop=True))
        return pd.concat(all_features, axis=1)
