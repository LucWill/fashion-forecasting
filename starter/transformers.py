from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.pipeline import clone
from scipy.sparse import hstack


class ReviewCountEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer that encodes each row with the count of how often a given
    value (e.g., product ID) appears in the column. Useful for
    frequency-based encoding.

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
        encoded = X[self.column].map(
            self.review_counts_).fillna(self.fill_value)
        return encoded.to_frame(name=f'{self.column}_review_count')


class CountCharacter(BaseEstimator, TransformerMixin):
    """
    Transformer that counts the occurrences of a specified character
    in a string column.

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
        return [[(text.count(self.character)/len(text))] for text in X]


class ContrastWordCounter(BaseEstimator, TransformerMixin):
    """
    Transformer that counts the number of predefined contrast/negative words
    in text entries.

    Parameters
    ----------
    words : list of str, optional
        List of words to count in each text entry. Defaults to a set
        of common contrastive/negative words.
    """

    def __init__(self, words=None):
        if words is None:
            self.words = ['but', 'however', 'not', "n't", 'although',
                          'yet', 'too', 'no', 'dissapoint', 'poor',
                          'unfortunat']
        else:
            self.words = words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            [sum(word in text.lower() for word in self.words)] for text in X]


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


class CharacterCountsMultiColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a base character-counting pipeline
    to multiple text columns. It clones and fits one instance
    of the base pipeline per column, and then prefixes the resulting
    feature names with the respective column name for clarity.

    Parameters
    ----------
    text_columns : list of str
        The text columns to which the base pipeline will be applied.

    base_pipeline : sklearn Pipeline
        A pipeline or transformer that extracts character-based features
        from a single column.
    """

    def __init__(self, text_columns, base_pipeline):
        self.text_columns = text_columns
        self.base_pipeline = base_pipeline
        self.pipelines = {}

    def fit(self, X, y=None):
        # Fit a separate copy of the base pipeline to each text column
        for col in self.text_columns:
            pipe = clone(self.base_pipeline)
            pipe.fit(X[col], y)
            self.pipelines[col] = pipe
        return self

    def transform(self, X):
        all_features = []

        # Apply the fitted pipeline to each column individually
        for col in self.text_columns:
            col_df = self.pipelines[col].transform(X[col])

            # If the output is already a DataFrame, rename its columns
            if isinstance(col_df, pd.DataFrame):
                col_df.columns = [f"{col}__{c}" for c in col_df.columns]

            else:
                # Handle 2D NumPy arrays (e.g., shape [n_samples, >1])
                if len(col_df.shape) == 2 and col_df.shape[1] > 1:
                    col_df = pd.DataFrame(
                        col_df,
                        columns=[f"{col}_char_count_{i}"
                                 for i in range(col_df.shape[1])]
                    )
                else:
                    # Handle 1D arrays or single-column 2D arrays
                    col_df = pd.DataFrame(
                        col_df, columns=[f"{col}_char_count"])

            # Reset index to avoid alignment issues during concatenation
            all_features.append(col_df.reset_index(drop=True))

        # Concatenate all column feature DataFrames side by side
        return pd.concat(all_features, axis=1)


class TfidfMultiColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a base text pipeline (e.g., TF-IDF)
    to multiple text columns, and horizontally stacks the resulting
    feature matrices.

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
