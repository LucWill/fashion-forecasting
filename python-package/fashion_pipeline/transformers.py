from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.pipeline import clone
from scipy.sparse import hstack
import numpy as np
from sklearn.utils.validation import check_is_fitted


def normalize_text_input(X):
    """
    Ensures X is a list-like object of strings suitable for iteration.

    Supports input types:
    - str
    - list of str
    - np.ndarray (1D or 0D)
    - pd.Series
    - other array-likes

    Returns:
    - pd.Series of strings
    """
    if isinstance(X, str):
        return pd.Series([X])
    elif isinstance(X, (np.ndarray, pd.Series)) and X.ndim == 0:
        return pd.Series([str(X.item())])
    elif isinstance(X, (np.ndarray, pd.Series, list)):
        return pd.Series(X).astype(str)
    else:
        raise TypeError(f"Unsupported input type: {type(X)}")


class ReviewCountEncoder(BaseEstimator, TransformerMixin):
    """a
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
        # Ensure X is a list or Series of strings
        X = normalize_text_input(X)

        return np.array([[(text.count(self.character) / max(
            len(text), 1))] for text in X])


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
        # Ensure X is a list or Series of strings
        X = normalize_text_input(X)

        return [
            [sum(word in text.lower() for word in self.words)] for text in X
        ]


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
        X = normalize_text_input(X)
        return [[len(text)] for text in X]


class CharacterCountSingleColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a base character-counting pipeline
    to a single column.
    """

    def __init__(self, column, base_pipeline):
        self.column = column
        self.base_pipeline = base_pipeline
        self.fitted_pipeline = None

    def fit(self, X, y=None):
        print(f"[CharacterCountSingleColumnTransformer] "
              f"fit() called on column '{self.column}'")
        col_data = normalize_text_input(X[self.column]
                                        if isinstance(X, pd.DataFrame) else X)
        self.fitted_pipeline = clone(self.base_pipeline)
        self.fitted_pipeline.fit(col_data, y)
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_pipeline')
        print(f"[CharacterCountSingleColumnTransformer] "
              f"transform() called on column '{self.column}'")

        col_data = normalize_text_input(X[self.column]
                                        if isinstance(X, pd.DataFrame) else X)
        transformed = self.fitted_pipeline.transform(col_data)

        # Wrap output in DataFrame and name columns
        if isinstance(transformed, pd.DataFrame):
            transformed.columns = [f"{self.column}__{c}"
                                   for c in transformed.columns]
            return transformed.reset_index(drop=True)
        elif isinstance(transformed, np.ndarray):
            if transformed.ndim == 1 or transformed.shape[1] == 1:
                return pd.DataFrame(transformed,
                                    columns=[f"{self.column}_char_count"])
            else:
                return pd.DataFrame(
                    transformed,
                    columns=[f"{self.column}_char_count_{i}"
                             for i in range(transformed.shape[1])]
                )
        else:
            raise TypeError(f"Unsupported transformed "
                            f"output type: {type(transformed)}")


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
        for col in self.text_columns:
            pipe = clone(self.base_pipeline)
            pipe.fit(X[col], y)
            self.pipelines[col] = pipe
        return self

    def transform(self, X):
        all_features = []

        for col in self.text_columns:
            col_data = X[col] if isinstance(X, pd.DataFrame) else X

            # Ensure col_data is a list of strings
            if isinstance(col_data, str):
                col_data = [col_data]
            elif isinstance(col_data, pd.Series):
                col_data = col_data.astype(str).tolist()
            elif isinstance(col_data, (np.ndarray, list)):
                col_data = pd.Series(col_data).astype(str).tolist()
            else:
                raise TypeError(f"Unsupported input type for "
                                f"column '{col}': {type(col_data)}")

            col_df = self.pipelines[col].transform(col_data)

            if isinstance(col_df, pd.DataFrame):
                col_df.columns = [f"{col}__{c}" for c in col_df.columns]
            else:
                if len(col_df.shape) == 2 and col_df.shape[1] > 1:
                    col_df = pd.DataFrame(
                        col_df, columns=[f"{col}_char_count_{i}"
                                         for i in range(col_df.shape[1])]
                    )
                else:
                    col_df = pd.DataFrame(col_df,
                                          columns=[f"{col}_char_count"])

            all_features.append(col_df.reset_index(drop=True))

        result = pd.concat(all_features, axis=1)

        # Remove or relax this assertion for single-row inference
        if result.shape[0] != X.shape[0]:
            print(f"[Warning] Output rows ({result.shape[0]}) "
                  f"!= input rows ({X.shape[0]})")

        return result


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
        # ✅ Ensure X is a DataFrame with the expected columns
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = pd.DataFrame(X, columns=self.text_columns)
        elif isinstance(X, list):
            X = pd.DataFrame(X, columns=self.text_columns)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError(f"Unsupported input type: {type(X)}")

        # ✅ Validate that expected columns are present
        if set(self.text_columns) - set(X.columns):
            missing = list(set(self.text_columns) - set(X.columns))
            raise ValueError(f"Missing expected columns: {missing}")

        # ✅ Apply transformers column-wise
        transformed = [
            self.pipelines[col].transform(X[col].astype(str))
            for col in self.text_columns
        ]
        return hstack(transformed)
