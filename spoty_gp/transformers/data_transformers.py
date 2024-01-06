from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical features encoder",
                    OneHotEncoder(drop="first"),
                    self.columns,
                ),
            ],
            remainder="passthrough",
        )

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X, y=None):
        return self.preprocessor.transform(X)
