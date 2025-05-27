import pandas as pd
import numpy as np
from typing import List, Union, Any, Dict, Optional
from pandas.api.types import is_numeric_dtype, CategoricalDtype

class NullImputer:
    """
    Impute or drop missing values in specified columns.
    
    Parameters:
        strategy: str, default='mean'
            Imputation strategy: 'mean', 'median', 'mode', 'constant', or 'drop'.
        fill_value: Any, optional
            Value to use if strategy='constant'.
        columns: List[str], optional
            Columns to impute. If None, all columns with missing values are imputed.
    """
    def __init__(self, strategy: str = "mean", fill_value: Any = None, columns: Optional[List[str]] = None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self.impute_values_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or X.columns
        if self.strategy == "drop":
            return self
        for col in cols:
            if self.strategy == "mean":
                self.impute_values_[col] = X[col].mean()
            elif self.strategy == "median":
                self.impute_values_[col] = X[col].median()
            elif self.strategy == "mode":
                self.impute_values_[col] = X[col].mode().iloc[0] if not X[col].mode().empty else np.nan
            elif self.strategy == "constant":
                self.impute_values_[col] = self.fill_value
            else:
                raise ValueError(f"Invalid strategy: {self.strategy}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = self.columns or X.columns
        if self.strategy == "drop":
            return X.dropna(subset=cols)
        for col in cols:
            X[col] = X[col].fillna(self.impute_values_[col])
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

class DTypeCorrector:
    """
    Convert columns in a DataFrame to specified data types.
    
    Parameters:
        dtype_map: Dict[str, str]
            Dictionary mapping column names to their target data types.
    """
    def __init__(self, dtype_map: Dict[str, Any]):
        self.dtype_map = dtype_map

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, dtype in self.dtype_map.items():
            X[col] = X[col].astype(dtype)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.transform(X)

class OutlierImputer:
    """
    Impute, clip, or drop outliers in numeric columns using IQR or Z-score.
    
    Parameters:
        method: str, default='iqr'
            Outlier detection method: 'iqr', 'zscore', or 'drop'.
        factor: float, default=1.5
            Multiplier for IQR or Z-score.
        strategy: str, default='clip'
            What to do with outliers: 'clip', 'impute', or 'drop'.
        columns: List[str], optional
            Columns to process. If None, all numeric columns are used.
    """
    def __init__(
        self, 
        method: str = "iqr",
        factor: float = 1.5,
        strategy: str = "clip",   # 'clip', 'impute', or 'drop'
        columns: Optional[List[str]] = None
    ):
        self.method = method
        self.factor = factor
        self.strategy = strategy
        self.columns = columns
        self.bounds_: Dict[str, tuple] = {}
        self.impute_values_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or X.select_dtypes(include=np.number).columns
        for col in cols:
            s = X[col]
            if self.method == "iqr":
                Q1 = s.quantile(0.25)
                Q3 = s.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.factor * IQR
                upper = Q3 + self.factor * IQR
            elif self.method == "zscore":
                mean = s.mean()
                std = s.std()
                lower = mean - self.factor * std
                upper = mean + self.factor * std
            else:
                raise ValueError(f"Unknown method: {self.method}")
            self.bounds_[col] = (lower, upper)
            if self.strategy == "impute":
                # Use median for imputation
                self.impute_values_[col] = s.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = self.columns or X.select_dtypes(include=np.number).columns
        if self.strategy == "drop":
            # Drop any row with outlier in specified columns
            mask = pd.Series([True] * len(X), index=X.index)
            for col in cols:
                lower, upper = self.bounds_[col]
                mask &= X[col].between(lower, upper)
            return X[mask]
        for col in cols:
            lower, upper = self.bounds_[col]
            if self.strategy == "clip":
                X[col] = X[col].clip(lower, upper)
            elif self.strategy == "impute":
                mask = ~X[col].between(lower, upper)
                X.loc[mask, col] = self.impute_values_[col]
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

class DataPipeline:
    """
    Chain together multiple data transformers with fit/transform interface.
    
    Parameters:
        steps: List of (str, transformer)
            Each step is a tuple of a name and a transformer object.
    """
    def __init__(self, steps: List[tuple]):
        self.steps = steps

    def fit(self, X: pd.DataFrame, y=None):
        for name, step in self.steps:
            X = step.fit(X, y).transform(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for name, step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        for name, step in self.steps:
            X = step.fit_transform(X, y)
        return X
