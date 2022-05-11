from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass
