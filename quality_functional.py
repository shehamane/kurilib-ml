from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @staticmethod
    @abstractmethod
    def get_single_loss(y_exp: np.ndarray, y_pred: np.ndarray):
        pass

    @staticmethod
    @abstractmethod
    def get_loss(y_exp: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(LossFunction):
    @staticmethod
    @abstractmethod
    def get_single_loss(y_exp: float, y_pred: float):
        return (y_exp - y_pred) ** 2

    @staticmethod
    def get_loss(y_exp: np.ndarray, y_pred: np.ndarray):
        size = len(y_exp)
        if size != len(y_pred):
            raise Exception('Incompatible sizes')
        return ((y_exp - y_pred) ** 2).sum() / size


class MAE(LossFunction):
    @staticmethod
    @abstractmethod
    def get_single_loss(y_exp: float, y_pred: float):
        return abs(y_exp - y_pred)

    @staticmethod
    def get_loss(y_exp: np.ndarray, y_pred: np.ndarray):
        size = len(y_exp)
        if size != len(y_pred):
            raise Exception('Incompatible sizes')
        return (abs(y_exp - y_pred)).sum() / size
