from abc import ABC, abstractmethod

import numpy as np


class Norm(ABC):
    @staticmethod
    @abstractmethod
    def get_distance(x: np.ndarray, y: np.ndarray):
        pass


class EuclidianNorm(Norm):
    @staticmethod
    def get_distance(x: np.ndarray, y: np.ndarray):
        return np.sqrt(np.sum(np.power(x - y, 2)))
