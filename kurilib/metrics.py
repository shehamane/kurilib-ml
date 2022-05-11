from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    @staticmethod
    @abstractmethod
    def get_distance(x: np.ndarray, y: np.ndarray):
        pass


class EuclidianNorm(Metric):
    @staticmethod
    def get_distance(x: np.ndarray, y: np.ndarray):
        return np.sqrt(np.sum(np.power(x - y, 2)))


class ManhattanNorm(Metric):
    @staticmethod
    def get_distance(x: np.ndarray, y: np.ndarray):
        return np.sum(np.abs(x - y))


class CosineDistance(Metric):
    @staticmethod
    def get_distance(x: np.ndarray, y: np.ndarray):
        return 1 - (x.dot(y))/(np.linalg.norm(x)*np.linalg.norm(y))
