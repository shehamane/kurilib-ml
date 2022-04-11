from abc import ABC, abstractmethod
import numpy as np
from data import add_ones_feature
from quality_functional import LossFunction, MSE, LogisticLoss, sigmoid, MAE


class Model(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass


class AnalyticalSolution(Model):
    w: np.ndarray

    def __init__(self):
        pass

    def fit(self, design_matrix: np.ndarray, target: np.ndarray):
        design_matrix = add_ones_feature(design_matrix)
        self.w = np.linalg.inv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(target)

    def predict(self, design_matrix: np.ndarray) -> np.ndarray:
        design_matrix = add_ones_feature(design_matrix)
        return design_matrix.dot(self.w)


class GradientDescent(Model, ABC):
    def __init__(self, alpha, tolerance,
                 regularization, reg_lmb, descent_method, loss):
        self.w = None
        self._alpha = alpha
        self._tolerance = tolerance
        self.__reg_lmb = reg_lmb
        self.__regularization = regularization
        self.__descent_method = descent_method
        self.__loss = loss

    def _get_gradient(self, X, y):
        if self.__loss == MSE:
            gradient = (2 / X.shape[0]) * X.T.dot(X.dot(self.w) - y)
        elif self.__loss == LogisticLoss:
            gradient = -(X * (y - sigmoid(X.dot(self.w)))).sum(axis=0)
        elif self.__loss == MAE:
            gradient = np.sign(X.dot(self.w) - y)

        if 'normalization' in self.__descent_method:
            gradient /= np.linalg.norm(gradient)
        return gradient

    def regularize(self, N: int):
        if self.__regularization == 'L2':
            self.w -= (2 * self.__reg_lmb / N) * self.w
        elif self.__regularization == 'L1':
            self.w -= (self.__reg_lmb / N) * np.sign(self.w)

    def predict(self, X: np.ndarray):
        X = add_ones_feature(X)
        if self.__loss == LogisticLoss:
            return sigmoid(X.dot(self.w))
        return X.dot(self.w)


class StandardGradientDescent(GradientDescent):
    def __init__(self, alpha, S: int, tolerance: float = 1, regularization=None, reg_lmb=None,
                 descent_method: str = 'const', loss: LossFunction = MSE):
        super().__init__(alpha, tolerance, regularization, reg_lmb, descent_method, loss)
        self.__S = S

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = add_ones_feature(X)

        i = 0
        self.w = np.zeros(X.shape[1])
        y_ = X.dot(self.w)
        while i < self.__S and MSE.get_loss(y_, y) > self._tolerance:
            self.w -= self._alpha * self._get_gradient(X, y)
            self.regularize(X.shape[0])
            y_ = X.dot(self.w)
            i += 1


class StochasticGradientDescent(GradientDescent):
    def __init__(self, alpha: float, eras: int, batch_size: int, tolerance: float, loss: LossFunction,
                 descent_method: str = 'const', regularization: str = None, reg_lmb: float = None):
        super().__init__(alpha, tolerance, regularization, reg_lmb, descent_method, loss)
        self.__eras = eras
        self.__batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = add_ones_feature(X)
        self.w = np.zeros(X.shape[1])

        for era in range(0, self.__eras):
            i = self.__batch_size
            while i <= X.shape[0]:
                X_batch = X[i - self.__batch_size: i]
                y_batch = y[i - self.__batch_size: i]
                self.w -= self._alpha * self._get_gradient(X_batch, y_batch)
                self.regularize(X.shape[0])

                if MSE.get_loss(X_batch.dot(self.w), y_batch) < self._tolerance:
                    return
                i += self.__batch_size

            X_batch = X[i - self.__batch_size: X.shape[0]]
            y_batch = y[i - self.__batch_size: X.shape[0]]
            self.w -= self._alpha * self._get_gradient(X_batch, y_batch)
            self.regularize(X.shape[0])

            if MSE.get_loss(X_batch.dot(self.w), y_batch) < self._tolerance:
                return
