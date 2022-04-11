import random
from abc import ABC, abstractmethod
import numpy as np
from data import add_ones_feature
from quality_functional import LossFunction, MSE


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


class GradientDescend(Model, ABC):
    def __init__(self):
        self.w = None
        self.reg_lmb = None
        self.regularization = None

    @abstractmethod
    def get_gradient(self, X, y):
        pass

    def regularize(self, N: int):
        if self.regularization == 'L2':
            self.w -= (2 * self.reg_lmb / N) * self.w
        elif self.regularization == 'L1':
            self.w -= (self.reg_lmb / N) * np.sign(self.w)


class StandardGradientDescend(GradientDescend):
    def __init__(self, alpha, S: int, tolerance: float = 1, regularization=None, reg_lmb=None,
                 descend_method: str = 'const', loss: LossFunction = MSE):
        self.w = None
        self.descend_method = descend_method
        if descend_method == 'const' or descend_method == 'normalization_const':
            self.alpha = alpha
        if S <= 0 or tolerance <= 0:
            raise Exception('S or tolerance must be positive')
        self.S = S
        self.tolerance = tolerance
        self.regularization = regularization
        self.reg_lmb = reg_lmb
        self.loss = loss

    def get_gradient(self, X: np.ndarray, y: np.ndarray):
        gradient = (2 * X.T.dot(X.dot(self.w) - y)) / X.shape[0]
        if self.descend_method == 'normalization_const':
            gradient /= np.linalg.norm(gradient)

        return gradient

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = add_ones_feature(X)

        i = 0
        self.w = np.zeros(X.shape[1])
        y_ = X.dot(self.w)
        while i < self.S and MSE.get_loss(y_, y) > self.tolerance:
            self.w -= self.alpha * self.get_gradient(X, y)
            self.regularize(X.shape[0])
            y_ = X.dot(self.w)
            i += 1

    def predict(self, design_matrix: np.ndarray) -> np.ndarray:
        design_matrix = add_ones_feature(design_matrix)
        return design_matrix.dot(self.w)


class StohasticAverageGradient:
    def __init__(self, alpha, forgetting_rate, E=1, tolerance=1):
        self.alpha = alpha
        self.forgetting_rate = forgetting_rate
        self.E = max(1, E)
        self.moving_average = 0
        self.tolerance = tolerance

    def calc_moving_average(self, new_error: float):
        if self.moving_average > 0:
            self.moving_average *= 1 - self.forgetting_rate
        self.moving_average += new_error * self.forgetting_rate

    def recalc_grad_sum(self, old_grad, new_grad):
        self.grads_sum += new_grad - old_grad

    def recalc_grad(self, X, y, idx):
        if not self.is_calculated[idx]:
            self.grads[idx] = (2 * X.T.dot(X.dot(self.w) - y)) / X.shape[0]
            self.recalc_grad_sum(0, self.grads[idx])
            self.is_calculated[idx] = 1
        else:
            old_grad = self.grads[idx]
            self.grads[idx] = (2 * X.T.dot(X.dot(self.w) - y)) / X.shape[0]
            self.recalc_grad_sum(old_grad, self.grads[idx])

    def fit(self, design_matrix: np.ndarray, target: np.ndarray):
        design_matrix = add_ones_feature(design_matrix)  # Добавляет признак-единичку

        self.grads = -np.ones(shape=design_matrix.shape)
        self.grads_sum = np.zeros(shape=design_matrix.shape[1])
        self.is_calculated = np.zeros(shape=design_matrix.shape[0])
        self.w = np.zeros(shape=design_matrix.shape[1])

        for e in range(0, self.E):
            for _ in range(0, design_matrix.shape[0]):
                i = random.randint(0, design_matrix.shape[0] - 1)
                self.recalc_grad(design_matrix[i], target[i], i)  # Пересчитать градиент
                self.w -= self.alpha * self.grads_sum / design_matrix.shape[0]  # Обновить веса

                err = MSE.get_single_loss(target[i], design_matrix[i].dot(self.w))
                self.calc_moving_average(err)  # Пересчитать скользящую среднюю

    def predict(self, design_matrix: np.ndarray):
        design_matrix = add_ones_feature(design_matrix)
        return design_matrix.dot(self.w)
