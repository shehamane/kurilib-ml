import random
from abc import ABC, abstractmethod
import numpy as np
from data import add_ones_feature
from quality_functional import LossFunction, MSE, MAE
from copy import deepcopy


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


class StandardGradientDescent(GradientDescent):
    def __init__(self, alpha, S: int, tolerance: float = 1, regularization=None, reg_lmb=None,
                 descent_method: str = 'const', loss: LossFunction = MSE):
        self.w = None
        self.Descent_method = descent_method
        if descent_method == 'const' or descent_method == 'normalization_const':
            self.alpha = alpha
        if S <= 0 or tolerance <= 0:
            raise Exception('S and tolerance must be positive')
        self.S = S
        self.tolerance = tolerance
        self.regularization = regularization
        self.reg_lmb = reg_lmb
        self.loss = loss

    def get_gradient(self, X: np.ndarray, y: np.ndarray):
        gradient = (2 * X.T.dot(X.dot(self.w) - y)) / X.shape[0]
        if self.Descent_method == 'normalization_const':
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


# class StochasticAverageGradientDescent(GradientDescent):
#     def __init__(self, alpha, E: int, tolerance: float = 1, forgetting_rate: float = 0.5, regularization=None,
#                  reg_lmb=None,
#                  descent_method: str = 'const', loss: LossFunction = MSE):
#         self.grads = None
#         self.descent_method = descent_method
#         self.alpha = alpha
#         self.grads_sum = None
#         self.forgetting_rate = forgetting_rate
#         self.moving_average = 0
#         self.w = None
#         self.Descent_method = descent_method
#         if descent_method == 'const' or descent_method == 'normalization_const':
#             self.alpha = alpha
#         if E <= 0 or tolerance <= 0:
#             raise Exception('E and tolerance must be positive')
#         self.E = E
#         self.tolerance = tolerance
#         self.regularization = regularization
#         self.reg_lmb = reg_lmb
#         self.loss = loss
#
#     def get_gradient(self, X, y):
#         gradient = (2 * X.T.dot(X.dot(self.w) - y)) / X.shape[0]
#
#         return gradient
#
#     def calc_grads(self, X, y):
#         self.grads = np.empty(X.shape)
#         for i in range(0, X.shape[0]):
#             self.grads[i] = self.get_gradient(X[i], y[i])
#             self.grads_sum += self.grads[i]
#
#     def calc_moving_average(self, new_error: float):
#         if self.moving_average > 0:
#             self.moving_average *= 1 - self.forgetting_rate
#         self.moving_average += new_error * self.forgetting_rate
#
#     def recalc_grad_sum(self, old_grad, new_grad):
#         self.grads_sum -= old_grad - new_grad
#
#     def recalc_grad(self, X, y, idx):
#         old_grad = deepcopy(self.grads[idx])
#         self.grads[idx] = self.get_gradient(X, y)
#         self.recalc_grad_sum(old_grad, self.grads[idx])
#
#     def shuffle(self, X, y) -> (np.ndarray, np.ndarray):
#         M = np.c_[X, y]
#         np.random.shuffle(M)
#         return M[:, :-1], M[:, -1]
#
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         X = add_ones_feature(X)
#
#         self.w = np.zeros(shape=X.shape[1])
#         self.grads_sum = np.zeros(shape=X.shape[1])
#         self.calc_grads(X, y)
#
#         for e in range(0, self.E):
#             X, y = self.shuffle(X, y)
#             for i in range(0, X.shape[0]):
#                 self.recalc_grad(X[i], y[i], i)
#
#                 if self.descent_method == 'normalization const':
#                     self.w -= (self.alpha / X.shape[0]) * self.grads_sum / np.linalg.norm(self.grads_sum)
#                 else:
#                     self.w -= (self.alpha / X.shape[0]) * self.grads_sum
#
#                 err = MSE.get_single_loss(y[i], X[i].dot(self.w))
#                 self.calc_moving_average(err)
#
#     def predict(self, design_matrix: np.ndarray):
#         design_matrix = add_ones_feature(design_matrix)
#         return design_matrix.dot(self.w)


