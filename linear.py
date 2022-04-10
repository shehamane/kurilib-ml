import random

import numpy as np
from data import add_ones_feature
from quality_functional import MSE


class AnalyticalSolution:
    w: np.ndarray

    def __init__(self):
        pass

    def fit(self, design_matrix: np.ndarray, target: np.ndarray):
        design_matrix = add_ones_feature(design_matrix)
        self.w = np.linalg.inv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(target)

    def predict(self, design_matrix: np.ndarray) -> np.ndarray:
        design_matrix = add_ones_feature(design_matrix)
        return design_matrix.dot(self.w)


class GradientDescend:
    

class StandardGradientDescend:
    alpha: float
    w: np.ndarray
    S: int
    tolerance: float
    regularization: str
    regularizer: float
    descend_method: str

    def __init__(self, alpha, S: int, tolerance: float = 1, regularization=None, regularizer=None,
                 descend_method: str = 'const'):
        self.descend_method = descend_method
        if descend_method == 'const' or descend_method == 'normalization_const':
            self.alpha = alpha
        if S <= 0 or tolerance <= 0:
            raise Exception('S or tolerance must be positive')
        self.S = S
        self.tolerance = tolerance
        self.regularization = regularization
        self.regularizer = regularizer

    def get_gradient(self, X: np.ndarray, y: np.ndarray):
        gradient = (2 * X.T.dot(X.dot(self.w) - y)) / X.shape[0]

        if self.descend_method == 'normalization_const':
            gradient /= np.linalg.norm(gradient)

        if self.regularization == 'L2':
            gradient += 2 * self.regularizer * self.w / X.shape[0]
        elif self.regularization == 'L1':
            gradient += self.regularizer * np.sign(self.w) / X.shape[0]

        return gradient

    def fit(self, design_matrix: np.ndarray, target: np.ndarray):
        design_matrix = add_ones_feature(design_matrix)

        i = 0
        self.w = np.zeros(design_matrix.shape[1])
        y_pred = design_matrix.dot(self.w)
        while i < self.S and MSE.get_loss(y_pred, target) > self.tolerance:
            self.w -= self.alpha * self.get_gradient(design_matrix, target)
            y_pred = design_matrix.dot(self.w)

            i += 1

    def predict(self, design_matrix: np.ndarray) -> np.ndarray:
        design_matrix = add_ones_feature(design_matrix)
        return design_matrix.dot(self.w)


class StohasticAverageGradient:
    w: np.ndarray
    grads: np.ndarray
    is_calculated: np.ndarray
    grads_sum: np.ndarray
    forgetting_rate: float
    alpha: float
    E: int
    moving_average: float
    tolerance: float

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
