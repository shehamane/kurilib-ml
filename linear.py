import copy
from abc import ABC, abstractmethod
import numpy as np
from data import add_ones_feature, allocate_positive_class
from quality_functional import LossFunction, MSE, LogisticLoss, sigmoid, MAE
import pandas as pd


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
    def __init__(self, alpha: float, tolerance: float,
                 regularization: str, reg_lmb: float, descent_method: str, loss: LossFunction):
        self.w = None
        if alpha > 0:
            self._alpha = alpha
        else:
            raise Exception('Alpha must be positive')
        if tolerance is None or tolerance > 0:
            self._tolerance = tolerance
        else:
            raise Exception('Tolerance must be positive')
        self.__reg_lmb = reg_lmb
        if regularization in ['L1', 'L2', None]:
            self.__regularization = regularization
        else:
            raise Exception('No such regularization method')
        if descent_method in ['const', 'normalization const']:
            self.__descent_method = descent_method
        else:
            raise Exception('No such descent method')
        self._loss = loss

    def _get_gradient(self, X, y):
        if self._loss == MSE:
            gradient = (2 / X.shape[0]) * X.T.dot(X.dot(self.w) - y)
        elif self._loss == LogisticLoss:
            gradient = -(X.T.dot(y - sigmoid(X.dot(self.w))))
        elif self._loss == MAE:
            gradient = np.sign(X.dot(self.w) - y)
        else:
            raise Exception('Gradient cannot be found for this loss-function')

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
        if self._loss == LogisticLoss:
            return sigmoid(X.dot(self.w))
        return X.dot(self.w)


class StandardGradientDescent(GradientDescent):
    def __init__(self, alpha, S: int, tolerance: float = 1, regularization=None, reg_lmb=None,
                 descent_method: str = 'const', loss: LossFunction = MSE):
        super().__init__(alpha, tolerance, regularization, reg_lmb, descent_method, loss)
        if S > 0:
            self.__S = S
        else:
            raise Exception('S must be positive')

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = add_ones_feature(X)

        i = 0
        self.w = np.zeros(X.shape[1])
        y_pred = X.dot(self.w)
        while i < self.__S and (self._loss == LogisticLoss or self._loss.get_loss(y_pred, y) > self._tolerance):
            self.w -= self._alpha * self._get_gradient(X, y)
            self.regularize(X.shape[0])
            y_pred = X.dot(self.w)
            i += 1


class StochasticGradientDescent(GradientDescent):
    def __init__(self, alpha: float, eras: int, batch_size: int, tolerance: float = 1, loss: LossFunction = MSE,
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


class OneVsAllClassifier(Model):
    def __init__(self, base_classifier):
        if type(base_classifier) not in [StandardGradientDescent, StochasticGradientDescent]:
            raise Exception('This classifier is not supported')
        self.__base_classifier = base_classifier
        self.__classifiers = None
        self.__classes = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if y.dtype != 'category':
            raise Exception('Target must be categorical')
        self.__classes = y.unique()
        self.__classifiers = [copy.deepcopy(self.__base_classifier) for i in range(0, self.__classes.size)]
        for idx, cls in enumerate(self.__classes):
            target = allocate_positive_class(y, cls)
            self.__classifiers[idx].fit(X.to_numpy(), target.to_numpy())

    def predict(self, X: np.ndarray):
        probas = np.zeros(shape=(self.__classes.size, X.shape[0]))
        for idx, classifier in enumerate(self.__classifiers):
            probas[idx] = classifier.predict(X)
        return [self.__classes[idx] for idx in np.argmax(probas, axis=0)]
