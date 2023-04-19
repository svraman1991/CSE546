"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        self.theta = None
        # raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        n, d1 = X.shape
        if degree < 1:
            pfX_ = np.zeros((1, 1))
        else:
            if d1 != 1:
                pfX_ = np.zeros((1, 1))
            else:
                pfX_ = np.zeros((n, degree))
                for i in range(n):
                    for j in range(degree):
                        pfX_[i, j] = X[i] ** (j + 1)
        return pfX_

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        # 1. applying polynomial expansion to the input data
        #print(f'shape of input data is {X.shape}')
        X_ = self.polyfeatures(X, self.degree)

        # 2. Standardizing the expanded matrix
        X_std = (X_ - np.mean(X_, axis=0)) / (np.std(X_, axis=0))

        # 3. Adding the bias term
        n1, d1 = X_std.shape
        newdata = np.zeros((n1, d1 + 1))
        newdata[:, 0] = 1

        for i in range(n1):
            for j in range(d1):
                newdata[i, j + 1] = X_std[i, j]

        # 4. Solving for the coefficients
        reg_matrix = self.reg_lambda * np.eye(d1 + 1)
        reg_matrix[0, 0] = 0
        #print(f'shape of newdata is is {newdata.shape}')

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.solve(newdata.T @ newdata + reg_matrix, newdata.T @ y)
        #print(self.theta.shape)
        #print(f'theta values are {self.theta}')

        # raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        #print(f'shape of test data is {X.shape}')
        #n = len(X)
        X_ = self.polyfeatures(X, self.degree)
        X_std = (X_ - np.mean(X_, axis=0)) / (np.std(X_, axis=0))
        # X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0))

        # add 1s column
        # X_ = np.c_[np.ones([n, 1]), X_std]
        n1, d1 = X_std.shape
        newdata = np.zeros((n1, d1 + 1))
        newdata[:, 0] = 1

        for i in range(n1):
            for j in range(d1):
                newdata[i, j + 1] = X_std[i, j]

        # predict
        return newdata.dot(self.theta)
        # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    if a.shape != b.shape:
        return -1
    else:
        return np.square(np.subtract(a, b)).mean()


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    # raise NotImplementedError("Your Code Goes Here")
