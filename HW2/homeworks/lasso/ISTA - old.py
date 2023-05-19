from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


def random_data_generator(n, d, k, mean, variance) -> Tuple[np.ndarray, np.ndarray]:
    """Generates random data sets for X and y based on the dimensions and noise

    Args:
        n: number of rows of X
        d: number of features of X
        k: number of relevant features of X
        mean: mean of the noise term
        variance: variance of the noise term

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple with 2 entries. First represents input data X, second represents response y

    """
    w = np.zeros((d, 1))

    for j in range(k) :
        w[j] = (j + 1) / k

    X = np.random.normal(size=(n, d))
    noise = np.random.normal(scale=np.sqrt(variance), size=(n,))

    y = np.reshape(np.dot(w.T, X.T) + noise.T, (n,))

    return (X, y)


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.

    """
    weight = weight.reshape(len(weight), 1)
    weight_ = np.zeros(weight.shape)
    bias_ = bias - 2 * eta * np.sum(X.dot(weight) + bias - y)

    for i in range(weight.shape[1]):
        temp_weight = 2 * eta * (X.T[i] @ (X.dot(weight) + bias_ - y))
        weight_i = weight[i] - temp_weight[i]
        if weight_i < -2 * eta * _lambda:
            weight_i = weight_i + 2 * eta * _lambda
        elif weight_i > 2 * eta * _lambda:
            weight_i = weight_i - 2 * eta * _lambda
        else:
            weight_i = 0
        weight_[i] = weight_i

    return (weight_, bias_)

    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """

    return np.square(np.subtract(y, np.dot(X, weight))).sum() + _lambda * np.linalg.norm(weight, 1)

    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: Optional[np.ndarray] = None

    while not convergence_criterion(start_weight, old_w, start_bias, old_b, convergence_delta):
        old_w = start_weight.copy()
        old_b = start_bias
        start_weight, start_bias = step(X, y, start_weight, start_bias, _lambda, eta)

    return (start_weight, start_bias)

    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    if (weight is None or old_w is None):
        return False
    else:
        return (np.any(abs((np.subtract(weight, old_w))) > convergence_delta))
    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    n = 500
    d = 1000
    k = 100
    mean = 0
    var = 1
    # Step 1 - Create the random set of data
    X, y = random_data_generator(n, d, k, mean, var)

    # Step 2 - Standardize X and save values
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0))
    fit_mean = np.mean(X, axis=0)
    fit_std = np.std(X, axis=0)

    # Step 3 - Calculate lambda_max, the initial step
    _lambda = 2 * np.max(np.abs(np.dot(y.T - np.mean(y), X_std)))
    _lambda_step = 2
    eta = 0.01
    convergence_delta = 0.01

    # Step 4 - Solve multiple lasso problems using decreasing lambda
    _lambda_values = [_lambda]
    W_all = np.zeros((X.shape[1], 1))

    while np.count_nonzero(W_all[:, -1]) != k:
        (w_new, bias) = train(X_std, y, _lambda, eta, convergence_delta)
        W_all = np.concatenate((W_all, np.expand_dims(w_new, axis=1)), axis=1)
        _lambda = _lambda / _lambda_step
        _lambda_values.append(_lambda)

    # Step 5 - Plot the relationship between the lambda values and sparsity of weight vector
    plt.figure(1)
    plt.semilogx(_lambda_values, np.count_nonzero(W_all, axis=0), 'r-')
    plt.xlabel('Lambda')
    plt.ylabel('Nonzero Coefficients in w')
    plt.title('A5a: Nonzero weights versus Lambda')
    plt.show()

    # raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
