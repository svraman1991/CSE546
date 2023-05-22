from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """

    K = (np.outer(x_i, x_j) + 1) ** d
    return K
    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    sq_diff = np.square(np.subtract.outer(x_i, x_j))
    K = np.exp(-gamma * sq_diff)
    return K
    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    n = len(x)
    K = kernel_function(x, x, kernel_param)
    alpha_hat = np.linalg.solve(K + _lambda * np.eye(n), y)
    return alpha_hat
    # raise NotImplementedError("Your Code Goes Here")


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


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    loss_sum = 0.0

    for i in range(num_folds):
        start = i * fold_size
        end = start + fold_size

        x_train = np.concatenate((x[:start], x[end:]))
        y_train = np.concatenate((y[:start], y[end:]))
        x_val = x[start:end]
        y_val = y[start:end]

        alpha_hat = train(x_train, y_train, kernel_function, kernel_param, _lambda)

        y_pred = np.dot(kernel_function(x_val, x_train, kernel_param), alpha_hat)

        # mse = np.mean((y_val - y_pred) ** 2)
        mse = mean_squared_error(y_val, y_pred)
        loss_sum += mse

    avg_loss = loss_sum / num_folds
    return avg_loss
    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    opt_loss = float('inf')
    opt_lambda = None
    opt_gamma = None

    # Implementing Grid search

    lambdas = [10 ** i for i in np.linspace(-5, -1, num=100)]
    for _lambda in lambdas:
        gamma = 1 / np.median(np.square(np.subtract.outer(x, x)))
        loss = cross_validation(x, y, rbf_kernel, gamma, _lambda, num_folds)
        if loss < opt_loss:
            opt_loss = loss
            opt_lambda = _lambda
            opt_gamma = gamma

    return opt_lambda, opt_gamma

    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You can use gamma = 1 / median((x_i - x_j)^2) for all unique pairs x_i, x_j in x) for this problem.
          However, if you would like to search over other possible values of gamma, you are welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """
    opt_loss = float('inf')
    opt_lambda = None
    opt_d = None

    # Implementing Grid search

    lambdas = [10 ** i for i in np.linspace(-5, -1, num=100)]
    d_values = range(5, 25)

    for _lambda in lambdas:
        for d in d_values:
            loss = cross_validation(x, y, poly_kernel, d, _lambda, num_folds)
            if loss < opt_loss:
                opt_loss = loss
                opt_lambda = _lambda
                opt_d = d

    return opt_lambda, opt_d
    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")

    # print(f'x_30 is {x_30} and \n , y_30 is {y_30}')
    rbf_lambda, rbf_gamma = rbf_param_search(x_30, y_30, num_folds=30)
    poly_lambda, poly_d = poly_param_search(x_30, y_30, num_folds=30)

    print('Optimal Parameters: \n')

    print(f'RBF Lambda:{rbf_lambda}  and RBF Gamma:{rbf_gamma} \n')
    print(f'Poly Lambda:{poly_lambda} and Poly d:{poly_d} \n')

    fine_grid = np.linspace(0, 1, num=100)

    # RBF Plots
    rbf_pred = rbf_kernel(x_30, fine_grid[:, np.newaxis], rbf_gamma)
    rbf_alpha = train(x_30, y_30, rbf_kernel, rbf_gamma, rbf_lambda)
    rbf_pred = np.squeeze(rbf_pred)
    rbf_pred = np.dot(rbf_alpha, rbf_pred)

    # True function plot
    plt.plot(x_30, y_30, 'mx', label='True Data')
    plt.plot(fine_grid, f_true(fine_grid), 'm--', label='True Function')
    # RBF Plots
    plt.plot(fine_grid, rbf_pred, label='RBF Kernel')
    plt.title('Plot for RBF Kernel and True function')
    plt.ylim(-6, 14)
    plt.legend()
    plt.show()

    # Poly Plots
    poly_pred = poly_kernel(x_30, fine_grid[:, np.newaxis], poly_d)
    poly_alpha = train(x_30, y_30, poly_kernel, poly_d, poly_lambda)
    poly_pred = np.squeeze(poly_pred)
    poly_pred = np.dot(poly_alpha, poly_pred)

    # True function plot
    plt.plot(x_30, y_30, 'mx', label='True Data')
    plt.plot(fine_grid, f_true(fine_grid), 'm--', label='True Function')
    # Poly Plots
    plt.plot(fine_grid, poly_pred, label='Poly Kernel')
    plt.title('Plot for Poly Kernel and True function')
    plt.ylim(-6, 14)
    plt.legend()
    plt.show()

    # raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
