if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets

    # Step 1 - Data prep
    n = 500
    k = 100
    mean = 0
    var = 1
    df_train = pd.read_table("crime-train.txt")
    df_test = pd.read_table("crime-test.txt")

    y = df_train["ViolentCrimesPerPop"]
    X = df_train.drop("ViolentCrimesPerPop", axis=1)

    # Step 2 - Initialize variables

    _lambda = 2 * np.max(np.abs(np.dot(y.T - np.mean(y), X)))
    _lambda_step = 2
    eta = 0.001
    convergence_delta = 0.01

    current_lambda = _lambda
    lambda_vals = [_lambda]

    FDR_list = []
    TPR_list = []
    nonzeros_list = []
    d = X.shape[1]
    W_all = np.zeros((d, 1))

    while _lambda > 0.01:

        print("Current lambda = ", current_lambda)

        (w_new, bias) = train(X, y, current_lambda, eta, convergence_delta)
        W_all = np.concatenate((W_all, np.expand_dims(w_new, axis=1)), axis=1)

        nonzeros = np.nonzero(W_all)[0]
        # incorrect_nonzeros = np.sum(W_all[nonzeros] == 0)
        # total_nonzeros = len(nonzeros)
        # FDR = incorrect_nonzeros / total_nonzeros

        # correct_nonzeros = np.sum(np.abs(W_all[nonzeros]) > 0)
        # TPR = correct_nonzeros / k

        nonzeros_list.append(nonzeros)
        # FDR_list.append(FDR)
        # TPR_list.append(TPR)

        current_lambda = current_lambda / _lambda_step
        lambda_vals.append(current_lambda)

    # raise NotImplementedError("Your Code Goes Here")

    # A6c Plots
    plt.figure(1)
    plt.semilogx(lambda_vals, nonzeros_list, 'r-')
    plt.xlabel('log(lambda)')
    plt.ylabel('Nonzero weights')
    plt.title('A6c: Nonzero weights versus Lambda')
    plt.show()

    # A6d Plots
    c1 = np.where(df_train.columns == "agePct12t29")[0] - 1
    c2 = np.where(df_train.columns == "pctWSocSec")[0] - 1
    c3 = np.where(df_train.columns == "pctUrban")[0] - 1
    c4 = np.where(df_train.columns == "agePct65up")[0] - 1
    c5 = np.where(df_train.columns == "householdsize")[0] - 1


    n_l = len(lambda_vals)

    plt.figure(2)
    plt.semilogx(lambda_vals, np.reshape(W_all[c1, :], (n_l, )), \
                lambda_vals, np.reshape(W_all[c2, :], (n_l,)), \
                lambda_vals, np.reshape(W_all[c3, :], (n_l,)), \
                lambda_vals, np.reshape(W_all[c4, :], (n_l,)), \
                lambda_vals, np.reshape(W_all[c5, :], (n_l,)))

    plt.xlabel('Lambda values')
    plt.ylabel('Column Values')
    plt.title('A6d: Regularization Paths(oneplot)')
    plt.legend(["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"])

if __name__ == "__main__":
    main()
