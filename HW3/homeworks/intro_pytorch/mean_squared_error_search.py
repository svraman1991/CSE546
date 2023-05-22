if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)

# Code for the various methods that will be used


# Linear Regression
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.layer = LinearLayer(2, 2, generator=RNG)

    def forward(self, x):
        return self.layer(x)


# Sigmoid
class sigmoid(nn.Module):

    def __init__(self):
        super(sigmoid, self).__init__()
        self.layer1 = LinearLayer(2, 2, generator=RNG)
        self.sigma = SigmoidLayer()
        self.layer2 = LinearLayer(2, 2, generator=RNG)

    def forward(self, x):
        return self.layer2(self.sigma(self.layer1(x)))


# RELU
class relu(nn.Module):

    def __init__(self):
        super(relu, self).__init__()
        self.layer1 = LinearLayer(2, 2, generator=RNG)
        self.relu = ReLULayer()
        self.layer2 = LinearLayer(2, 2, generator=RNG)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


# Sigmoid and RELU
class sigmarelu(nn.Module):

    def __init__(self):
        super(sigmarelu, self).__init__()
        self.layer1 = LinearLayer(2, 2, generator=RNG)
        self.sigma = SigmoidLayer()
        self.layer2 = LinearLayer(2, 2, generator=RNG)
        self.relu = ReLULayer()
        self.layer3 = LinearLayer(2, 2, generator=RNG)

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.sigma(self.layer1(x)))))


# RELU and Sigmoid
class relusigma(nn.Module):

    def __init__(self):
        super(relusigma, self).__init__()
        self.layer1 = LinearLayer(2, 2, generator=RNG)
        self.sigma = SigmoidLayer()
        self.layer2 = LinearLayer(2, 2, generator=RNG)
        self.relu = ReLULayer()
        self.layer3 = LinearLayer(2, 2, generator=RNG)

    def forward(self, x):
        return self.layer3(self.sigma(self.layer2(self.relu(self.layer1(x)))))


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    is_accurate = 0
    all = 0

    with torch.no_grad():
        for (x, y) in dataloader:
            is_accurate += torch.sum(model(x).argmax(dim=1) == y.argmax(dim=1)).item()
            all += y.shape[0]

    v_accuracy = is_accurate / all
    return v_accuracy
    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Notes:
        - Try using learning rate between 1e-5 and 1e-3.
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.
        - When searching over batch_size using powers of 2 (starting at around 32) is typically a good heuristic.
            Make sure it is not too big as you can end up with standard (or almost) gradient descent!

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    v_models = {
        "LinearRegression": LinearRegression(),
        "Sigmoid": sigmoid(),
        "ReLU": relu(),
        "Sigmoid_ReLU": sigmarelu(),
        "ReLU_Sigmoid": relusigma()
    }

    v_train_history = dict()

    for v_model_name, v_model in v_models.items():
        print(v_model_name)
        v_optimizer = SGDOptimizer(v_model.parameters(), lr=1e-3)
        v_train_results = train(dataset_train, v_model, MSELossLayer(), v_optimizer, dataset_val, epochs=75)
        v_train_history[v_model_name] = {
            "train": v_train_results["train"],
            "val": v_train_results["val"],
            "model": v_model
        }
    return v_train_history

    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)).float()), generator=RNG, batch_size=8)
    dataset_val = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val)).float()), batch_size=8)
    dataset_test = DataLoader(TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test)).float()), batch_size=8)

    mse_configs = mse_parameter_search(dataset_train, dataset_val)

    # Plotting the results
    v_colors = ['k', 'r', 'g', 'b', 'm', 'k', 'r', 'g', 'b', 'm']
    v_labels = []
    v_count = 0
    v_values = {}

    for key, value in mse_configs.items():
        plt.plot(value['train'], v_colors[v_count])
        plt.plot(value['val'], v_colors[v_count + 1])
        v_labels.append(key + ' training loss')
        v_labels.append(key + ' validation loss')
        v_count += 2
        v_values[key] = value['val']

    plt.legend(v_labels)
    plt.xlabel('epoch')
    plt.ylabel('MSE Loss')
    plt.title('MSE Training and Validation loss by model')
    plt.savefig('MSE.png')
    plt.show()

    # Finding the best model from above
    v_values = {v_name: min(v_val) for v_name, v_val in v_values.items()}
    v_sorted_values = sorted(v_values.items(), key=lambda x : x[1])
    v_best_model, v_value = v_sorted_values[0]

    print(f'The best model is {v_best_model} with loss value of {v_value}')

    # print(f'MSEconfig= {mse_configs[v_best_model]["model"]}')
    # print(f'dataset_test= {dataset_test}')

    plot_model_guesses(
        dataset_test,
        model=mse_configs[v_best_model]["model"],
        title='MSE: {}'.format(v_best_model)
    )

    v_acc_score = accuracy_score(mse_configs[v_best_model]["model"], dataset_test)
    print(f'The best model is {v_best_model} with an accuracy of {v_acc_score}')

    # raise NotImplementedError("Your Code Goes Here")


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
