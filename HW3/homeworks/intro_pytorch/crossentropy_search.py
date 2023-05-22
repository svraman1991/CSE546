if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
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
        self.softmax = SoftmaxLayer()

    def forward(self, x):
        return self.softmax(self.layer(x))


# Sigmoid
class sigmoid(nn.Module):

    def __init__(self):
        super(sigmoid, self).__init__()
        self.layer1 = LinearLayer(2, 2, generator=RNG)
        self.sigma = SigmoidLayer()
        self.layer2 = LinearLayer(2, 2, generator=RNG)
        self.softmax = SoftmaxLayer()

    def forward(self, x):
        return self.softmax(self.layer2(self.sigma(self.layer1(x))))


# RELU
class relu(nn.Module):

    def __init__(self):
        super(relu, self).__init__()
        self.layer1 = LinearLayer(2, 2, generator=RNG)
        self.relu = ReLULayer()
        self.layer2 = LinearLayer(2, 2, generator=RNG)
        self.softmax = SoftmaxLayer()

    def forward(self, x):
        return self.softmax(self.layer2(self.relu(self.layer1(x))))


# Sigmoid and RELU
class sigmarelu(nn.Module):

    def __init__(self):
        super(sigmarelu, self).__init__()
        self.layer1 = LinearLayer(2, 2, generator=RNG)
        self.sigma = SigmoidLayer()
        self.layer2 = LinearLayer(2, 2, generator=RNG)
        self.relu = ReLULayer()
        self.layer3 = LinearLayer(2, 2, generator=RNG)
        self.softmax = SoftmaxLayer()

    def forward(self, x):
        return self.softmax(self.layer3(self.relu(self.layer2(self.sigma(self.layer1(x))))))


# RELU and Sigmoid
class relusigma(nn.Module):

    def __init__(self):
        super(relusigma, self).__init__()
        self.layer1 = LinearLayer(2, 2, generator=RNG)
        self.sigma = SigmoidLayer()
        self.layer2 = LinearLayer(2, 2, generator=RNG)
        self.relu = ReLULayer()
        self.layer3 = LinearLayer(2, 2, generator=RNG)
        self.softmax = SoftmaxLayer()

    def forward(self, x):
        return self.softmax(self.layer3(self.sigma(self.layer2(self.relu(self.layer1(x))))))


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

    Notes:
        - Try using learning rate between 1e-5 and 1e-3.
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.
        - When searching over batch_size using powers of 2 (starting at around 32) is typically a good heuristic.
            Make sure it is not too big as you can end up with standard (or almost) gradient descent!

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

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
        v_optimizer = SGDOptimizer(v_model.parameters(), lr=1e-2)
        v_train_results = train(dataset_train, v_model, CrossEntropyLossLayer(), v_optimizer, dataset_val, epochs=150)
        v_train_history[v_model_name] = {
            "train": v_train_results["train"],
            "val": v_train_results["val"],
            "model": v_model
        }
    return v_train_history
    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """

    is_accurate = 0
    all = 0

    with torch.no_grad():
        for (x, y) in dataloader:
            is_accurate += torch.sum(model(x).argmax(dim=1) == y).item()
            all += y.shape[0]

    v_accuracy = is_accurate / all
    return v_accuracy
    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y)), generator=RNG, batch_size=128)
    dataset_val = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val)), batch_size=128)
    dataset_test = DataLoader(TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test)), batch_size=128)

    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)

    # Plotting the results
    v_colors = ['k', 'r', 'g', 'b', 'm', 'k', 'r', 'g', 'b', 'm']
    v_labels = []
    v_count = 0
    v_values = {}

    for key, value in ce_configs.items():
        plt.plot(value['train'], v_colors[v_count])
        plt.plot(value['val'], v_colors[v_count + 1])
        v_labels.append(key + ' training loss')
        v_labels.append(key + ' validation loss')
        v_count += 2
        v_values[key] = value['val']

    plt.legend(v_labels)
    plt.xlabel('epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Cross Entropy Training and Validation loss by model')
    plt.savefig('CEL.png')
    plt.show()

    # Finding the best model from above
    v_values = {v_name: min(v_val) for v_name, v_val in v_values.items()}
    v_sorted_values = sorted(v_values.items(), key=lambda x : x[1])
    v_best_model, v_value = v_sorted_values[0]

    print(f'The best model is {v_best_model} with loss value of {v_value}')

    plot_model_guesses(
        dataset_test,
        model=ce_configs[v_best_model]["model"],
        title='Cross Entropy: {}'.format(v_best_model)
    )

    v_acc_score = accuracy_score(ce_configs[v_best_model]["model"], dataset_test)
    print(f'The best model is {v_best_model} with an accuracy of {v_acc_score}')

    # raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
