# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()

        v_a0 = 1 / math.sqrt(d)
        v_unif_dist0 = Uniform(torch.tensor([-v_a0]), torch.tensor([v_a0]))
        self.w0 = Parameter(v_unif_dist0.sample([h, d]).view(h, d))
        self.b0 = Parameter(v_unif_dist0.sample([1, h]).view(1, h))

        v_a1 = 1 / math.sqrt(h)
        v_unif_dist1 = Uniform(torch.tensor([-v_a1]), torch.tensor([v_a1]))
        self.w1 = Parameter(v_unif_dist1.sample([k, h]).view(k, h))
        self.b1 = Parameter(v_unif_dist1.sample([1, k]).view(1, k))

        # raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        v_val = relu(torch.mm(x, self.w0.T) + self.b0)
        v_val = torch.mm(v_val, self.w1.T) + self.b1
        return v_val
        # raise NotImplementedError("Your Code Goes Here")


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()

        v_a0 = 1 / math.sqrt(d)
        v_unif_dist0 = Uniform(torch.tensor([-v_a0]), torch.tensor([v_a0]))
        self.w0 = Parameter(v_unif_dist0.sample([h0, d]).view(h0, d))
        self.b0 = Parameter(v_unif_dist0.sample([1, h0]).view(1, h0))

        v_a1 = 1 / math.sqrt(h0)
        v_unif_dist1 = Uniform(torch.tensor([-v_a1]), torch.tensor([v_a1]))
        self.w1 = Parameter(v_unif_dist1.sample([h1, h0]).view(h1, h0))
        self.b1 = Parameter(v_unif_dist1.sample([1, h1]).view(1, h1))

        v_a2 = 1 / math.sqrt(h1)
        v_unif_dist2 = Uniform(torch.tensor([-v_a2]), torch.tensor([v_a2]))
        self.w2 = Parameter(v_unif_dist2.sample([k, h1]).view(k, h1))
        self.b2 = Parameter(v_unif_dist2.sample([1, k]).view(1, k))

        # raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        v_val = relu(torch.mm(x, self.w0.T) + self.b0)
        v_val = relu(torch.mm(v_val, self.w1.T) + self.b1)
        v_val = torch.mm(v_val, self.w2.T) + self.b2
        return v_val
        # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    device = 'cuda'
    model.train()
    model = model.to(device)

    v_train_loss = []
    v_train_err = []
    v_epoch_err = 100
    v_epoch = 0

    print('Train till we reach 99 percent acc')

    while v_epoch_err > 0.01:
        v_epoch_err = 0
        v_epoch_loss = 0
        n_train = 0
        print(v_epoch)
        v_epoch += 1

        for (x, y) in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            v_loss = cross_entropy(y_pred, y)
            optimizer.zero_grad()
            v_loss.backward()
            optimizer.step()

            v_diff = y_pred.max(dim=1)[1] - y
            v_epoch_loss += v_loss.data.cpu()
            v_epoch_err += torch.count_nonzero(v_diff)
            n_train += len(y)

        v_epoch_err = v_epoch_err.cpu() / n_train
        v_train_err.append(v_epoch_err)

        v_epoch_loss = v_epoch_loss.cpu() / n_train
        v_train_loss.append(v_epoch_loss)

        print(f'   error: {v_epoch_err}')
        print(f'   loss: {v_epoch_loss}')

    return v_train_loss

    # raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    data_train = TensorDataset(x, y)
    data_test = TensorDataset(x_test, y_test)

    trainloader = DataLoader(data_train, batch_size=32, shuffle=True)
    testloader = DataLoader(data_test, batch_size=32, shuffle=False)

    h = 64
    h0 = 32
    h1 = 32
    d = x.shape[1]
    k = 10
    # v_model = F1(h, d, k)
    v_model = F2(h0, h1, d, k)
    # def __init__(self, h0: int, h1: int, d: int, k: int):

    v_lr = 0.005
    v_device = 'cuda'
    v_optimizer = Adam(v_model.parameters(), lr=v_lr)
    v_train_loss = train(v_model, v_optimizer, trainloader)

    # calculating accuracy
    v_loss_test = 0
    v_n_correct = 0

    for (x, y) in testloader:
        y = y.to(v_device)
        y_pred = v_model(x.to(v_device))

        v_loss_test += cross_entropy(y_pred, y).item()
        v_n_correct += torch.sum(y_pred.argmax(dim=1) == y).item()

    print(f'Test Loss: {v_loss_test / len(testloader)}')
    print(f'Test Accuracy: {v_n_correct / len(y_test)}')

    # Generating the plots
    plt.plot(v_train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('F2: Train Plot')
    plt.legend(labels=['train'])
    plt.show()

    # number of parameters
    v_param = sum([x.numel() for x in v_model.parameters()])
    print(f'The number of parameters for model F2 are {v_param}')

    # raise NotImplementedError("Your Code Goes Here")


if __name__ == "__main__":
    main()
