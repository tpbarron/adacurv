import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils import vector_to_parameters, parameters_to_vector

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    model = Net()

    params = list(model.parameters())
    params_old = []
    for i in range(len(params)):
        params_old.append(params[i] + torch.randn(params[i].shape) * 0.0001)
    x = Variable(torch.randn(10, 784)).float()

    temp_params = parameters_to_vector(model.parameters())
    vector_to_parameters(parameters_to_vector(params_old), model.parameters())
    out = model(x)
    f = torch.mean(out)
    actual_params = list(model.parameters())
    vector_to_parameters(temp_params, model.parameters())

    print ("Actual")
    grad = torch.autograd.grad(f, actual_params, create_graph=True)
    print ("Pre-set")
    grad = torch.autograd.grad(f, params_old, create_graph=True)
