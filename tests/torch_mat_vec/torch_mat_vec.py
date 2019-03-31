import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(784, 10)
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x, return_z=False):
        # x = x.view(-1, 784)
        # x = self.fc1(x)
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if return_z:
            return F.log_softmax(x, dim=1), x
        return F.log_softmax(x, dim=1)

def Fvp(f, x, vector, damping=1e-4):
    vec = Variable(vector, requires_grad=False)
    grad_fo = torch.autograd.grad(f, x, create_graph=True)
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
    h = torch.sum(flat_grad * vec)
    hvp = torch.autograd.grad(h, x, create_graph=True, retain_graph=True)
    hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
    return hvp_flat + damping * vector

if __name__ == '__main__':
    
