from __future__ import print_function
import argparse
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR
from fisher.optim.hvp_utils import build_Fvp, mean_kl_multinomial

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4*4*50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         print (x.shape)
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(784, 10)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x = x.view(-1, 784)
        # x = self.fc1(x)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def log_stats(accuracies, losses, times, args, model, device, test_loader, epoch, batch_idx):
    acc, loss = test(args, model, device, test_loader)
    accuracies.append(acc)
    losses.append(loss)

    dir = build_log_dir(args)
    #plt.gca().clear()
    #plt.plot(accuracies)
    #plt.savefig(dir+"/plot.pdf")
    np.save(dir+"/times.npy", np.array(times))
    np.save(dir+"/data.npy", np.array(accuracies))
    np.save(dir+"/losses.npy", np.array(losses))

    torch.save(model.state_dict(), dir+"/model_epoch"+str(epoch)+"_batch"+str(batch_idx)+"_acc"+str(acc)+".pth")


def train(args, model, device, train_loader, test_loader, optimizer, epoch, datas):
    model.train()
    accuracies, losses, times = datas
    acc, loss = test(args, model, device, test_loader)
    accuracies.append(acc)
    losses.append(loss)

    for batch_idx, (data, target) in enumerate(train_loader):
        stime = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if args.optim not in ["sgd", "adam", "rmsprop", "amsgrad", "adagrad"]:
            Fvp_fn = build_Fvp(model, data, target, mean_kl_multinomial)
            optimizer.step(Fvp_fn)
        else:
            optimizer.step()

        etime = time.time()
        step_time = etime - stime
        times.append(times[-1]+step_time)

        if batch_idx % args.log_interval == 0:
            log_stats(accuracies, losses, times, args, model, device, test_loader, epoch, batch_idx)

        if args.verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # return accuracies

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print ("target, output: ", data.shape, target.shape, output.shape)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if args.verbose:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return accuracy, test_loss

def build_log_dir(args):
    dir = os.path.join(args.log_dir, args.optim)
    if args.approx_adaptive:
        dir = os.path.join(dir, "approx_adaptive")
    else:
        dir = os.path.join(dir, "optim_adaptive")

    if args.shrunk:
        dir = os.path.join(dir, "shrunk_true")
        dir = os.path.join(dir, "lanczos_iters_"+str(args.lanczos_iters))
    else:
        dir = os.path.join(dir, "shrunk_false")

    dir = os.path.join(dir, "batch_size_"+str(args.batch_size))
    dir = os.path.join(dir, "lr_"+str(args.lr))

    if args.optim in ["natural_adam", "natural_amsgrad"]:
        dir = os.path.join(dir, "betas"+str(args.beta1)+"_"+str(args.beta2))

    dir = os.path.join(dir, str(args.seed))
    return dir


def launch_job(args):
    if args.batch_size == 1000:
        args.log_interval = 6
    elif args.batch_size == 500:
        args.log_interval = 12
    elif args.batch_size == 250:
        args.log_interval = 24
    elif args.batch_size == 125:
        args.log_interval = 48
    dir = build_log_dir(args)
    try:
        os.makedirs(dir)
    except:
        pass
    with open(os.path.join(dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./svhn_data', split="train", download=True,
                              transform=transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4435,), (0.1970,))
                              ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('./svhn_data', split="test", download=True,
                              transform=transforms.Compose([
                                   transforms.Grayscale(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4435,), (0.1970,))
                              ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "amsgrad":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    else:
        import fisher.optim as fisher_optim
        if args.optim == 'ngd':
            optimizer = fisher_optim.NGD(model.parameters(),
                                         lr=args.lr,
                                         shrunk=args.shrunk,
                                         lanczos_iters=args.lanczos_iters,
                                         batch_size=args.batch_size)
        elif args.optim == 'natural_adam':
            optimizer = fisher_optim.NaturalAdam(model.parameters(),
                                                     lr=args.lr,
                                                     shrunk=args.shrunk,
                                                     lanczos_iters=args.lanczos_iters,
                                                     batch_size=args.batch_size)
        elif args.optim == 'natural_amsgrad':
            optimizer = fisher_optim.NaturalAmsgrad(model.parameters(),
                                                        lr=args.lr,
                                                        shrunk=args.shrunk,
                                                        lanczos_iters=args.lanczos_iters,
                                                        batch_size=args.batch_size)
        elif args.optim == 'natural_adagrad':
            optimizer = fisher_optim.NaturalAdagrad(model.parameters(),
                                                        lr=args.lr,
                                                        shrunk=args.shrunk,
                                                        lanczos_iters=args.lanczos_iters,
                                                        batch_size=args.batch_size)
        else:
            raise NotImplementedError

    accuracies = []
    losses = []
    times = [0.0]

    if args.decay_lr:
        lambda_lr = lambda epoch: 1.0 / np.sqrt(epoch+1)
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda_lr])
    for epoch in range(1, args.epochs + 1):
        if args.decay_lr:
            scheduler.step()
        train(args, model, device, train_loader, test_loader, optimizer, epoch, [accuracies, losses, times])

    log_stats(accuracies, losses, times, args, model, device, test_loader, epoch, 'inf')


def main():
    import arguments
    args = arguments.get_args()
    launch_job(args)

if __name__ == '__main__':
    main()
