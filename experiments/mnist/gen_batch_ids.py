
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
#import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR
# from adacurv.torch.optim.hvp_utils import build_Fvp, mean_kl_multinomial

def log_stats(batch_idxs, args, model, device, test_loader, epoch, batch_idx):
    batch_idxs.append(epoch)
    # print (batch_idxs)
    dir = build_log_dir(args)
    np.save(dir+"/epoch"+str(args.epochs)+"_ids_batch"+str(args.batch_size)+".npy", np.array(batch_idxs))

def train(args, model, device, train_loader, test_loader, optimizer, epoch, datas):
    batch_idxs = datas

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx % args.log_interval == 0:
            log_stats(batch_idxs, args, model, device, test_loader, epoch, batch_idx)

        if args.verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # return accuracies

def build_log_dir(args):
    return "results/meta/"

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

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    batch_idxs = []
    for epoch in range(1, args.epochs + 1):
        train(args, None, device, train_loader, test_loader, None, epoch, batch_idxs)

    log_stats(batch_idxs, args, None, device, test_loader, epoch, 'inf')

def generate_batch_ids(epochs=10, batch_size=500):
    import arguments
    args = arguments.get_args()
    args.epochs = epochs
    args.batch_size = batch_size
    launch_job(args)

def main():
    import arguments
    args = arguments.get_args()
    launch_job(args)

if __name__ == '__main__':
    main()
