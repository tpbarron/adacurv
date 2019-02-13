
import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='Adam beta1')
parser.add_argument('--beta2', type=float, default=0.9,
                    help='Adam beta2')
parser.add_argument('--shrunk', action='store_true', default=False,
                    help='incorporate shrunk Fisher estimate')
parser.add_argument('--lanczos-iters', type=int, default=20,
                    help='number of lanczos iterations to estimate eigenvalues')
parser.add_argument('--optim', type=str, default='adam',
                    help='the optimizer to use')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-dir', type=str, default='results/',
                    help='dir to save output')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='print some extra stuff.')
parser.add_argument('--approx-adaptive', action='store_true', default=False,
                    help='Use approx linear adaptive update.')
parser.add_argument('--decay-lr', action='store_true', default=False,
                    help='Decay the LR.')
args = parser.parse_args()

def get_args():
    return args
