
import argparse

parser = argparse.ArgumentParser(description='SVHN')


parser.add_argument('--optim', type=str, default='adam',
                    help='the optimizer to use')
parser.add_argument('--curv-type', type=str, default='fisher',
                    help='curvature type, one of fisher, gauss_newton (default: fisher)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--decay-lr', action='store_true', default=False,
                    help='Decay the LR.')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--cg-iters', type=int, default=10,
                    help='number of CG solver iterations (default: 10)')
parser.add_argument('--cg-residual-tol', type=float, default=1e-10,
                    help='error tolerance to terminate CG solver (default: 1e-10)')
parser.add_argument('--cg-prev-init-coef', type=float, default=0.0,
                    help='whether to initialize the CG solver with a portion of the previous direction (default: 0.0)')
parser.add_argument('--cg-precondition-empirical', action='store_true', default=False,
                    help='whether to use the diagonal empirical Fisher as a preconditioner (default: False)')
parser.add_argument('--cg-precondition-regu-coef', type=float, default=0.001,
                    help='damping on empirical Fisher preconditioner (default: 0.001)')
parser.add_argument('--cg-precondition-exp', type=float, default=0.75,
                    help='exponent on empirical Fisher preconditioner to soften extremes (default: 0.75)')

parser.add_argument('--shrinkage-method', type=str, default=None,
                    help='incorporate shrunk Fisher estimate, one of (None, cg, lanczos), (default: None)')
parser.add_argument('--lanczos-amortization', type=int, default=10,
                    help='frequency to compute shrinkage factor by Lanczos shrinkage-method=lanzcos (default: 10)')
parser.add_argument('--lanczos-iters', type=int, default=20,
                    help='number of lanczos iterations to estimate eigenvalues (default: 20)')

parser.add_argument('--beta1', type=float, default=0.9,
                    help='Adam beta1, used only in second-order methods (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.9,
                    help='Adam beta2, used only in second-order methods (default: 0.9)')
parser.add_argument('--approx-adaptive', action='store_true', default=False,
                    help='Use approx linear adaptive update (default: False)')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-dir', type=str, default='results/',
                    help='dir to save output')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='print some extra stuff.')
args = parser.parse_args()

def get_args():
    return args
