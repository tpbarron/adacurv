import argparse
parser = argparse.ArgumentParser(description='Random quadratic generator')
parser.add_argument('--batch-size', type=int, default=10,
                    help='samples drawn per batch (default: 10)')
parser.add_argument('--iters', type=int, default=100,
                    help='iterations (default: 100)')
parser.add_argument('--dimension', type=int, default=100,
                    help='dimension of the quadratic (default: 100)')
parser.add_argument('--condition', type=float, default='1.0',
                    help='Approximate conditioning (default: 1.0)')
parser.add_argument('--noise', type=float, default=0.0,
                    help='noise on samples (default: 0)')
parser.add_argument('--grad-sparsity', type=float, default=0.0,
                    help='gradient sparsity, g_i is set to zero with this probability (default: 0.0)')
parser.add_argument('--rotate', action='store_true', default=False,
                    help='Rotate the matrix randomly')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-dir', type=str, default='results/',
                    help='dir to save output')
parser.add_argument('--sgd', action='store_true', default=False)
parser.add_argument('--adaptive', action='store_true', default=False)

def get_args():
    return parser.parse_args()
