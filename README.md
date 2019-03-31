# Implementation of Fisher-Adaptive Natural Gradient

## Package contents:

This repository contains two main components.

  * `adacurv/torch`: this module contains [PyTorch](https://pytorch.org) code for the adaptive NGD methods we study along with utilities for Fisher-vector products, CG optimization, Lanczos iteration, and line search.
  * `experiments`: this subdirectory contains the code used to run the experiments in the paper.
    * `mnist` and `svhn`: code used for the supervised training experiments.
    * `mjrl`: code adapted from an existing MuJoCo RL library, [MJRL](https://github.com/aravindr93/mjrl), to accommodate [PyBullet](https://pybullet.org/wordpress/) and our new optimizers.

    These experiments all contain a `run.py` file which was used to launch our experiments. See below for a quick-start script.


## Installation:

  1. Clone this package `git clone https://github.com/tpbarron/adacurv.git`.
  2. Add the path to this folder to your python path `export PYTHONPATH=$PYTHONPATH:/path/to/adacurv/`.
  3. Install python dependencies `pip install -r requirements.txt`.

#### Dependencies

The majority of the dependencies can be installed using the requirements file.
The code has been run using `python3.[5,6]` and has not been tested with `python2`.

## Running the example scripts:

For a faster start we include a `quick_start.py` file in each of the `mnist` and `mjrl` experiment directories that runs a single sample experiment.
Once this has finished the data can be plotted with the `plot_quick_start.py` script.

<!-- ## In-progress updates:

There is a branch named `experimental` that has some in-progress extensions. These changes include:
  * **A parallel, randomized line search**. The existing code uses a randomized search by default but does so sequentially. This improves computation time by ~W%.
  * **An approximate, parallel CG optimization**. Using the fact that random vectors are approximately orthogonal in high dimensions (and that at each step we care only for an *approximate* solution) we are able to parallelize the Fisher-vector products required by CG resulting in approximately ~X% speedup.
  * **Amortization of the cost of shrinkage over multiple time steps**. The shrinkage estimator adds overhead to each update step while computing eigenvalues. We find computing this factor once every 10 or 20 updates is sufficient. ~Y%.

Combined, these three improvements result in a Z% improvement in runtime. -->
