"""
Compare the time to compute the Hvp and CG for Pytorch vs Jax
"""


import torch_timing
import jax_timing

def run():
    # torch_timing.run()
    jax_timing.run()

if __name__ == "__main__":
    run()
