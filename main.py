# pylint: disable=missing-module-docstring
# pylint: disable=unused-import
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=not-an-iterable

import random
import argparse
import numpy as np
from tqdm import tqdm
from scipy.special import erf, erfc
import numba
#from numba_progress import ProgressBar

def parse():
    '''
    Parse command line arguments. Arguments:
    --repeats: The number of monte-carlo samples to generate
    --n: The number of walkers to simulate
    --T: The time up to which we will perform the stochastic process
    --dt: The atomic time-step within which we perform the stochastic action
    --a: The atomic length-scale of the underlying lattice
    --target: The position of the target that we need to reach.
    '''
    parser = argparse.ArgumentParser(description='Simulate critical walkers.')
    parser.add_argument('--repeats', type=int, default = int(4e4),
                        help='The number of monte-carlo samples to generate')
    parser.add_argument('--N', type=int, default = 4,
                        help='The number of walkers to simulate')
    parser.add_argument('--T', type=float, default = 30,
                        help='The time up to which we will perform the stochastic process')
    parser.add_argument('--dt', type=float, default = 1e-5,
                        help='The atomic time-step within which we perform the stochastic action')
    parser.add_argument('--a', type=float, default = 1e-1 * np.sqrt(2),
                        help='The atomic length-scale of the underlying lattice')
    parser.add_argument('--target', type=float, default = 1,
                        help='The position of the target that we need to reach.')
    parser.add_argument('--sim', action='store_true')

    args = parser.parse_args()
    return args

def D(args):
    '''
    Returns the diffusion constant from the atomic time and length scales.
    '''
    return args.a**2/(2 * args.dt)

@numba.jit(nopython = True)
def run(args):
    '''
    Executes one run of the stochastic process
    '''
    X = np.full(shape = (args.N,), fill_value = args.target, dtype = np.float64)
    T = 0
    flag = True
    for _ in range(int(args.T/args.dt)):
        T += args.dt
        if args.sim:
            if random.random() < args.r * args.dt:
                X = np.full(shape = (args.N,), fill_value = args.target, dtype = np.float64)
            else:
                X = X + np.sqrt(2 * D(args) * args.dt) \
                    * np.random.normal(size = X.shape)
        else:
            for k in range(args.N):
                if random.random() < args.r * args.dt:
                    X[k] = args.target
                else:
                    X[k] = X[k] + np.sqrt(2 * D(args) * args.dt) \
                        * np.random.normal()
        if (X < 0).any():
            flag = False
            break
    if flag:
        return args.T, flag
    return T, flag

@numba.jit(parallel = True, nopython = True)
def compute(args, ProgressProxy = None):
    '''
    Computes the Mean First Passage Time (MFPT) of a $N$ resetting stochastic processes
    to a fixed target by direct sampling.
    Parameters:
    - Target: A float corresponding to the fixed spatial target.
    - Repeats: An integer corresponding to the number of samples to take.
    - ResetProb: A float between 0 and 1 corresponding to the probability to reset
    at each atomic time-step.
    - NbWalkers: An integer corresponding to the number $N$ of resetting stochastic processes.
    - Simultaneous: A boolean, if true then the resetting happens simultaneously for all
    the processes, otherwise the resetting happens independently.
    '''
    res = np.empty(args.repeats, dtype = np.float64)
    untouched = 0
    for i in numba.prange(args.repeats):
        T, flag = run(args)
        res[i] = T
        untouched += flag
        if ProgressProxy is not None:
            ProgressProxy.update(1)
    return res.mean(), res.std() / np.sqrt(args.repeats), untouched

def main():
    '''
    The main function which runs the code.
    '''
    args = parse()
    mean, std, untouched = compute(args)
    print(mean, std, untouched)
