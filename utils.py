# pylint: disable=missing-module-docstring
# pylint: disable=unused-import
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=not-an-iterable
# pylint: disable=too-many-arguments

import random
import argparse
import numpy as np
import numba

def parse():
    '''
    Parse command line arguments.

    Parameters
    ----------
    None

    Command-line Arguments
    ----------------------
    repeats : positive integer
              The number of monte-carlo samples to generate
    n : positive integer
        The number of walkers to simulate
    T : positive float
        The time up to which we will perform the stochastic process
    dt : postive float
         The atomic time-step within which we perform the stochastic action
    a : positive float
        The atomic length-scale of the underlying lattice
    r : postive float
        The resetting rate at which the stochastic system resets.
    target : positive float
             The position of the target that we need to reach.
    sim : boolean
          If true the resetting action will reset all particles simultaneously,
          otherwise the resetting will happen independently for each particle.

    Returns
    -------
    args : Namespace
           The Namespace containing the above arguments.

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
    parser.add_argument('--r', type=float, default = 1,
                        help='The resetting rate at which the stochastic system can reset.')
    parser.add_argument('--target', type=float, default = 1,
                        help='The position of the target that we need to reach.')
    parser.add_argument('--sim', action='store_true')

    args = parser.parse_args()
    return args

@numba.jit(nopython = True)
def D(a, dt):
    '''
    Returns the diffusion constant from the atomic time and length scales.

    Parameters
    ----------
    args : Namespace
           The Namespace containing the command line arguments

    See Also
    --------
    parse : the function which defines the variables of args.

    Returns
    -------
    D : float
        The diffusion constant of the macroscopic problem equivalent to the
        microscopic problem.

    Examples
    --------
    >>> args = parse()
    >>> print(D(args))

    '''
    return a**2/(2 * dt)

@numba.jit(nopython = True)
def run(N, T, dt, a, target, sim, r):
    '''
    Executes one run of the stochastic process whose constants are defined
    by the command-line arguments.

    Parameters
    ----------
    N : positive integer
        The number of walkers to simulate
    T : positive float
        The time up to which we will perform the stochastic process
    dt : postive float
         The atomic time-step within which we perform the stochastic action
    a : positive float
        The atomic length-scale of the underlying lattice
    target : positive float
             The position of the target that we need to reach.
    sim : boolean
          If true the resetting action will reset all particles simultaneously,
          otherwise the resetting will happen independently for each particle.
    r : postive float
        The resetting rate at which the stochastic system resets.

    See Also
    --------
    parse : the function which defines the variables of args.

    Returns
    -------
    T : float
        The first time at which the stochastic process hits the target,
        if we never hit the target then T will be the ending time of the
        simulation.
    flag : boolean
           True if we have never hit the target, False otherwise.

    Examples
    -------
    Compute an average first passage time:

    >>> args = parse()
    >>> MFPT = np.mean( [ run(args)[0] for _ in range(args.repeats)] )

    Estimate the amount of walks which are not accounted for and
    falsify our statistics:

    >>> args = parse()
    >>> error_walks = np.sum( [ run(args)[1] for _ in range(args.repeats) ] )

    '''
    X = np.full(shape = (N,), fill_value = target, dtype = np.float64)
    curT = 0
    flag = True
    for _ in range(int(T/dt)):
        curT += dt
        if sim:
            if random.random() < r * dt:
                X = np.full(shape = (N,), fill_value = target, dtype = np.float64)
            else:
                for i, _ in enumerate(X):
                    X[i] = X[i] + np.sqrt(2 * D(a, dt) * dt) \
                        * np.random.normal()
        else:
            for k in range(N):
                if random.random() < r * dt:
                    X[k] = target
                else:
                    X[k] = X[k] + np.sqrt(2 * D(a, dt) * dt) \
                        * np.random.normal()
        if (X < 0).any():
            flag = False
            break
    if flag:
        return T, flag
    return curT, flag

@numba.jit(parallel = True, nopython = True)
def compute(repeats, N, T, dt, a, target, sim, r, ProgressProxy = None):
    '''
    Computes the Mean First Passage Time (MFPT) of N resetting stochastic processes
    to a fixed target by direct sampling.

    Parameters
    ----------
    args : Namespace
           The Namespace containing the command line arguments
    ProgressProxy : ProgressBar, optional
                    Handle to a progress bar which will be used to display the evolution
                    of the simulation.

    See Also
    --------
    repeats : positive integer
              The number of monte-carlo samples to generate
    N : positive integer
        The number of walkers to simulate
    T : positive float
        The time up to which we will perform the stochastic process
    dt : postive float
         The atomic time-step within which we perform the stochastic action
    a : positive float
        The atomic length-scale of the underlying lattice
    target : positive float
             The position of the target that we need to reach.
    sim : boolean
          If true the resetting action will reset all particles simultaneously,
          otherwise the resetting will happen independently for each particle.
    r : postive float
        The resetting rate at which the stochastic system resets.

    Returns
    -------
    mean : float
           The mean first passage time computed by the simulation.
    epsilon : float
              The estimated error coming from our sampling.
    untouched : integer
                The number of rare-event walks which have not reached the target and which
                are skewing our statistics.

    Examples
    --------
    >>> args = parse()
    >>> with ProgressBar(total=args.repeats, leave = False) as progress:
            mean, error, _ = compute(args, progress)
    >>> print(f'MFPT = {mean} +- {error}')

        '''
    res = np.empty(repeats, dtype = np.float64)
    untouched = 0
    for i in numba.prange(repeats):
        FPT, flag = run(N, T, dt, a, target, sim, r)
        res[i] = FPT
        untouched += flag
        if ProgressProxy is not None:
            ProgressProxy.update(1)
    return res.mean(), res.std() / np.sqrt(repeats), untouched
