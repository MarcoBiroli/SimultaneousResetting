# pylint: disable=missing-module-docstring
# pylint: disable=unused-import
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=not-an-iterable

import argparse
from numba_progress import ProgressBar
import numpy as np
from utils import compute

def parse():
    '''
    Parse command line arguments.

    Parameters
    ----------
    None

    Command-line Arguments
    ----------------------
    repeats : integer list
              The list of numbers of monte-carlo samples to generate
    N : integer list
        The list of numbers of walkers to simulate
    T : float list
        The list of times up to which we will perform the stochastic process
    dt : float list
         The list of atomic time-steps within which we perform the stochastic action
    a : float list
        The list of atomic length-scales of the underlying lattice
    target : float list
             The list of positions of the target that we need to reach.
    sim : boolean list
          If true the resetting action will reset all particles simultaneously,
          otherwise the resetting will happen independently for each particle.
    folder : string
             Directory in which to save the results.

    Returns
    -------
    args : dict
           The dictionary containing the above arguments and their values.

    Examples
    --------
    bash% python compute_all.py --N 1 2 3 4 5 6 7 8 9 --sim 0 1 --r 0.1 1 10

    '''
    parser = argparse.ArgumentParser(description='Simulate critical walkers.')
    parser.add_argument('--repeats', type=int, nargs = '+', default = [int(4e4)],
                        help='The number of monte-carlo samples to generate')
    parser.add_argument('--N', type=int, nargs = '+', default = [4],
                        help='The number of walkers to simulate')
    parser.add_argument('--T', type=float, nargs = '+', default = [30],
                        help='The time up to which we will perform the stochastic process')
    parser.add_argument('--dt', type=float, nargs = '+', default = [1e-5],
                        help='The atomic time-step within which we perform the stochastic action')
    parser.add_argument('--a', type=float, nargs = '+', default = [1e-1 * np.sqrt(2)],
                        help='The atomic length-scale of the underlying lattice')
    parser.add_argument('--r', type=float, nargs = '+', default = [1],
                        help='The resetting rate at which the stochastic process can reset.')
    parser.add_argument('--target', type=float, nargs = '+', default = [1],
                        help='The position of the target that we need to reach.')
    parser.add_argument('--sim', type=int, nargs = '+', default = [1])
    parser.add_argument('--folder', default = './results/',
                        help='Directory of the folder in which to save the results.')

    return vars(parser.parse_args())

args = parse()
print(args)
save_folder = args['folder']
del args['folder']
args = dict(sorted(args.items(), key=lambda x: len(x[1]), reverse=False))
VARNAMES = list(args.keys())
RANGES = list(args.values())
NPARAMS = len(args)
cur_config = {}
cur_config['repeats'] = None

def save(folder, config, res):
    """
    Save results to a file.
    """
    base = ''
    for varname, varval in config.items():
        base += f'_{varname}={varval}'
    for varname, varval in res.items():
        filename = f'{varname}' + base + '.npy'
        np.save(folder + filename, np.array(varval))


def traverse(i):
    """
    Recursive function which iterates over all possible parameter combinations.

    Parameters
    ----------
    i : integer
        The recursion depth, i.e. the position in PARAMS

    Returns
    ----------
    None

    """
    if i == NPARAMS:
        with ProgressBar(total=cur_config['repeats'], leave = False) as progress:
            print(cur_config)
            return compute(**cur_config, ProgressProxy = progress)
    elif i == NPARAMS - 1:
        res = {'mean' : [], 'std' : [], 'untouched' : []}
    for val in RANGES[i]:
        cur_config[VARNAMES[i]] = val
        if i != NPARAMS - 1:
            print(VARNAMES[i])
            return traverse(i + 1)
        res['mean'], res['std'], res['untouched'] = traverse(i + 1)
        save(save_folder, cur_config, res)
    return 0

traverse(0)
