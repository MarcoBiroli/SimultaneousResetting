# SimultaneousResetting

This repository contains the code to replicate the results obtained in the papers [Exact order, gap and counting statistics of a Brownian gas correlated by resetting](https://arxiv.org/abs/2211.00563) and [Critical number of walkers for diffusive search processes with resetting](https://arxiv.org/abs/2303.18012). The file `utils.py` contains all the methods used to simulate the stochastic processes. The file `compute_all.py` can be run to iteratively call the `compute` function define in `utils.py` with all possible parameter combinations. For example running

`bash% python compute_all.py --N 1 2 3 4 5 6 7 8 9 --sim 0 1 --r 0.1 1 10`

will run the desired computations for every combination of the specified parameters `N`, `sim` and `r`. The `environment.yml` file contains the necessary dependencies to create a conda environement. 

## Example run

A full example run can be done by running 

```
bash% git clone https://github.com/MarcoBiroli/SimultaneousResetting/
bash% cd SimultaneousResetting
bash% conda env create -f environment.yml
bash% conda activate SimultaneousResetting
SimultaneousResetting% python compute_all.py --N 1 2 3 4 5 6 7 8 9 --sim 0 1 --r 0.1 1 10
```
## Command-line arguments

All the parameters of the simulation can be set from the command-line. Here are all the possible arguments.
- repeats : The list of numbers of monte-carlo samples to generate
- N : The list of numbers of walkers to simulate
- T : The list of times up to which we will perform the stochastic process
- dt : The list of atomic time-steps within which we perform the stochastic action
- a : The list of atomic length-scales of the underlying lattice
- target : The list of positions of the target that we need to reach.
- sim : If true the resetting action will reset all particles simultaneously, otherwise the resetting will happen independently for each particle.
- folder : Directory in which to save the results.
             
