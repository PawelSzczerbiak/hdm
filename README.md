# HDM
Python scripts that calculate hot dark matter relic density in [S. Weinberg's model](https://arxiv.org/pdf/1305.1971.pdf) 
for two Goldstone bosons annihilationg into muon/pion pairs, using the method proposed by 
[Dolgov & Kainulainen](https://www.sciencedirect.com/science/article/pii/0550321393906467?via%3Dihub). 

Execution time of the sample scan: ~5 min ( i5-3210M CPU, 8GB RAM, Ubuntu 16.04 64bit).

## Content
- `do` contains scans (bash commands) to be performed
- `scan.py` executes scan for specific model parameters
- `RK4-solver.py` solves Boltzmann equation for pseudopotential, using Runge-Kutta 4th order method
- `utils.py` contains definitons of some useful functions
- `data/` contains tabulated data (relativistic degrees of freedom and their derivative)
- `scripts/` contains [Mathematica](https://www.wolfram.com/mathematica/) scripts that integrate (normalized) cross sections
- `results/` contains results (here: for the sample scan)

## Requirements
- [Python 3.x](https://www.python.org/)
- [Mathematica 10](https://www.wolfram.com/mathematica/) - crucial! If other version is used, 
please make sure to set appropriate path in Mathematica scripts.
