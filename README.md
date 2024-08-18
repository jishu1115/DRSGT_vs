# Distributed Riemannian Stochastic Gradient Tracking Algorithm on the Stiefel Manifold

## Requirements

To install requirements:
scipy, numpy and matplotlib:
```
python -m pip install --user numpy scipy matplotlib
```
Pymanopt:
```
pip install --user pymanopt
```
bluefog_env

## Code structure

experiment:
- contains two experiments in the paper

lib:
- contains object functions and some operations
  
solver:
- defines DRSGD/DRSGT/variable sampling algorithms

## Instructions
1. Open the folder "experiment" to choose the experiment;
2. Open config.py to set parameters in the optimization;
3. Run realrun_er.ipynb/realrun_ring.ipynb/realrun_star.ipynb and est_er.ipynb/est_ring.ipynb/est_star.ipynb to get empirical results;
4. Run plot.py to get figures

