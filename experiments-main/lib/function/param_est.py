import os
import sys
import numpy as np
sys.path.append('/home/yl-02/下载/experiments-main/experiment/param_est')
import  config
from pymanopt.manifolds import Stiefel
n = 20
r = 5

foldname = config.foldname
X_0 = np.load(foldname+'data_X_0.npy')
conU = np.load(foldname + 'cov_U.npy' )
E = np.eye(r)

def func (X, node):
    sum = np.trace((X - X_0).T @ conU @ (X - X_0) + E)
    return sum/node


def grad (X, node):
    N = X.shape[0]
    Egrad = conU @ (X - X_0)
    Rgrad = (np.eye(N) - X @ X.T) @ Egrad
    return Rgrad/node


def pgrad (X, U, E):
    N = X.shape[0]
    block = U.shape[0]
    d  = U @ X_0 + E
    Egrad = 2 * U.T @(U @ X - d)
    Rgrad = (np.eye(N) - X @ X.T) @ Egrad
    return Rgrad/block
