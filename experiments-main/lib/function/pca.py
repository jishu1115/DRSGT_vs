import numpy as np
from pymanopt.manifolds import stiefel
from pymanopt.manifolds.stiefel import Stiefel
import scipy.linalg as la

def func(A, X):
    # A为(block，N)矩阵
    block = A.shape[0]
    sum = - 0.5 * np.trace(A @ X @ X.T @ A.T)
    return sum/block

def norm_func(A, X):
    block = A.shape[0]
    sum = np.linalg.norm(A - A @ X @X.T) ** 2
    return sum / block

def grad(A, X):
    block = A.shape[0]
    N = X.shape[0]
    Egrad = - np.einsum('ij,ik->jk', A, A) @ X
    Rgrad = (np.eye(N) - X @ X.T) @ Egrad
    #Rgrad = np.clip(Rgrad, -10, 10)

    return Rgrad/block

def norm_grad(A, X):
    block = A.shape[0]
    N = X.shape[0]
    Egrad = -2 * (A.T @ (A - A @ X @ X.T) + (A - A @ X @ X.T).T @ A) @ X
    Rgrad = (np.eye(N) - X @ X.T) @ Egrad
    # Rgrad = np.clip(Rgrad, -10, 10)

    return Rgrad / block

def new_grad(A, X):
    block = A.shape[0]
    N = X.shape[1]
    Egrad = - np.einsum('ij,ik->jk', A, A) @ (X @ X.T @ X)
    # Rgrad = (np.eye(N) - X @ X.T) @ Egrad + 4000.0 * X @ (X.T @ X - np.eye(X.shape[1]))
    Rgrad = Egrad @ ((3 * np.eye(N) - X.T @ X) / 2) - X @ ((X.T @ Egrad) + (X.T @ Egrad).T) / 2 + 20 * X @ (X.T @ X - np.eye(N))
    return np.clip(Rgrad / block, -10, 10)


def sum_f( A , X ):
    N = A.shape[-1]
    return func(A.reshape(-1,N),X) 


def sum_grad( A , X ):
    N = A.shape[-1]
    return grad(A.reshape(-1,N),X) 

