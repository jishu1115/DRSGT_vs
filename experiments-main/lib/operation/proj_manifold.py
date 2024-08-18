import numpy as np
import scipy.linalg as la

def proj_manifold(x):  # orthogonal projection onto the manifold
    u, s, vh = la.svd(x, full_matrices=False)
    return np.matmul(u, vh)  # 返回矩阵乘积