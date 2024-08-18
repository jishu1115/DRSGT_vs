import os
import numpy as np
from sklearn.datasets import make_spd_matrix
foldname = os.path.dirname(__file__) + '/data/'

from pymanopt.manifolds import Stiefel
n = 20
r = 5
mfd = Stiefel(n, r, retraction="polar")


mean = np.zeros(n)
covU = make_spd_matrix(n, random_state=42) # 生成一个5x5的随机稀疏对称正定矩阵
U = np.random.multivariate_normal(mean, covU, 15000)

noise = np.random.normal(0, 1, (15000, r))

np.random.seed(22)
X_0 = mfd.random_point()
np.save( foldname + 'data_X_0.npy', X_0)

np.save( foldname + 'cov_U.npy' , covU)
np.save( foldname + 'data_U.npy' , U)
np.save( foldname + 'data_E.npy' , noise)
