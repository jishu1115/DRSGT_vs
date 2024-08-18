import os
import sys

import numpy as np

foldname = os.path.dirname(__file__) + '/data/'
# dim,rounds and blocks
n=40
r=5
T = 1000
block=1

#stepsizes
alpha = 1
beta = 0.005
eta = 0.005

# manifold
from pymanopt.manifolds import Stiefel
mfd = Stiefel(n, r, retraction="polar")
X = np.load(foldname+'X_0.npy')
mfd.center = X

curvature_above = 2
diameter = np.pi / ( 2*np.sqrt(curvature_above) )
mfd.D = diameter

np.random.seed(42)
X_0 = mfd.random_point()
X_0 = mfd.exp( mfd.center , (0.99* diameter) * (X_0 - mfd.center)/mfd.norm(mfd.center,X_0)  )
y_0 = mfd.random_tangent_vector(X_0)
S_0 = y_0
D_0 = y_0
np.random.seed()

foldname = os.path.dirname(__file__) + '/data/'
print(foldname)