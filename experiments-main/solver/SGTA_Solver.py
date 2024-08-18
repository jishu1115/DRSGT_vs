import numpy as np
import scipy.linalg as la

class SGTASolver:
    def __init__(self) -> None:
        pass
#fenpeikongjian
    def initial_with_problem(self, T, Y_0, track_list):
        self.value_histories = np.zeros(T)
        for item in track_list:
            exec('self.' + item + ' = np.zeros( (T+1,) + Y_0.shape ) ')
        self.opt_error_histories = np.zeros(T)
        self.op_histories = np.zeros(T)


    def sub_dist(self,x, y):
        xty = np.matmul(x.T, y)
        if xty.size == 1:
            R = np.sign(xty)
            return la.norm(x - y * R)
        else:
            u, s, vh = la.svd(xty)
            R = np.matmul(u, vh)
            return la.norm(x - np.matmul(y, R.T))

