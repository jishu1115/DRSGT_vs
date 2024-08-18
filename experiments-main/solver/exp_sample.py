import numpy as np
import torch
import bluefog.torch as bf
from .SGTA_Solver import SGTASolver
from lib.operation.proj_manifold import proj_manifold
import time


class ExpSample(SGTASolver):
    def __init__(self) -> None:
        self.solver_type = 'exp'
        self.grad_norm_his = []

    def optimize(self, problem, X_0, y_0, X_opt):
        T = problem.time
        track_list = ['X', 'F']
        self.initial_with_problem(T, X_0, track_list)
        self.X[0] = X_0
        self.F[0] = y_0
        self.gradient_solver(problem,X_0,y_0,X_opt)

    def gradient_solver(self,problem,X_0,y_0,X_opt):
        T = problem.time
        mfd = problem.mfd
        alpha = problem.alpha
        beta = problem.beta
        y = self.F[0]
        for t in range(T):
            X_t = self.X[t]
            F_t = self.F[t]
            # suffer the loss
            #value = problem.loss(problem.data, X_t)#loss value
            #self.value_histories[t] = value
            # update
            if not type(y) == np.ndarray:
                y = y.numpy()
            v = mfd.projection(X_t, y)
            if not torch.is_tensor(X_t):
                X_t = torch.tensor(X_t)
            X_t_plus_1 = mfd.retraction(X_t, mfd.projection(X_t, alpha * bf.neighbor_allreduce(X_t)) - beta * v)
            sample_size = int(0.9 ** (-t))
            F_t_plus_1 = problem.g_sample(X_t_plus_1, sample_size) # stochastic_gradient
            if not torch.is_tensor(y):
                y = torch.tensor(y)
            y = bf.neighbor_allreduce(y)+ F_t_plus_1 - F_t
            if np.isnan(X_t_plus_1).any():
                raise ValueError
            self.X[t + 1] = X_t_plus_1
            self.F[t + 1] = F_t_plus_1
            #metric_values
            X_t_plus_1_torch = torch.from_numpy(X_t_plus_1)
            X_mean = bf.allreduce(X_t_plus_1_torch)  # 算术平均
            X_hat = proj_manifold(X_mean)  # IAM

            func_X_hat = problem.loss(problem.data, X_hat)
            self.op_histories[t] = func_X_hat

            opt_error = self.sub_dist(X_hat,X_opt)
            self.opt_error_histories[t] = opt_error

            consensus_error = np.linalg.norm(X_t_plus_1 - X_hat)
            self.c_error_histories[t] = consensus_error

            grad = problem.grad(problem.data, X_hat)
            self.grad_norm_his.append(np.linalg.norm(grad))