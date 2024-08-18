import numpy as np
import torch
import bluefog.torch as bf
from .SGTA_Solver import SGTASolver
from lib.operation.proj_manifold import proj_manifold

class DR_SGD(SGTASolver):
    def __init__(self) -> None:
        self.solver_type = 'DR_SGD'
        self.grad_norm_his = []
        self.X_hat = []
        self.oracle = []

    def optimize(self, problem, X_0, X_opt):
        T = problem.time
        track_list = ['X']
        self.initial_with_problem(T, X_0, track_list)
        self.X[0] = X_0
        self.gradient_solver(problem,X_0,X_opt)

    def gradient_solver(self,problem,X_0,X_opt):
        T = problem.time
        mfd = problem.mfd
        alpha = problem.alpha
        beta = problem.beta
        oracle = 0
        for t in range(T):
            X_t = self.X[t]

            # update
            sample_size = 5
            #v = problem.g_sample(X_t, sample_size)  # stochastic_gradient
            v = problem.pg_sample(X_t, sample_size)
            #beta_t = beta/np.sqrt(t+1)
            if not torch.is_tensor(X_t):
                X_t = torch.tensor(X_t)
            X_t_plus_1 = mfd.retraction(X_t, mfd.projection(X_t, alpha * bf.neighbor_allreduce(X_t)) - beta * v)
            if np.isnan(X_t_plus_1).any():
                raise ValueError
            self.X[t + 1] = X_t_plus_1
            #metric_values
            X_t_plus_1_torch = torch.from_numpy(X_t_plus_1)
            X_mean = bf.allreduce(X_t_plus_1_torch)  # 算术平均
            X_hat = proj_manifold(X_mean)  # IAM
            self.X_hat.append(X_hat)

            # func_X_hat = problem.loss(problem.data, X_hat)
            # self.op_histories[t] = func_X_hat

            # grad_est = problem.loss(X_hat, 15)
            # self.value_histories[t] = np.linalg.norm(grad_est)
            #
            # opt_error = self.sub_dist(X_hat,X_opt)
            # self.opt_error_histories[t] = opt_error
            #
            # consensus_error = np.linalg.norm(X_t_plus_1 - X_hat)
            # self.c_error_histories[t] = consensus_error

            # grad = problem.grad(problem.data, X_hat)
            # self.grad_norm_his.append(np.linalg.norm(grad))

            oracle = oracle + sample_size
            self.oracle.append(oracle)
