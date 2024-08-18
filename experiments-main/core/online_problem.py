import numpy as np
import os
import sys



class OnlineProblem:
    def __init__(self,mfd,data,time,alpha,beta,eta,loss,grad):

        self.mfd = mfd
        self.dim = int(mfd.dim)
        
        self.data = data
        self.time = time
        
        self.loss = loss
        self.grad = grad
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
            #def f_t(self,time,X):
        #return self.loss(self.data[time],X)

    #def g_t(self,time,X):
        #return self.grad(self.data[time],X)
    def random_g(self,X):
        data_size = self.data.shape[0]
        i = np.random.randint(1, data_size+1)
        return self.grad(self.data[i], X)

    def g_sample(self, X, sample_size):
        data_size = self.data.shape[0]
        batchsize = min(data_size, sample_size)
        row_rand_data = np.arange(data_size)
        np.random.shuffle(row_rand_data)
        row_rand = self.data[row_rand_data[0:batchsize]]
        return self.grad(row_rand, X)

    def pg_sample(self, X, sample_size):
        data_size = self.data.shape[0]
        batchsize = min(data_size, sample_size)
        row_rand_data = np.arange(data_size)
        np.random.shuffle(row_rand_data)
        Usample = self.data[row_rand_data[0:batchsize]]
        Esample = self.eta[row_rand_data[0:batchsize]]
        return self.grad(X, Usample, Esample)



    
    