import pandas as pd
import numpy as np

class MyLogisticRegression():
    def __init__(self, theta, alpha=0.1, n_cycle=1000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.theta = theta

    def sigmoid_(self, x):

        new = np.array(x)
        new = new.astype('float64')
        if len(x.shape) != 0:
            for index, elem in enumerate(new):
                new[index] = 1/(1+np.exp(-elem))
        else:
            new = np.array(1/(1+np.exp(-x)))
        return new

    def add(self, x):    
        b = np.array([1 for i in range(x.shape[0])])
        new_table = np.c_[b,x]
        return new_table

    def ligistic_predict_(self, x):
        return self.sigmoid_(self.add(x)@self.theta)

    def vec_log_gradient(self, x, y):
        return (self.add(x).T @ (self.sigmoid_((self.add(x) @ self.theta))-y))/x.shape[0]

    def fit_(self, x, y):
        cycle = self.max_iter
        save = None
        while cycle > 0 or save.all() != self.theta.all():
            save = self.theta
            grad = self.vec_log_gradient(x,y)
            self.theta = self.theta - (self.alpha * (grad))
            cycle -= 1
    
    def vec_log_loss_(self, y, y_hat, eps=1e-15):
        vec_one = np.ones(y.shape[0]).reshape((-1,1))
        res = y.flatten() @ np.log(y_hat + eps).flatten() + (vec_one - y).flatten() @ np.log(1 - y_hat + eps).flatten()
        return res/-y.shape[0]
