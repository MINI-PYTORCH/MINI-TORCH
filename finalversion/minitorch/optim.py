'''
不同的学习率调整策略
'''

import numpy as np
import minitorch
from minitorch.tensor_functions import zeros


class SGD():
    def __init__(self,parameters,lr=0.01,weight_decay_l2=None):
        self.parameters=list(parameters)
        self.lr=lr
        self.weight_decay_l2=weight_decay_l2
    def step(self):
        if self.weight_decay_l2 is None:
            for p in self.parameters:
                if p.value.grad is not None:
                    p.update(p.value - self.lr * (p.value.grad))
        else:
            for p in self.parameters:
                if p.value.grad is not None:
                    p.update(p.value - self.lr * (p.value.grad)-self.weight_decay_l2*2*p.value)

                

    def zero_grad(self):
        for p in self.parameters:
            if p.value.grad is not None:
                p.value.grad *= 0.0



class Adam(SGD):
    def __init__(self, parameters, beta1=0.9, beta2=0.99, eta=1e-6,lr=0.1,weight_decay_l2=None):
        super(Adam, self).__init__(parameters)
        self.m = [0.0 for i in range(len(parameters))]
        self.v = [0.0 for i in range(len(parameters))]
        self.weight_decay_l2=weight_decay_l2

        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta

    def step(self):
        '''
        Adam optimizer stepping.
        We ignore the unbiasing process as it hurts efficiency.
        '''

        beta1 = self.beta1
        beta2 = self.beta2


        if self.weight_decay_l2 is None:
            for idx, p in enumerate(self.parameters):
                if p.value.grad is not None:
                    m = self.m[idx]
                    v = self.v[idx]
                    grad = p.value.grad
                    self.m[idx] = beta1*m + (1-beta1)*grad
                    self.v[idx] = beta2*v + (1-beta2)*(grad *grad)
                    p.update(p.value - self.lr/(self.v[idx].sqrt()) * (self.m[idx]))
        else:
            for idx, p in enumerate(self.parameters):
                if p.value.grad is not None:
                    m = self.m[idx]
                    v = self.v[idx]
                    grad = p.value.grad
                    self.m[idx] = beta1*m + (1-beta1)*grad
                    self.v[idx] = beta2*v + (1-beta2)*(grad *grad)
                    p.update(p.value - self.lr/(self.v[idx].sqrt()) * (self.m[idx])-self.weight_decay_l2*2*p.value)