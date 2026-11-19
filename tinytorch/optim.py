import numpy as np


class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = (
            [np.zeros_like(param.data) for param in self.params]
            if momentum > 0.0
            else None
        )

    def step(self):
        for idx, param in enumerate(self.params):
            grad = param.grad
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * param.data

            if self.velocity is not None:
                self.velocity[idx] = self.momentum * self.velocity[idx] + grad
                grad = self.velocity[idx]

            param.data -= self.lr * grad

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


class Adam:
    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        self.params = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        self.t += 1
        for idx, param in enumerate(self.params):
            grad = param.grad
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * param.data

            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * grad * grad

            m_hat = self.m[idx] / (1.0 - self.beta1**self.t)
            v_hat = self.v[idx] / (1.0 - self.beta2**self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
