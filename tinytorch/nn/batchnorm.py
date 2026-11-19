import numpy as np

from tinytorch.module import Module
from tinytorch.parameter import Parameter


class BatchNorm(Module):
    def __init__(self, size, momentum=0.1, eps=1e-5):
        super().__init__()
        self.size = size
        self.gamma = Parameter(np.ones(size))
        self.beta = Parameter(np.zeros(size))
        self.running_mean = np.zeros(size)
        self.running_var = np.ones(size)
        self.momentum = momentum
        self.eps = eps
        self.x_hat = None
        self.inv_std = None
        self.batch_size = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.size:
            raise ValueError(
                f"BatchNorm({self.size}) menerima input shape (batch, {self.size}), "
                f"dapat {x.shape}"
            )

        self.batch_size = x.shape[0]
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_hat = (x - mean) * inv_std

            self.running_mean = (
                1.0 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            inv_std = 1.0 / np.sqrt(self.running_var + self.eps)
            x_hat = (x - self.running_mean) * inv_std

        self.x_hat = x_hat
        self.inv_std = inv_std
        return self.gamma.data * x_hat + self.beta.data

    def backward(self, grad):
        grad = np.asarray(grad, dtype=np.float64)
        self.gamma.grad += (grad * self.x_hat).sum(axis=0)
        self.beta.grad += grad.sum(axis=0)

        if not self.training:
            return grad * self.gamma.data * self.inv_std

        n = self.batch_size
        return (
            self.gamma.data
            * self.inv_std
            / n
            * (
                n * grad
                - grad.sum(axis=0)
                - self.x_hat * (grad * self.x_hat).sum(axis=0)
            )
        )

    def parameters(self):
        return [self.gamma, self.beta]
