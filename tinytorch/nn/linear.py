import numpy as np

from tinytorch.module import Module
from tinytorch.parameter import Parameter


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = np.sqrt(2.0 / fan_in)
        self.weight = Parameter(np.random.normal(0.0, std, size=(fan_out, fan_in)))
        self.bias = Parameter(np.zeros(fan_out))
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.fan_in:
            raise ValueError(
                f"Linear({self.fan_in}, {self.fan_out}) menerima input "
                f"shape (batch, {self.fan_in}), dapat {x.shape}"
            )
        self.input = x.copy()
        return x @ self.weight.data.T + self.bias.data

    def backward(self, grad):
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != (self.input.shape[0], self.fan_out):
            raise ValueError(
                f"Gradien Linear({self.fan_in}, {self.fan_out}) harus "
                f"berbentuk ({self.input.shape[0]}, {self.fan_out}), dapat {grad.shape}"
            )
        self.weight.grad += grad.T @ self.input
        self.bias.grad += grad.sum(axis=0)
        return grad @ self.weight.data

    def parameters(self):
        return [self.weight, self.bias]
