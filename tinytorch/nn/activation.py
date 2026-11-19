import numpy as np

from tinytorch.module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mask = x > 0
        return np.where(self.mask, x, 0.0)

    def backward(self, grad):
        return grad * self.mask


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        output = np.empty_like(x)
        positive = x >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        output[~positive] = exp_x / (1.0 + exp_x)
        self.output = output
        return output

    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = np.tanh(np.asarray(x, dtype=np.float64))
        return self.output

    def backward(self, grad):
        return grad * (1.0 - self.output**2)


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
        self.output = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        shifted = x - x.max(axis=self.axis, keepdims=True)
        exp = np.exp(shifted)
        self.output = exp / exp.sum(axis=self.axis, keepdims=True)
        return self.output

    def backward(self, grad):
        dot = (grad * self.output).sum(axis=self.axis, keepdims=True)
        return self.output * (grad - dot)
