import numpy as np

from tinytorch.module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p harus berada di rentang [0, 1)")
        self.p = p
        self.mask = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not self.training:
            self.mask = None
            return x

        keep = 1.0 - self.p
        self.mask = (np.random.random(x.shape) < keep) / keep
        return x * self.mask

    def backward(self, grad):
        if self.mask is None:
            return grad
        return grad * self.mask
