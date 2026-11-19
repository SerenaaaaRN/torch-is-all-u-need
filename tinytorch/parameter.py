import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad.fill(0.0)

    def __repr__(self):
        return f"Parameter(shape={self.shape})"
