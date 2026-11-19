import random
from tinytorch.module import Module


class Dropout(Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x

        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad

        return [g * m for g, m in zip(grad, self.mask)]
