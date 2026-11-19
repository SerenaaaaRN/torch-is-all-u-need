class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def children(self):
        return []

    def train(self):
        self.training = True
        for child in self.children():
            child.train()

    def eval(self):
        self.training = False
        for child in self.children():
            child.eval()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def __call__(self, x):
        return self.forward(x)
