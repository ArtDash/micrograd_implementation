import numpy as np


class Module:
    """
    Base class for every layer.
    """
    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        stdv = 1 / np.sqrt(in_features)
        self.W = np.random.uniform(-stdv, stdv, size=(out_features, in_features))
        self.b = np.random.uniform(-stdv, stdv, size=out_features) if bias == True else 0

    def forward(self, inp):
        """Y = W * x + b"""
        return inp @ self.W.T + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):
    """The most simple and popular activation function"""
    def forward(self, inp):
        # Create ReLU Module
        return inp.relu()


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""
    def forward(self, inp, label):
        # Create CrossEntropy Loss Module
        # TODO: если не правильно, то применить -np.sum(label * np.log(inp), axis=0)
        loss = 0
        for label_item in label:
            loss += np.log(inp[label_item])
        return - loss
